#pragma once
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

/**
 * @file
 * @brief Actuador del reloj de GPU (`nvidia-smi -lgc/-rgc`) para el daemon de
 * GPU en C++. Puerto de las reglas de `common/hpc/gpu_freqctl.py`
 * (`apply_gpu_frequency`, `restore_gpu_state`), no un diseño nuevo.
 *
 * Por qué sigue siendo un subproceso: `nvmlDeviceSetGpuLockedClocks` exige
 * root en este driver (ARC-62/104); paccaA100 delega ese permiso vía `sudo`
 * restringido a `nvidia-smi`. Solo la ESCRITURA es subproceso; la lectura
 * (muestreo, relectura del reloj) va por NVML directo, en el mismo proceso.
 *
 * Reglas preservadas:
 *  - `mhz == 0` significa "no fijar el reloj" (política no_actuar): aplica
 *    `-rgc` (idempotente), que además deshace un candado de una fase anterior.
 *  - Tras `-lgc t,t` se espera `settle_s` y se relee el reloj SM. ARC-112: un
 *    observado POR ENCIMA del techo solo cuenta como candado no aplicado si la
 *    GPU tiene carga (util > 0); con la GPU ociosa el reloj cae a reposo y el
 *    exceso es inconcluyente.
 *  - `restore()` es `-rgc` incondicional, idempotente, no lanza.
 * Diferencia declarada: si la relectura contradice el candado (ARC-112), se
 * restaura antes de devolver false (Python devolvía false dejando el candado
 * puesto, un estado que el controlador no cree tener).
 *
 * `settle_s` por defecto 1.5 s es el valor validado de gpu_freqctl.py. Medido
 * (job 7599): el comando cuesta ~50 ms y el reloj observado llega ~70-80 ms
 * después; la espera domina el costo por cambio, y bajarla (p.ej. 0.3 s) es
 * decisión explícita del llamador.
 */
namespace hyperion::gpu_loop {

struct GpuClockActuatorConfig {
    std::string gpu_index = "0";
    /** Comando previo a los argumentos de nvidia-smi. `-n`: si sudo pidiera contraseña, falla. */
    std::vector<std::string> nvidia_smi = {"sudo", "-n", "nvidia-smi"};
    double settle_s = 1.5;
    std::function<std::optional<int>()> query_sm_clock_mhz;  // opcional; sin él no hay relectura
    std::function<std::optional<int>()> query_util_pct;
};

class GpuClockActuator {
public:
    explicit GpuClockActuator(GpuClockActuatorConfig cfg) : cfg_(std::move(cfg)) {}
    GpuClockActuator(const GpuClockActuator&) = delete;
    GpuClockActuator& operator=(const GpuClockActuator&) = delete;
    ~GpuClockActuator() { restore(); }

    bool set_clock_mhz(unsigned int mhz) {
        if (mhz == 0) {
            if (!run({"-rgc"})) return fail("nvidia-smi -rgc falló");
            dirty_ = false;
            return true;
        }
        const std::string t = std::to_string(mhz);
        if (!run({"-lgc", t + "," + t})) return fail("nvidia-smi -lgc " + t + " falló");
        dirty_ = true;
        std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.settle_s));
        if (cfg_.query_sm_clock_mhz) {
            const auto observed = cfg_.query_sm_clock_mhz();
            const auto util = cfg_.query_util_pct ? cfg_.query_util_pct() : std::nullopt;
            if (observed && *observed > static_cast<int>(mhz) && util && *util > 0) {
                restore();
                return fail("relectura " + std::to_string(*observed) + " MHz supera el techo " + t +
                            " con la GPU bajo carga (util=" + std::to_string(*util) + "%): el candado no se aplicó");
            }
            last_observed_mhz_ = observed;
        }
        return true;
    }

    /** `-rgc` incondicional cuando hay (o pudo haber) un candado puesto. Idempotente, no lanza. */
    bool restore() noexcept {
        if (!dirty_) return true;
        try {
            if (run({"-rgc"})) { dirty_ = false; return true; }
        } catch (...) {
        }
        return false;
    }

    bool dirty() const { return dirty_; }
    const std::string& last_error() const { return last_error_; }
    std::optional<int> last_observed_mhz() const { return last_observed_mhz_; }

private:
    bool run(const std::vector<std::string>& args) const {
        std::vector<std::string> all = cfg_.nvidia_smi;
        all.push_back("-i");
        all.push_back(cfg_.gpu_index);
        all.insert(all.end(), args.begin(), args.end());
        std::vector<char*> argv;
        for (auto& a : all) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        const pid_t pid = fork();
        if (pid < 0) return false;
        if (pid == 0) {
            const int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) dup2(devnull, STDOUT_FILENO);  // el mensaje de confirmación de nvidia-smi es ruido
            execvp(argv[0], argv.data());
            _exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) return false;
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    bool fail(const std::string& why) {
        last_error_ = why;
        std::fprintf(stderr, "gpu_clock_actuator: %s\n", why.c_str());
        return false;
    }

    GpuClockActuatorConfig cfg_;
    bool dirty_ = false;
    std::string last_error_;
    std::optional<int> last_observed_mhz_;
};

}  // namespace hyperion::gpu_loop
