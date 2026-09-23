#pragma once
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/**
 * @file
 * @brief Actuador nativo de frecuencia de CPU del daemon (Bloque C8): el
 * `FrequencySetter` real que `CpuPhaseController` recibe inyectado.
 *
 * Es un PORTE a C++ de las reglas ya probadas de `common/hpc/freqctl.py`
 * (estrategia `bounded_range`, la de intel_pstate en paccaA100), no un
 * diseño nuevo. Se portó en vez de invocar el módulo Python porque un
 * segundo proceso con intérprete, sondeando un archivo, cuenta en el RAPL
 * de paquete y por tanto dentro de la sobrecarga que el brazo *sombra* debe
 * medir. Reglas preservadas sin cambio:
 *
 *  - Orden de escritura protegido: min<=max se cumple en CADA escritura
 *    individual (el kernel lo exige por escritura, no al final). Si el
 *    nuevo techo queda por debajo del piso vigente, el piso se escribe
 *    primero; si no, el techo primero (`min_write_first`).
 *  - Toda escritura se relee y se compara; unos pocos reintentos con espera
 *    corta antes de declarar la falla (ARC-108: el HWP de intel_pstate puede
 *    rechazar transitoriamente una escritura bajo carga; un permiso ausente
 *    falla siempre, así que el reintento no lo enmascara).
 *  - Los CPU pedidos se expanden a sus hermanos SMT
 *    (`topology/thread_siblings_list`, ARC-163): un hermano sin restringir
 *    deja que el reloj físico del núcleo supere el candado.
 *  - Un único snapshot del estado original (min/max/governor por CPU y el
 *    estado de `no_turbo`) y restauración idempotente que intenta TODOS los
 *    CPU aunque uno falle, sin lanzar excepciones (puede correr desde el
 *    camino de parada por señal, sin segunda oportunidad).
 *
 * Diferencias declaradas frente a freqctl.py:
 *  - Solo `bounded_range`: `discrete_bounds` (acpi-cpufreq, felix) está
 *    descartado como plataforma y no se porta.
 *  - Recibe kHz absolutos (la política ya los trae resueltos: F0=3200000,
 *    F1=2900000), no una fracción sobre el rango.
 *  - Control de turbo incluido (freqctl.py lo delega a los sbatch): al
 *    `enter()` lo desactiva y al `restore()` lo devuelve a su valor original,
 *    vía el wrapper con permiso (`set_turbo_state`, ruta absoluta), con
 *    relectura de `no_turbo`. El argumento se escribe literal en `no_turbo`:
 *    `1` desactiva el turbo, `0` lo activa.
 *  - Falla cerrado: ante una falla de escritura/relectura durante `set_khz`
 *    el actuador restaura el estado original y queda deshabilitado; los
 *    `set_khz` siguientes devuelven false sin tocar sysfs. Sin esto, el
 *    controlador reintentaría la escritura fallida en cada ventana (~1 ms),
 *    con reintentos y esperas dentro del camino caliente.
 *
 * Límite conocido: un SIGKILL no se puede atrapar; el estado solo se
 * recupera con los caminos normales (fin del loop, SIGINT/SIGTERM que el
 * llamador convierte en parada limpia, destructor).
 */
namespace hyperion::cpu_loop {

struct CpuFreqActuatorConfig {
    std::string sysfs_cpu_root = "/sys/devices/system/cpu";
    std::vector<int> cpus;
    bool manage_turbo = true;
    /** Vacío = <sysfs_cpu_root>/intel_pstate/no_turbo. */
    std::string no_turbo_path;
    /** Comando + argumentos previos al valor ("0"/"1"). `-n`: si sudo pidiera
     * contraseña, falla en vez de quedarse esperando. */
    std::vector<std::string> turbo_command = {"sudo", "-n", "/usr/local/bin/set_turbo_state"};
    int write_verify_retries = 3;
    int write_verify_retry_delay_ms = 50;
};

namespace detail {

inline std::optional<std::string> read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
    return s.substr(start);
}

inline std::optional<long> read_long(const std::string& path) {
    auto t = read_text(path);
    if (!t || t->empty()) return std::nullopt;
    char* end = nullptr;
    long v = std::strtol(t->c_str(), &end, 10);
    if (end == t->c_str() || *end != '\0') return std::nullopt;
    return v;
}

inline bool write_text(const std::string& path, const std::string& value) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << value;
    out.flush();
    return static_cast<bool>(out);
}

/** "0,6" / "0-1,4" -> {0,6} / {0,1,4}. Entradas ilegibles se ignoran. */
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        size_t dash = part.find('-');
        try {
            if (dash == std::string::npos) {
                out.push_back(std::stoi(part));
            } else {
                int lo = std::stoi(part.substr(0, dash)), hi = std::stoi(part.substr(dash + 1));
                for (int c = lo; c <= hi; ++c) out.push_back(c);
            }
        } catch (...) {
        }
    }
    return out;
}

/** Espejo de `_write_range_safe`: true = escribir primero el piso. Ocurre
 * cuando el nuevo techo quedaría por debajo del piso vigente. */
inline bool min_write_first(std::optional<long> current_min, long target_max) {
    return current_min.has_value() && target_max < *current_min;
}

}  // namespace detail

class CpuFreqActuator {
public:
    explicit CpuFreqActuator(CpuFreqActuatorConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.no_turbo_path.empty()) cfg_.no_turbo_path = cfg_.sysfs_cpu_root + "/intel_pstate/no_turbo";
    }
    CpuFreqActuator(const CpuFreqActuator&) = delete;
    CpuFreqActuator& operator=(const CpuFreqActuator&) = delete;
    ~CpuFreqActuator() { restore(); }

    /** Lee (sin escribir) el estado original. Falla si algún CPU no expone
     * min/max o, con manage_turbo, si no se puede leer no_turbo. */
    bool snapshot() {
        cpus_ = expand_with_smt_siblings(cfg_.cpus);
        if (cpus_.empty()) return fail("sin CPUs que controlar");
        original_.clear();
        for (int cpu : cpus_) {
            Original o;
            o.min = detail::read_long(attr_path(cpu, "scaling_min_freq"));
            o.max = detail::read_long(attr_path(cpu, "scaling_max_freq"));
            o.governor = detail::read_text(attr_path(cpu, "scaling_governor"));
            if (!o.min || !o.max) return fail("cpu" + std::to_string(cpu) + " no expone scaling_min/max_freq");
            original_[cpu] = o;
        }
        if (cfg_.manage_turbo) {
            original_no_turbo_ = detail::read_text(cfg_.no_turbo_path);
            if (!original_no_turbo_ || (*original_no_turbo_ != "0" && *original_no_turbo_ != "1")) {
                return fail("no se pudo leer no_turbo en " + cfg_.no_turbo_path);
            }
        }
        snapshotted_ = true;
        return true;
    }

    /** Desactiva el turbo (si se administra) y lo verifica por relectura. */
    bool enter() {
        if (!snapshotted_) return fail("enter() antes de snapshot()");
        if (disabled_) return false;
        entered_ = true;  // desde aquí restore() debe deshacer, aunque enter falle a medias
        if (cfg_.manage_turbo) {
            if (!run_turbo_command("1")) return fail_and_restore("set_turbo_state 1 falló");
            if (!verify_no_turbo("1")) return fail_and_restore("no_turbo no quedó en 1 tras desactivarlo");
        }
        return true;
    }

    /** Fija min=max=khz en todos los CPU (con hermanos SMT). Falla cerrado. */
    bool set_khz(unsigned int khz) {
        if (disabled_ || !entered_) return false;
        long target = static_cast<long>(khz);
        auto lo = detail::read_long(attr_path(cpus_.front(), "cpuinfo_min_freq"));
        auto hi = detail::read_long(attr_path(cpus_.front(), "cpuinfo_max_freq"));
        if (lo && target < *lo) target = *lo;
        if (hi && target > *hi) target = *hi;
        for (int cpu : cpus_) {
            if (!write_range_safe(cpu, target, target)) {
                return fail_and_restore("cpu" + std::to_string(cpu) + ": escritura/relectura de rango falló");
            }
        }
        last_written_khz_ = static_cast<unsigned int>(target);
        return true;
    }

    /** Idempotente; intenta todo aunque algo falle; nunca lanza. */
    bool restore() noexcept {
        if (!entered_) return true;
        bool ok = true;
        try {
            for (const auto& [cpu, o] : original_) {
                ok = write_range_safe(cpu, *o.min, *o.max) && ok;
            }
            if (cfg_.manage_turbo && original_no_turbo_) {
                ok = run_turbo_command(*original_no_turbo_) && ok;
                ok = verify_no_turbo(*original_no_turbo_) && ok;
            }
        } catch (...) {
            ok = false;
        }
        if (ok) entered_ = false;  // si falló, un llamado posterior (destructor) reintenta
        return ok;
    }

    bool disabled() const { return disabled_; }
    bool entered() const { return entered_; }
    unsigned int last_written_khz() const { return last_written_khz_; }
    const std::string& last_error() const { return last_error_; }
    const std::vector<int>& cpus() const { return cpus_; }

private:
    struct Original {
        std::optional<long> min, max;
        std::optional<std::string> governor;
    };

    std::string attr_path(int cpu, const char* attr) const {
        std::string per_cpu = cfg_.sysfs_cpu_root + "/cpu" + std::to_string(cpu) + "/cpufreq";
        if (access(per_cpu.c_str(), F_OK) == 0) return per_cpu + "/" + attr;
        return cfg_.sysfs_cpu_root + "/cpufreq/policy" + std::to_string(cpu) + "/" + attr;
    }

    std::vector<int> expand_with_smt_siblings(const std::vector<int>& cpus) const {
        std::set<int> all(cpus.begin(), cpus.end());
        for (int cpu : cpus) {
            auto text = detail::read_text(cfg_.sysfs_cpu_root + "/cpu" + std::to_string(cpu) +
                                          "/topology/thread_siblings_list");
            if (text) for (int s : detail::parse_cpu_list(*text)) all.insert(s);
        }
        return {all.begin(), all.end()};
    }

    bool write_and_verify(const std::string& path, const std::string& value) const {
        for (int attempt = 0; attempt < cfg_.write_verify_retries; ++attempt) {
            detail::write_text(path, value);
            auto observed = detail::read_text(path);
            if (observed && *observed == value) return true;
            if (attempt + 1 < cfg_.write_verify_retries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.write_verify_retry_delay_ms));
            }
        }
        return false;
    }

    bool write_range_safe(int cpu, long target_min, long target_max) const {
        const std::string min_path = attr_path(cpu, "scaling_min_freq");
        const std::string max_path = attr_path(cpu, "scaling_max_freq");
        const std::string min_v = std::to_string(target_min), max_v = std::to_string(target_max);
        if (detail::min_write_first(detail::read_long(min_path), target_max)) {
            bool ok = write_and_verify(min_path, min_v);
            return write_and_verify(max_path, max_v) && ok;
        }
        bool ok = write_and_verify(max_path, max_v);
        return write_and_verify(min_path, min_v) && ok;
    }

    bool verify_no_turbo(const std::string& expected) const {
        for (int attempt = 0; attempt < cfg_.write_verify_retries; ++attempt) {
            auto observed = detail::read_text(cfg_.no_turbo_path);
            if (observed && *observed == expected) return true;
            if (attempt + 1 < cfg_.write_verify_retries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.write_verify_retry_delay_ms));
            }
        }
        return false;
    }

    /** fork/exec directo (sin shell): argv = turbo_command + value. */
    bool run_turbo_command(const std::string& value) const {
        if (cfg_.turbo_command.empty()) return false;
        std::vector<std::string> args = cfg_.turbo_command;
        args.push_back(value);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        pid_t pid = fork();
        if (pid < 0) return false;
        if (pid == 0) {
            execvp(argv[0], argv.data());
            _exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) return false;
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

    bool fail(const std::string& why) {
        last_error_ = why;
        std::fprintf(stderr, "cpu_freq_actuator: %s\n", why.c_str());
        return false;
    }

    bool fail_and_restore(const std::string& why) {
        fail(why);
        restore();
        disabled_ = true;
        return false;
    }

    CpuFreqActuatorConfig cfg_;
    std::vector<int> cpus_;
    std::map<int, Original> original_;
    std::optional<std::string> original_no_turbo_;
    bool snapshotted_ = false;
    bool entered_ = false;
    bool disabled_ = false;
    unsigned int last_written_khz_ = 0;
    std::string last_error_;
};

}  // namespace hyperion::cpu_loop
