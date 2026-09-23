// Sonda de paridad (Bloque C8): aplica una secuencia de niveles con el
// actuador C++ sobre un sysfs simulado e imprime el estado de cada CPU tras
// cada paso y tras restaurar. `tests/test_freqctl_parity.py` corre la misma
// secuencia con common/hpc/freqctl.py sobre un arbol gemelo y compara.
// El turbo no participa: freqctl.py no lo administra.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu_freq_actuator.hpp"

using namespace hyperion::cpu_loop;

static void dump(const CpuFreqActuator& a, const std::string& root, const std::string& label) {
    for (int cpu : a.cpus()) {
        std::string d = root + "/cpu" + std::to_string(cpu) + "/cpufreq/";
        auto mn = detail::read_text(d + "scaling_min_freq");
        auto mx = detail::read_text(d + "scaling_max_freq");
        std::printf("%s cpu%d min=%s max=%s\n", label.c_str(), cpu, mn ? mn->c_str() : "?", mx ? mx->c_str() : "?");
    }
}

int main(int argc, char** argv) {
    CpuFreqActuatorConfig cfg;
    cfg.manage_turbo = false;
    cfg.write_verify_retry_delay_ms = 1;
    std::vector<unsigned> seq;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--root") cfg.sysfs_cpu_root = v;
        else if (k == "--cpus") cfg.cpus = detail::parse_cpu_list(v);
        else if (k == "--khz-seq") for (int x : detail::parse_cpu_list(v)) seq.push_back(static_cast<unsigned>(x));
    }
    CpuFreqActuator a(cfg);
    if (!a.snapshot() || !a.enter()) return 2;
    int n = 0;
    for (unsigned khz : seq) {
        if (!a.set_khz(khz)) return 3;
        dump(a, cfg.sysfs_cpu_root, "paso" + std::to_string(++n));
    }
    if (!a.restore()) return 4;
    dump(a, cfg.sysfs_cpu_root, "restaurado");
    return 0;
}
