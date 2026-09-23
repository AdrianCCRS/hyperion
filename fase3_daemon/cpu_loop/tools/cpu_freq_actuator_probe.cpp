// Sonda del actuador de frecuencia (Bloque C8). Dos usos:
//
//  1. Paridad (por defecto): aplica una secuencia de niveles sobre un sysfs
//     simulado e imprime el estado de cada CPU tras cada paso y tras
//     restaurar. `tests/test_freqctl_parity.py` corre la misma secuencia con
//     common/hpc/freqctl.py sobre un arbol gemelo y compara. Sin turbo
//     (freqctl.py no lo administra). SOLO stdout lleva el estado.
//  2. Preflight real (`--manage-turbo 1 --iterations N [--settle-cpu C]`):
//     contra el sysfs REAL, alterna los niveles de --khz-seq N veces y mide,
//     en stderr, la latencia de enter()/set_khz()/restore() y, si se pide, el
//     tiempo hasta que scaling_cur_freq del CPU C llega a +-5% del objetivo
//     (requiere carga en ese CPU: sin carga cur_freq no refleja el limite).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpu_freq_actuator.hpp"

using namespace hyperion::cpu_loop;
using Clock = std::chrono::steady_clock;

static long ns_since(Clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
}

static void dump(const CpuFreqActuator& a, const std::string& root, const std::string& label) {
    for (int cpu : a.cpus()) {
        std::string d = root + "/cpu" + std::to_string(cpu) + "/cpufreq/";
        auto mn = detail::read_text(d + "scaling_min_freq");
        auto mx = detail::read_text(d + "scaling_max_freq");
        std::printf("%s cpu%d min=%s max=%s\n", label.c_str(), cpu, mn ? mn->c_str() : "?", mx ? mx->c_str() : "?");
    }
}

static void stats(const char* name, std::vector<long> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) { return v[std::min(v.size() - 1, static_cast<size_t>(p * v.size()))]; };
    std::fprintf(stderr, "latencia %s: n=%zu p50=%.1fus p95=%.1fus max=%.1fus\n", name, v.size(),
                 pct(0.50) / 1e3, pct(0.95) / 1e3, v.back() / 1e3);
}

int main(int argc, char** argv) {
    CpuFreqActuatorConfig cfg;
    cfg.manage_turbo = false;
    cfg.write_verify_retry_delay_ms = 1;
    std::vector<unsigned> seq;
    int iterations = 1, settle_cpu = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--root") cfg.sysfs_cpu_root = v;
        else if (k == "--cpus") cfg.cpus = detail::parse_cpu_list(v);
        else if (k == "--khz-seq") for (int x : detail::parse_cpu_list(v)) seq.push_back(static_cast<unsigned>(x));
        else if (k == "--manage-turbo") cfg.manage_turbo = (v == "1");
        else if (k == "--iterations") iterations = std::stoi(v);
        else if (k == "--settle-cpu") settle_cpu = std::stoi(v);
    }
    CpuFreqActuator a(cfg);
    auto t0 = Clock::now();
    if (!a.snapshot()) return 2;
    const long snapshot_ns = ns_since(t0);
    t0 = Clock::now();
    if (!a.enter()) return 2;
    const long enter_ns = ns_since(t0);

    std::vector<long> set_ns, settle_ns;
    int n = 0;
    for (int it = 0; it < iterations; ++it) {
        for (unsigned khz : seq) {
            t0 = Clock::now();
            if (!a.set_khz(khz)) return 3;
            set_ns.push_back(ns_since(t0));
            if (settle_cpu >= 0) {
                const std::string cur = cfg.sysfs_cpu_root + "/cpu" + std::to_string(settle_cpu) + "/cpufreq/scaling_cur_freq";
                long settled = -1;
                while (ns_since(t0) < 500'000'000L) {
                    auto c = detail::read_long(cur);
                    if (c && std::labs(*c - static_cast<long>(khz)) <= static_cast<long>(khz) / 20) { settled = ns_since(t0); break; }
                    usleep(1000);
                }
                settle_ns.push_back(settled);  // -1 = no se asento en 500 ms
            }
            if (iterations == 1) dump(a, cfg.sysfs_cpu_root, "paso" + std::to_string(++n));
        }
    }
    t0 = Clock::now();
    if (!a.restore()) return 4;
    const long restore_ns = ns_since(t0);
    if (iterations == 1) dump(a, cfg.sysfs_cpu_root, "restaurado");

    std::fprintf(stderr, "latencia snapshot=%.1fus enter(turbo off)=%.1fms restore(turbo on)=%.1fms\n",
                 snapshot_ns / 1e3, enter_ns / 1e6, restore_ns / 1e6);
    stats("set_khz", set_ns);
    if (!settle_ns.empty()) {
        long unsettled = std::count(settle_ns.begin(), settle_ns.end(), -1L);
        std::vector<long> ok;
        for (long s : settle_ns) if (s >= 0) ok.push_back(s);
        std::fprintf(stderr, "asentamiento de cur_freq (+-5%%): %zu/%zu asentaron en <500ms, %ld no\n",
                     ok.size(), settle_ns.size(), unsettled);
        stats("asentamiento", ok);
    }
    return 0;
}
