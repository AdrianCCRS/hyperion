/**
 * @file
 * @brief Punto de entrada real del loop de CPU (§4.3 puntos 2/5), equivalente
 * en C++ a `fase3_daemon/run_daemon.py` del lado GPU.
 *
 * Solo corre con permisos de PMU reales (`perf_event_open`) -- construir Y
 * ejecutar este binario va siempre por `sbatch` en pacca
 * (`scripts/pacca/hyp_cpu_loop_cpp_run.sbatch`), nunca en una máquina local
 * (ver memoria feedback-never-run-compute-locally). El núcleo que sí se
 * prueba sin pacca (`cpu_loop_consumer.hpp`, con muestras sintéticas
 * empujadas al ring) es el mismo que este binario invoca sin cambios.
 *
 * Decisiones de alcance tomadas aquí, explícitas, no huecos escondidos:
 *
 * 1. **No parsea `policy_table.yaml`.** Añadir un parser YAML en C++ solo
 *    para esto sería una dependencia nueva para leer 2 booleanos y como
 *    mucho 2 enteros -- la política de CPU hoy es "no_actuar" en ambas
 *    clases (medido, ver el libro), así que ese es el default seguro. Un
 *    script wrapper (Python, ya tiene `yaml`) resuelve
 *    `policy_table.yaml` y pasa los valores concretos como flags
 *    (`--compute-actuar/--compute-freq-khz`, ídem memory) el día que la
 *    política cambie a `actuar`.
 * 2. **No escribe frecuencia real.** No existe todavía un escritor nativo
 *    de `scaling_min/max_freq` con verificación por relectura (el que
 *    describe §4.3 punto 5, "reutilizando la lógica de freqctl.py pero
 *    implementada nativamente") -- construir y probar ese código sin
 *    ninguna política real en `actuar` sería código muerto sin ejercitar.
 *    El `FrequencySetter` de este binario solo registra en log lo que
 *    haría, exactamente igual que `_dry_run_setter` del lado GPU.
 * 3. **`gpu_active` siempre false.** Ver el docstring de
 *    `cpu_loop_consumer.hpp` -- no hay todavía un mecanismo de
 *    coordinación entre este proceso C++ y `run_daemon.py` (Python,
 *    proceso separado), y no hace falta uno mientras la política GPU siga
 *    bloqueada por H1.
 */
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_loop_consumer.hpp"
#include "onnx_cpu_classifier.hpp"
#include "telemetry/collector.hpp"

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

struct Args {
    std::string model_path = "xgboost_cpu.onnx";
    float threshold = 0.85f;
    pid_t target_pid = 0;
    std::vector<int> perf_cpus;
    int collector_cpu = -1;
    int consumer_cpu = -1;
    long interval_ns = 1'000'000;
    std::string cpu_freq_sysfs_path;
    bool compute_actuar = false;
    unsigned int compute_freq_khz = 0;
    bool memory_actuar = false;
    unsigned int memory_freq_khz = 0;
    bool verbose = false;
};

[[noreturn]] void usage_and_exit(const char* prog) {
    std::fprintf(stderr,
        "uso: %s --perf-cpus 0,1,2,3 [--model xgboost_cpu.onnx] [--threshold 0.85]\n"
        "  [--target-pid PID] [--collector-cpu N] [--consumer-cpu N]\n"
        "  [--interval-ns 1000000] [--cpu-freq-sysfs-path RUTA]\n"
        "  [--compute-actuar --compute-freq-khz N] [--memory-actuar --memory-freq-khz N] [-v]\n"
        "Escribe frecuencia SOLO si se pasan --compute-actuar/--memory-actuar; sin eso,\n"
        "corre en modo observacion (clasifica y decide, nunca escribe) -- la politica de\n"
        "CPU medida hoy es no_actuar en ambas clases, ver Plan_Fase3_Daemon.md.\n",
        prog);
    std::exit(2);
}

std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        out.push_back(std::stoi(s.substr(pos, comma - pos)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requiere un valor\n", flag); usage_and_exit(argv[0]); }
            return argv[++i];
        };
        if (arg == "--model") a.model_path = need("--model");
        else if (arg == "--threshold") a.threshold = std::stof(need("--threshold"));
        else if (arg == "--target-pid") a.target_pid = std::stoi(need("--target-pid"));
        else if (arg == "--perf-cpus") a.perf_cpus = parse_int_list(need("--perf-cpus"));
        else if (arg == "--collector-cpu") a.collector_cpu = std::stoi(need("--collector-cpu"));
        else if (arg == "--consumer-cpu") a.consumer_cpu = std::stoi(need("--consumer-cpu"));
        else if (arg == "--interval-ns") a.interval_ns = std::stol(need("--interval-ns"));
        else if (arg == "--cpu-freq-sysfs-path") a.cpu_freq_sysfs_path = need("--cpu-freq-sysfs-path");
        else if (arg == "--compute-actuar") a.compute_actuar = true;
        else if (arg == "--compute-freq-khz") a.compute_freq_khz = std::stoul(need("--compute-freq-khz"));
        else if (arg == "--memory-actuar") a.memory_actuar = true;
        else if (arg == "--memory-freq-khz") a.memory_freq_khz = std::stoul(need("--memory-freq-khz"));
        else if (arg == "-v" || arg == "--verbose") a.verbose = true;
        else if (arg == "-h" || arg == "--help") usage_and_exit(argv[0]);
        else { std::fprintf(stderr, "flag desconocida: %s\n", arg.c_str()); usage_and_exit(argv[0]); }
    }
    if (a.perf_cpus.empty()) { std::fprintf(stderr, "--perf-cpus es obligatorio\n"); usage_and_exit(argv[0]); }
    return a;
}

const char* label_name(hyperion::cpu_loop::CpuPhaseLabel l) {
    return l == hyperion::cpu_loop::CpuPhaseLabel::MemoryBound ? "memory_bound" : "compute_bound";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace hyperion::cpu_loop;
    Args args = parse_args(argc, argv);

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::printf("cargando clasificador: %s\n", args.model_path.c_str());
    OnnxCpuClassifier classifier(args.model_path);

    CpuPhaseControllerConfig ctrl_cfg{};
    ctrl_cfg.compute_bound = {args.compute_actuar, args.compute_freq_khz};
    ctrl_cfg.memory_bound = {args.memory_actuar, args.memory_freq_khz};
    // Sin escritor real (ver punto 2 del docstring del archivo): registra
    // en log lo que haria, nunca toca hardware. Mismo patron que
    // run_daemon.py::_dry_run_setter del lado GPU.
    CpuPhaseController controller(ctrl_cfg, [](unsigned int khz) {
        std::printf("[observacion] aplicaria %u kHz (no se escribe: sin escritor nativo, ver docstring)\n", khz);
        return true;
    });

    telemetry::CollectorConfig cfg{};
    cfg.producer_cpu = args.collector_cpu;
    cfg.interval_ns = args.interval_ns;
    cfg.enable_perf = true;
    cfg.enable_gpu = false;
    cfg.target_pid = args.target_pid;
    cfg.perf_cpus = args.perf_cpus;
    cfg.cpu_freq_sysfs_path = args.cpu_freq_sysfs_path;

    telemetry::Collector::Ring ring;
    telemetry::Collector collector(cfg, ring);

    std::printf("iniciando collector: target_pid=%d perf_cpus=%zu interval_ns=%ld\n",
                args.target_pid, args.perf_cpus.size(), args.interval_ns);
    collector.start();
    std::printf("collector activo. has_stalled_cycles_mem_any=%d\n", collector.has_stalled_cycles_mem_any());

    uint64_t n_acted = 0, n_abstained = 0, n_feature_failed = 0;
    auto on_tick = [&](const TickResult& r) {
        switch (r.outcome) {
            case TickOutcome::kActed:
                n_acted++;
                if (args.verbose) {
                    std::printf("[actuo] p_memory_bound=%.3f clase=%s target_khz=%u\n",
                                *r.p_memory_bound, label_name(r.decision->label), r.decision->target_freq_khz);
                }
                break;
            case TickOutcome::kAbstained:
                n_abstained++;
                if (args.verbose) std::printf("[abstuvo] p_memory_bound=%.3f\n", *r.p_memory_bound);
                break;
            case TickOutcome::kFeatureBuildFailed:
                n_feature_failed++;
                break;
        }
    };

    run_consumer_loop(
        ring, g_stop,
        [&classifier](const FeatureVector& f) { return classifier.predict_memory_bound_proba(f); },
        args.threshold, []{ return false; }, controller, on_tick);

    std::printf("deteniendo collector...\n");
    collector.stop();
    std::printf("resumen: actuo=%lu abstuvo=%lu features_fallidas=%lu push_retries=%lu\n",
                (unsigned long)n_acted, (unsigned long)n_abstained, (unsigned long)n_feature_failed,
                (unsigned long)collector.push_retries());
    return 0;
}
