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
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_loop_consumer.hpp"
#include "decision_log.hpp"
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
    std::string arm;       // "sombra" | "activo", obligatorio (Plan_Fase3_Daemon.md SS0.1, requisito 1)
    std::string log_path;  // ruta del registro JSONL de decisiones (requisito 2); vacio = sin registro
};

[[noreturn]] void usage_and_exit(const char* prog) {
    std::fprintf(stderr,
        "uso: %s --arm {sombra|activo} --perf-cpus 0,1,2,3 [--model xgboost_cpu.onnx] [--threshold 0.85]\n"
        "  [--target-pid PID] [--collector-cpu N] [--consumer-cpu N] [--log-path RUTA]\n"
        "  [--interval-ns 1000000] [--cpu-freq-sysfs-path RUTA]\n"
        "  [--compute-actuar --compute-freq-khz N] [--memory-actuar --memory-freq-khz N] [-v]\n"
        "--arm es obligatorio (Plan_Fase3_Daemon.md SS0.1, requisito 1): 'sombra' corre exactamente\n"
        "el mismo trabajo que 'activo' pero nunca pasa --compute-actuar/--memory-actuar de verdad al\n"
        "FrequencySetter (hoy este binario no tiene escritor nativo, ver el docstring del archivo, asi\n"
        "que en la practica ambos brazos solo registran lo que harian). --log-path activa el registro\n"
        "JSONL de decisiones (requisito 2), mismo esquema que decision_log.py del lado GPU.\n",
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
        else if (arg == "--arm") a.arm = need("--arm");
        else if (arg == "--log-path") a.log_path = need("--log-path");
        else if (arg == "-v" || arg == "--verbose") a.verbose = true;
        else if (arg == "-h" || arg == "--help") usage_and_exit(argv[0]);
        else { std::fprintf(stderr, "flag desconocida: %s\n", arg.c_str()); usage_and_exit(argv[0]); }
    }
    if (a.perf_cpus.empty()) { std::fprintf(stderr, "--perf-cpus es obligatorio\n"); usage_and_exit(argv[0]); }
    if (a.arm != "sombra" && a.arm != "activo") {
        std::fprintf(stderr, "--arm es obligatorio y debe ser 'sombra' o 'activo' (valor recibido: '%s')\n",
                      a.arm.c_str());
        usage_and_exit(argv[0]);
    }
    return a;
}

const char* label_name(hyperion::cpu_loop::CpuPhaseLabel l) {
    return l == hyperion::cpu_loop::CpuPhaseLabel::MemoryBound ? "memory_bound" : "compute_bound";
}

const char* feature_error_name(hyperion::cpu_loop::FeatureBuildError e) {
    using hyperion::cpu_loop::FeatureBuildError;
    switch (e) {
        case FeatureBuildError::kNegativeOrZeroDeltaT: return "negative_or_zero_delta_t";
        case FeatureBuildError::kZeroCycles: return "zero_cycles";
        case FeatureBuildError::kZeroInstructions: return "zero_instructions";
        case FeatureBuildError::kZeroCacheReferences: return "zero_cache_references";
        case FeatureBuildError::kNegativeCounterDelta: return "negative_counter_delta";
    }
    return "unknown";
}

// Mismo orden que cpu_feature_builder.hpp -- ver su comentario de archivo.
constexpr std::array<const char*, 6> kFeatureNames = {
    "ipc", "mpki", "cache_miss_rate", "stall_mem_ratio", "ips", "freq_khz_observed",
};

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
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

    std::optional<DecisionLogWriter> decision_log;
    if (!args.log_path.empty()) {
        decision_log.emplace(args.log_path);
    }

    uint64_t n_acted = 0, n_abstained = 0, n_feature_failed = 0;
    auto on_tick = [&](const TickResult& r) {
        DecisionRecord rec;
        rec.ts_ns = now_ns();
        rec.arm = args.arm;
        if (r.features) {
            for (size_t i = 0; i < kFeatureNames.size(); ++i) {
                rec.features.push_back({kFeatureNames[i], (*r.features)[i]});
            }
        }
        if (r.p_memory_bound) {
            rec.confidence = (*r.p_memory_bound >= 0.5f) ? *r.p_memory_bound : (1.0f - *r.p_memory_bound);
        }

        switch (r.outcome) {
            case TickOutcome::kActed:
                n_acted++;
                rec.label = label_name(r.decision->label);
                rec.policy_action = r.decision->target_freq_khz ? "actuar" : "no_actuar";
                rec.target_freq_khz = r.decision->target_freq_khz;
                // Sin escritor nativo todavia (ver docstring del archivo, punto 2): nunca se escribe
                // frecuencia de verdad, en ningun brazo -- el registro debe reflejar eso, no fingir.
                rec.written = false;
                rec.write_failed = r.decision->actuation_attempted && r.decision->actuation_failed;
                if (args.verbose) {
                    std::printf("[actuo] p_memory_bound=%.3f clase=%s target_khz=%u\n",
                                *r.p_memory_bound, label_name(r.decision->label), r.decision->target_freq_khz);
                }
                break;
            case TickOutcome::kAbstained:
                n_abstained++;
                rec.policy_action = "n/a";
                if (args.verbose) std::printf("[abstuvo] p_memory_bound=%.3f\n", *r.p_memory_bound);
                break;
            case TickOutcome::kFeatureBuildFailed:
                n_feature_failed++;
                rec.policy_action = "n/a";
                rec.error = feature_error_name(*r.feature_error);
                break;
        }
        if (decision_log) decision_log->write(rec);
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
