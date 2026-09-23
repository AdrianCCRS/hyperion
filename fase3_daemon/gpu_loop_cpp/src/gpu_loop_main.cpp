/**
 * @file
 * @brief Punto de entrada del daemon de GPU en C++ (Bloque D): reemplaza a
 * `fase3_daemon/run_daemon.py` (Python) por decisión del usuario del
 * 2026-09-23, igual que el loop de CPU (`cpu_loop_main`). Binario aparte del
 * de CPU; se coordinan por el archivo de señal `--gpu-active-signal-path`.
 *
 * Una iteración = una muestra NVML directa cada `--poll-interval-ms` (sin
 * subprocesos `nvidia-smi`), el rastreador de actividad decide cuándo empieza y
 * termina una fase, y la CLASE de cada fase se decide UNA vez, tras
 * `--min-active-s` de actividad sostenida, con mediana/std sobre las muestras
 * de esa fase. La política llega por flags (`launch_gpu_daemon.py` traduce
 * `policy_table.yaml`): `--memory-clock-mhz`/`--compute-clock-mhz`, 0 = no
 * actuar. Brazos: `sombra` clasifica y decide igual pero NUNCA escribe;
 * `activo` escribe con `GpuClockActuator` y restaura al terminar (fin del loop,
 * SIGINT/SIGTERM, proceso objetivo terminado, destructor).
 */
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <signal.h>
#include <sys/types.h>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>

#include "decision_log.hpp"
#include "gpu_active_writer.hpp"
#include "gpu_activity_tracker.hpp"
#include "gpu_clock_actuator.hpp"
#include "gpu_window_classifier.hpp"
#include "nvml_sampler.hpp"
#include "onnx_gpu_classifier.hpp"
#include "telemetry/gpu_clock_controller.hpp"

namespace {

std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

struct Args {
    std::string arm;                 // "sombra" | "activo", obligatorio
    std::string model_path;
    std::string features_path;
    std::string gpu_index = "0";
    int target_pid = 0;              // 0 = corre hasta la señal; >0 = se detiene cuando ese proceso termina
    long long min_dwell_ns = -1;     // obligatorio: T_transición medido x10, sin default a propósito
    double min_active_s = 3.0;
    long poll_interval_ms = 50;
    double activity_threshold_pct = 5.0;
    size_t window = 200;
    double settle_s = 1.5;
    unsigned int compute_clock_mhz = 0;
    unsigned int memory_clock_mhz = 0;
    std::string log_path;
    std::string gpu_active_signal_path;
};

[[noreturn]] void usage_and_exit(const char* prog) {
    std::fprintf(stderr,
        "uso: %s --arm {sombra|activo} --model M.onnx --features F.txt --min-dwell-ns N\n"
        "  [--gpu-index 0] [--target-pid PID] [--min-active-s 3] [--poll-interval-ms 50]\n"
        "  [--activity-threshold-pct 5] [--window 200] [--settle-s 1.5]\n"
        "  [--compute-clock-mhz M] [--memory-clock-mhz M] [--log-path RUTA] [--gpu-active-signal-path RUTA]\n"
        "--min-dwell-ns no tiene default: debe salir de T_transicion_gpu MEDIDO (10 x p50). Las clases con\n"
        "reloj 0 no actuan (se libera el candado). 'sombra' nunca escribe el reloj.\n", prog);
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s requiere un valor\n", flag); usage_and_exit(argv[0]); }
            return argv[++i];
        };
        if (arg == "--arm") a.arm = need("--arm");
        else if (arg == "--model") a.model_path = need("--model");
        else if (arg == "--features") a.features_path = need("--features");
        else if (arg == "--gpu-index") a.gpu_index = need("--gpu-index");
        else if (arg == "--target-pid") a.target_pid = std::stoi(need("--target-pid"));
        else if (arg == "--min-dwell-ns") a.min_dwell_ns = std::stoll(need("--min-dwell-ns"));
        else if (arg == "--min-active-s") a.min_active_s = std::stod(need("--min-active-s"));
        else if (arg == "--poll-interval-ms") a.poll_interval_ms = std::stol(need("--poll-interval-ms"));
        else if (arg == "--activity-threshold-pct") a.activity_threshold_pct = std::stod(need("--activity-threshold-pct"));
        else if (arg == "--window") a.window = std::stoul(need("--window"));
        else if (arg == "--settle-s") a.settle_s = std::stod(need("--settle-s"));
        else if (arg == "--compute-clock-mhz") a.compute_clock_mhz = std::stoul(need("--compute-clock-mhz"));
        else if (arg == "--memory-clock-mhz") a.memory_clock_mhz = std::stoul(need("--memory-clock-mhz"));
        else if (arg == "--log-path") a.log_path = need("--log-path");
        else if (arg == "--gpu-active-signal-path") a.gpu_active_signal_path = need("--gpu-active-signal-path");
        else if (arg == "-h" || arg == "--help") usage_and_exit(argv[0]);
        else { std::fprintf(stderr, "flag desconocida: %s\n", arg.c_str()); usage_and_exit(argv[0]); }
    }
    if (a.arm != "sombra" && a.arm != "activo") {
        std::fprintf(stderr, "--arm es obligatorio y debe ser 'sombra' o 'activo' (recibido: '%s')\n", a.arm.c_str());
        usage_and_exit(argv[0]);
    }
    if (a.model_path.empty() || a.features_path.empty()) { std::fprintf(stderr, "--model y --features son obligatorios\n"); usage_and_exit(argv[0]); }
    if (a.min_dwell_ns < 0) { std::fprintf(stderr, "--min-dwell-ns es obligatorio\n"); usage_and_exit(argv[0]); }
    return a;
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool pid_alive(int pid) { return kill(pid, 0) == 0 || errno == EPERM; }

}  // namespace

int main(int argc, char** argv) {
    using namespace hyperion::gpu_loop;
    using hyperion::cpu_loop::DecisionLogWriter;
    using hyperion::cpu_loop::DecisionRecord;
    using hyperion::cpu_loop::DecisionRecordField;
    Args args = parse_args(argc, argv);

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    const auto names = read_feature_names(args.features_path);
    GpuFeatureWindow window(names, args.window);
    OnnxGpuClassifier classifier(args.model_path, names.size());

    if (args.target_pid > 0 && !pid_alive(args.target_pid)) {
        std::fprintf(stderr, "--target-pid: el proceso %d no existe\n", args.target_pid);
        return 2;
    }

    NvmlSampler nvml(static_cast<unsigned int>(std::stoul(args.gpu_index)));
    nvml.open();

    // Actuador real solo en 'activo'. En 'sombra' el setter solo registra: mismo trabajo, sin escribir.
    std::optional<GpuClockActuator> actuator;
    if (args.arm == "activo") {
        GpuClockActuatorConfig acfg;
        acfg.gpu_index = args.gpu_index;
        acfg.settle_s = args.settle_s;
        acfg.query_sm_clock_mhz = [&nvml] { return nvml.sm_clock_mhz(); };
        acfg.query_util_pct = [&nvml] { return nvml.util_pct(); };
        actuator.emplace(acfg);
    }
    telemetry::GpuClockControllerConfig ccfg{};
    ccfg.min_dwell_ns = args.min_dwell_ns;
    ccfg.compute_bound_clock_mhz = args.compute_clock_mhz;
    ccfg.memory_bound_clock_mhz = args.memory_clock_mhz;
    telemetry::GpuClockController controller(ccfg, [&](unsigned int mhz) {
        if (!actuator) {
            std::printf("[observacion] aplicaria reloj GPU -> %u MHz (brazo sombra: no se escribe)\n", mhz);
            return true;
        }
        return actuator->set_clock_mhz(mhz);
    });

    std::optional<DecisionLogWriter> decision_log;
    if (!args.log_path.empty()) decision_log.emplace(args.log_path);
    std::optional<GpuActiveWriter> active_signal;
    if (!args.gpu_active_signal_path.empty()) active_signal.emplace(args.gpu_active_signal_path);

    GpuActivityTracker tracker({args.activity_threshold_pct, static_cast<int64_t>(args.min_active_s * 1e9)});
    std::printf("gpu_loop_main: brazo=%s variables=%zu min_active=%.1fs min_dwell=%lldns memoria=%uMHz compute=%uMHz\n",
                args.arm.c_str(), names.size(), args.min_active_s, args.min_dwell_ns,
                args.memory_clock_mhz, args.compute_clock_mhz);

    uint64_t n_decisions = 0, n_phases = 0, n_read_failures = 0;
    const auto poll = std::chrono::milliseconds(args.poll_interval_ms);
    while (!g_stop.load(std::memory_order_relaxed)) {
        if (args.target_pid > 0 && !pid_alive(args.target_pid)) break;
        GpuSnapshot snap;
        if (nvml.read(snap)) {
            const int64_t now = now_ns();
            const TrackerStep st = tracker.step(snap, now);
            if (st.active_start) {
                ++n_phases;
                window.reset();  // la ventana solo contiene muestras de ESTA fase
                if (active_signal) active_signal->write(true);
            }
            window.record(snap);
            if (st.decide) {
                std::vector<float> features = window.build(snap);
                const int64_t t0 = now_ns();
                const float p_memory = classifier.predict_memory_bound_proba(features);
                const int64_t t1 = now_ns();
                const auto label = p_memory > 0.5f ? telemetry::GpuPhaseLabel::MemoryBound : telemetry::GpuPhaseLabel::ComputeBound;
                const telemetry::GpuPhaseDecision d = controller.on_phase_begin(label, static_cast<telemetry::ns_t>(now));
                const int64_t t2 = now_ns();
                ++n_decisions;
                if (decision_log) {
                    DecisionRecord rec;
                    rec.ts_ns = now;
                    rec.arm = args.arm;
                    rec.device = "gpu";
                    rec.label = label == telemetry::GpuPhaseLabel::MemoryBound ? "memory_bound" : "compute_bound";
                    rec.confidence = std::max(p_memory, 1.0f - p_memory);
                    for (size_t i = 0; i < names.size(); ++i) rec.features.push_back({names[i], features[i]});
                    rec.policy_action = d.target_clock_mhz ? "actuar" : "no_actuar";
                    rec.target_freq_khz = d.target_clock_mhz * 1000u;
                    rec.applied_freq_khz = d.applied_clock_mhz * 1000u;
                    rec.written = d.clock_changed && !d.clock_setter_failed && actuator.has_value();
                    rec.write_failed = d.clock_changed && d.clock_setter_failed;
                    rec.inference_time_ns = t1 - t0;
                    rec.actuation_time_ns = t2 - t1;
                    decision_log->write(rec);
                }
                std::printf("fase GPU: clase=%s p_memory=%.3f objetivo=%uMHz aplicado=%uMHz cambio=%d dwell_restante_ns=%lld\n",
                            label == telemetry::GpuPhaseLabel::MemoryBound ? "memory_bound" : "compute_bound", p_memory,
                            d.target_clock_mhz, d.applied_clock_mhz, (int)d.clock_changed, (long long)d.dwell_remaining_ns);
                std::fflush(stdout);
            }
            if (st.ended && active_signal) active_signal->write(false);
        } else {
            ++n_read_failures;  // sin muestra este tick: no cambia el estado, nunca se fabrica una transición
        }
        std::this_thread::sleep_for(poll);
    }

    std::printf("deteniendo...\n");
    if (active_signal) active_signal->write(false);
    bool restored = true;
    if (actuator) {
        restored = actuator->restore();
        std::printf("actuador GPU: reloj %s\n", restored ? "liberado (-rgc) y verificado por exit code" : "NO se pudo liberar");
    }
    std::printf("resumen: fases=%llu decisiones=%llu lecturas_fallidas=%llu\n",
                (unsigned long long)n_phases, (unsigned long long)n_decisions, (unsigned long long)n_read_failures);
    return restored ? 0 : 4;
}
