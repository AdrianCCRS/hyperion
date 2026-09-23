#include "onnx_cpu_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace hyperion::cpu_loop;

/**
 * Mide la latencia de una sola inferencia ONNX (fila a fila, no en lote --
 * así decide el loop real, un tick a la vez, mismo patrón que
 * export_gpu_historical_candidate.py::measure_latency).
 *
 * Debe correr SIEMPRE vía sbatch en paccaA100 (ver
 * scripts/pacca/hyp_cpu_loop_cpp_build_test.sbatch), el nodo de destino
 * real del daemon -- nunca en la laptop local (feedback-never-run-compute-locally).
 * Resultado real medido ahí (job 7562, 2026-09-22, cpu0 taskset,
 * gobernador performance, turbo activo): p50=16.4us, p95=18.4us,
 * p99=19.3us contra un presupuesto de ~1000us -- ~2%. No es una condición
 * de reloj fijo/turbo-desactivado (no hace falta para esta comprobación:
 * no se compara entre niveles de frecuencia, solo si la inferencia cabe
 * en el presupuesto bajo el estado nativo del nodo).
 */
int main() {
    OnnxCpuClassifier classifier("xgboost_cpu.onnx");
    FeatureVector features{0.8f, 15.0f, 0.3f, 0.4f, 2e9f, 2000000.0f};

    constexpr int kWarmup = 50;
    constexpr int kRepeats = 2000;
    for (int i = 0; i < kWarmup; ++i) {
        classifier.predict_memory_bound_proba(features);
    }

    std::vector<double> timings_us;
    timings_us.reserve(kRepeats);
    for (int i = 0; i < kRepeats; ++i) {
        auto start = std::chrono::steady_clock::now();
        classifier.predict_memory_bound_proba(features);
        auto end = std::chrono::steady_clock::now();
        timings_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(timings_us.begin(), timings_us.end());
    auto pctl = [&](double p) { return timings_us[static_cast<size_t>(p * (timings_us.size() - 1))]; };

    std::printf("[paccaA100] n=%d p50=%.1fus p95=%.1fus p99=%.1fus\n",
                kRepeats, pctl(0.50), pctl(0.95), pctl(0.99));
    std::printf("presupuesto del tick del loop de CPU: 1000us (~1ms, granularidad de Fase 1)\n");
    return 0;
}
