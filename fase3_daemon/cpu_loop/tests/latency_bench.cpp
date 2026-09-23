#include "onnx_cpu_classifier.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace hyperion::cpu_loop;

/**
 * Mide el orden de magnitud de la latencia de una sola inferencia ONNX
 * (fila a fila, no en lote -- así decide el loop real, un tick a la vez,
 * mismo patrón que export_gpu_historical_candidate.py::measure_latency).
 *
 * ADVERTENCIA -- leer antes de citar este número en cualquier parte:
 * corre en la laptop local de desarrollo, NO en paccaA100 (el nodo de
 * destino real del daemon). Sirve únicamente para una comprobación de
 * orden de magnitud ("¿esto cabe conceptualmente en un presupuesto de
 * ~1ms, o ni de lejos?") antes de invertir en integrar el camino caliente
 * completo con collector.hpp. La cifra AUTORITATIVA que decide si el loop
 * de CPU es viable en el presupuesto real debe medirse en paccaA100, igual
 * que se hizo para el candidato GPU (ver fase2_clasificador/models/
 * gpu_regresion_log_historical_20260922.metadata.json::latency_provenance
 * para el mismo principio aplicado del otro lado).
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

    std::printf("[LOCAL, NO paccaA100 -- solo orden de magnitud] n=%d p50=%.1fus p95=%.1fus p99=%.1fus\n",
                kRepeats, pctl(0.50), pctl(0.95), pctl(0.99));
    std::printf("presupuesto del tick del loop de CPU: 1000us (~1ms, granularidad de Fase 1)\n");
    return 0;
}
