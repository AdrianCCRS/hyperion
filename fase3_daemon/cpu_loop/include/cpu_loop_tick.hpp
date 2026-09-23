#pragma once
#include <functional>
#include <optional>

#include "cpu_feature_builder.hpp"
#include "cpu_phase_controller.hpp"

/**
 * @file
 * @brief Une, en un solo tick (§4.3 punto 5), construcción de variables ->
 * inferencia -> umbral de decisión selectiva -> CpuPhaseController.
 *
 * `PredictProbaFn` es inyectable a propósito (mismo principio que
 * `classify_fn` en gpu_loop.py): esta función no asume `OnnxCpuClassifier`
 * concreto, para poder probarse sin cargar un modelo ni un runtime real.
 * En producción, el llamador pasa
 * `[&](const FeatureVector& f){ return clf.predict_memory_bound_proba(f); }`.
 *
 * Umbral de decisión selectiva: la misma ecuación de §metodologia-lofo-cpu
 * del libro (q = max(p, 1-p), memory_bound si p>=0.5 y q>=tau, compute_bound
 * si p<0.5 y q>=tau, "revisar" si q<tau). threshold=0.85, el umbral
 * congelado del candidato (xgboost_cpu.metadata.json::decision.threshold).
 *
 * Interpretación de "revisar" en el daemon (decisión de diseño de este
 * módulo, no heredada de Fase 2 -- Fase 2 solo definió la salida
 * selectiva, no qué hace un daemon de control con ella): NO se llama a
 * `CpuPhaseController::on_window()` ese tick. La frecuencia queda en lo
 * último efectivamente aplicado, ni se actúa con una clase de baja
 * confianza ni se apaga el control. Coherente con la conclusión del libro
 * (§discusión CPU): "la utilidad del clasificador reside en decidir
 * cuándo abstenerse de actuar, más que en repartir frecuencias por
 * clase". Calibrar esa abstención en línea (medir la etiqueta física de
 * una fracción de ventanas en vez de solo abstenerse) es trabajo futuro
 * explícito del propio libro, no de este tick.
 */
namespace hyperion::cpu_loop {

using PredictProbaFn = std::function<float(const FeatureVector&)>;

enum class TickOutcome : uint8_t { kActed, kAbstained, kFeatureBuildFailed };

struct TickResult {
    TickOutcome outcome;
    std::optional<FeatureBuildError> feature_error;  // solo si outcome == kFeatureBuildFailed
    std::optional<float> p_memory_bound;              // solo si se llegó a inferir
    std::optional<CpuWindowDecision> decision;         // solo si outcome == kActed
};

/**
 * @brief Ejecuta un tick completo. No lee hardware ni un modelo concreto
 * por sí sola -- todo lo recibe ya resuelto o inyectado, para poder
 * probarse con datos sintéticos (ver tests/test_cpu_loop_tick.cpp).
 */
inline TickResult run_cpu_tick(
    const telemetry::CpuSample& prev, const telemetry::CpuSample& curr, int64_t delta_t_ns,
    const PredictProbaFn& predict_proba, float threshold, bool gpu_active,
    CpuPhaseController& controller
) {
    auto built = build_cpu_features(prev, curr, delta_t_ns);
    if (!built.ok()) {
        return {TickOutcome::kFeatureBuildFailed, built.error, std::nullopt, std::nullopt};
    }

    const float p_memory_bound = predict_proba(*built.features);
    const float confidence = (p_memory_bound >= 0.5f) ? p_memory_bound : (1.0f - p_memory_bound);
    if (confidence < threshold) {
        return {TickOutcome::kAbstained, std::nullopt, p_memory_bound, std::nullopt};
    }

    const CpuPhaseLabel label =
        (p_memory_bound >= 0.5f) ? CpuPhaseLabel::MemoryBound : CpuPhaseLabel::ComputeBound;
    CpuWindowDecision decision = controller.on_window(label, gpu_active);
    return {TickOutcome::kActed, std::nullopt, p_memory_bound, decision};
}

}  // namespace hyperion::cpu_loop
