#pragma once
#include <algorithm>

/**
 * @file
 * @brief Regla de decision del daemon de GPU con ABSTENCION (clasificacion selectiva): la clase se entrega solo si
 * la confianza max(P, 1-P) alcanza el umbral; si no, la fase queda en "revisar". Es la misma politica que el
 * candidato de CPU (umbral 0.85) y que la de GPU del libro (umbral elegido en el entrenamiento); hasta ahora el daemon
 * de GPU decidia siempre con P > 0.5.
 *
 * "Revisar" NO actua: el llamador libera el reloj al estado nativo (el mismo comportamiento del brazo `base`, sin
 * candado), de modo que una duda nunca puede empeorar frente al gobernador nativo. En GPU "nativo" no es un daemon:
 * es el DVFS por defecto del driver, `nvidia-smi -rgc`.
 */
namespace hyperion::gpu_loop {

enum class GpuDecision { kComputeBound, kMemoryBound, kRevisar };

inline float confidence_of(float p_memory_bound) { return std::max(p_memory_bound, 1.0f - p_memory_bound); }

inline GpuDecision decide(float p_memory_bound, float threshold) {
    if (confidence_of(p_memory_bound) < threshold) return GpuDecision::kRevisar;
    return p_memory_bound > 0.5f ? GpuDecision::kMemoryBound : GpuDecision::kComputeBound;
}

}  // namespace hyperion::gpu_loop
