#pragma once
#include <array>
#include <cstdint>
#include <optional>

#include "telemetry/metrics.hpp"

/**
 * @file
 * @brief Construye el vector de 6 variables del candidato CPU (§4.3 punto 5
 * del plan de realineación) a partir de dos `telemetry::CpuSample`
 * consecutivas del mismo `collector.hpp` que ya usa Fase 1.
 *
 * Reproduce EXACTAMENTE las fórmulas de `fase1_telemetria/postprocess.py`
 * (líneas ~700-712, la ruta real que generó el conjunto de entrenamiento),
 * no una aproximación nueva -- un desajuste aquí invalidaría en silencio
 * toda inferencia del loop de CPU, sin que ningún test unitario aislado
 * de `CpuPhaseController` lo detectara.
 *
 *   ips              = delta_instructions / (delta_t_ns / 1e9)
 *   ipc              = delta_instructions / delta_cycles
 *   mpki             = delta_cache_misses / delta_instructions * 1000
 *   cache_miss_rate  = delta_cache_misses / delta_cache_references
 *   stall_mem_ratio  = delta_stalled_cycles_mem_any / delta_cycles
 *   freq_khz_observed = CpuSample::scaling_cur_freq_khz de la muestra ACTUAL
 *                        (lectura en vivo del mismo tick, no un delta -- ver
 *                        el comentario de ARC-135 en metrics.hpp)
 *
 * Orden del vector de salida: el mismo que
 * `fase2_clasificador/models/xgboost_cpu.metadata.json::features`
 * ["ipc", "mpki", "cache_miss_rate", "stall_mem_ratio", "ips",
 * "freq_khz_observed"] -- fijo por convención de este módulo, no leído del
 * JSON en el camino caliente (evita I/O de archivo por tick).
 */
namespace hyperion::cpu_loop {

using FeatureVector = std::array<float, 6>;

/** Por qué no se pudo construir el vector esta ventana -- el llamador
 * decide qué hacer (típicamente: no actuar este tick, no inventar un
 * valor), igual que collector.hpp trata "no medido" como distinto de "0
 * real" en cada uno de sus lectores. */
enum class FeatureBuildError : uint8_t {
    kNegativeOrZeroDeltaT,     // delta_t_ns <= 0: muestras fuera de orden o reloj degradado
    kZeroCycles,               // delta_cycles == 0: ipc/stall_mem_ratio indefinidos
    kZeroInstructions,         // delta_instructions == 0: mpki indefinido
    kZeroCacheReferences,      // delta_cache_references == 0: cache_miss_rate indefinido
    kNegativeCounterDelta,     // un contador retrocedió -- overflow/reset entre muestras, contador degradado
};

/** Resultado de un intento de construcción: o el vector, o por qué falló. */
struct FeatureBuildResult {
    std::optional<FeatureVector> features;
    std::optional<FeatureBuildError> error;

    bool ok() const noexcept { return features.has_value(); }
};

/**
 * @brief Calcula el vector de 6 variables entre dos muestras CPU
 * consecutivas del mismo `collector.hpp`.
 *
 * @param prev muestra anterior (contadores acumulados hasta ese tick).
 * @param curr muestra actual.
 * @param delta_t_ns duración real entre ambas (nunca el intervalo nominal
 *        configurado -- mismo principio POST-04 que postprocess.py: usar
 *        el delta de tiempo REAL, no el declarado).
 */
inline FeatureBuildResult build_cpu_features(
    const telemetry::CpuSample& prev, const telemetry::CpuSample& curr, int64_t delta_t_ns
) {
    if (delta_t_ns <= 0) {
        return {std::nullopt, FeatureBuildError::kNegativeOrZeroDeltaT};
    }

    const auto delta = [](uint64_t a, uint64_t b) -> std::optional<int64_t> {
        if (b < a) return std::nullopt;  // contador retrocedio: overflow o reset, no restar en silencio
        return static_cast<int64_t>(b - a);
    };

    const auto d_instructions = delta(prev.instructions, curr.instructions);
    const auto d_cycles = delta(prev.cycles, curr.cycles);
    const auto d_cache_references = delta(prev.cache_references, curr.cache_references);
    const auto d_cache_misses = delta(prev.cache_misses, curr.cache_misses);
    const auto d_stalled_mem = delta(prev.stalled_cycles_mem_any, curr.stalled_cycles_mem_any);

    if (!d_instructions || !d_cycles || !d_cache_references || !d_cache_misses || !d_stalled_mem) {
        return {std::nullopt, FeatureBuildError::kNegativeCounterDelta};
    }
    if (*d_cycles == 0) {
        return {std::nullopt, FeatureBuildError::kZeroCycles};
    }
    if (*d_instructions == 0) {
        return {std::nullopt, FeatureBuildError::kZeroInstructions};
    }
    if (*d_cache_references == 0) {
        return {std::nullopt, FeatureBuildError::kZeroCacheReferences};
    }

    const double dt_s = static_cast<double>(delta_t_ns) / 1e9;
    const double ips = static_cast<double>(*d_instructions) / dt_s;
    const double ipc = static_cast<double>(*d_instructions) / static_cast<double>(*d_cycles);
    const double mpki = static_cast<double>(*d_cache_misses) / static_cast<double>(*d_instructions) * 1000.0;
    const double cache_miss_rate = static_cast<double>(*d_cache_misses) / static_cast<double>(*d_cache_references);
    const double stall_mem_ratio = static_cast<double>(*d_stalled_mem) / static_cast<double>(*d_cycles);
    const double freq_khz_observed = static_cast<double>(curr.scaling_cur_freq_khz);

    // Orden fijo: ipc, mpki, cache_miss_rate, stall_mem_ratio, ips, freq_khz_observed
    // (features_in_order de fase3_daemon/cpu_loop/xgboost_cpu.onnx.metadata.json).
    FeatureVector out{
        static_cast<float>(ipc), static_cast<float>(mpki), static_cast<float>(cache_miss_rate),
        static_cast<float>(stall_mem_ratio), static_cast<float>(ips), static_cast<float>(freq_khz_observed),
    };
    return {out, std::nullopt};
}

}  // namespace hyperion::cpu_loop
