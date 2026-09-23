#include "cpu_feature_builder.hpp"

#include <cmath>

using namespace hyperion::cpu_loop;
using telemetry::CpuSample;

namespace {

CpuSample make_sample(uint64_t instructions, uint64_t cycles, uint64_t cache_refs,
                       uint64_t cache_misses, uint64_t stalled_mem, uint64_t freq_khz) {
    CpuSample s{};
    s.instructions = instructions;
    s.cycles = cycles;
    s.cache_references = cache_refs;
    s.cache_misses = cache_misses;
    s.stalled_cycles_mem_any = stalled_mem;
    s.scaling_cur_freq_khz = freq_khz;
    return s;
}

bool close(float a, double b, double tol = 1e-4) {
    return std::abs(static_cast<double>(a) - b) < tol;
}

}  // namespace

int main() {
    // Caso base: deltas limpios, 1ms real.
    {
        auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 3'200'000);
        auto curr = make_sample(2'500'000, 3'800'000, 900'000, 130'000, 350'000, 2'900'000);
        auto result = build_cpu_features(prev, curr, /*delta_t_ns=*/1'000'000);
        if (!result.ok()) return 1;
        const auto& f = *result.features;

        const double d_instr = 1'500'000.0, d_cyc = 1'800'000.0, d_refs = 400'000.0, d_miss = 80'000.0, d_stall = 150'000.0;
        const double ipc = d_instr / d_cyc;
        const double mpki = d_miss / d_instr * 1000.0;
        const double cache_miss_rate = d_miss / d_refs;
        const double stall_mem_ratio = d_stall / d_cyc;
        const double ips = d_instr / (1'000'000.0 / 1e9);

        if (!close(f[0], ipc)) return 2;
        if (!close(f[1], mpki)) return 3;
        if (!close(f[2], cache_miss_rate)) return 4;
        if (!close(f[3], stall_mem_ratio)) return 5;
        if (!close(f[4], ips, /*tol=*/1e6)) return 6;  // ips es del orden de 1e9, tolerancia relativa mayor
        if (f[5] != 2'900'000.0f) return 7;  // freq_khz_observed = curr.scaling_cur_freq_khz, sin transformar
    }

    // delta_t_ns <= 0: fuera de orden, debe fallar explícitamente.
    {
        auto prev = make_sample(1, 1, 1, 1, 1, 1);
        auto curr = make_sample(2, 2, 2, 2, 2, 1);
        auto result = build_cpu_features(prev, curr, /*delta_t_ns=*/0);
        if (result.ok()) return 8;
        if (result.error != FeatureBuildError::kNegativeOrZeroDeltaT) return 9;
    }

    // Contador que retrocede (overflow/reset entre muestras): nunca restar en silencio.
    {
        auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 1);
        auto curr = make_sample(500'000 /* retrocede */, 3'000'000, 600'000, 60'000, 210'000, 1);
        auto result = build_cpu_features(prev, curr, 1'000'000);
        if (result.ok()) return 10;
        if (result.error != FeatureBuildError::kNegativeCounterDelta) return 11;
    }

    // delta_cycles == 0: ipc/stall_mem_ratio indefinidos, debe fallar (no dividir por cero en silencio).
    {
        auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 1);
        auto curr = make_sample(1'100'000, 2'000'000 /* sin cambio */, 510'000, 51'000, 201'000, 1);
        auto result = build_cpu_features(prev, curr, 1'000'000);
        if (result.ok()) return 12;
        if (result.error != FeatureBuildError::kZeroCycles) return 13;
    }

    // delta_instructions == 0: mpki indefinido.
    {
        auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 1);
        auto curr = make_sample(1'000'000 /* sin cambio */, 2'100'000, 510'000, 51'000, 201'000, 1);
        auto result = build_cpu_features(prev, curr, 1'000'000);
        if (result.ok()) return 14;
        if (result.error != FeatureBuildError::kZeroInstructions) return 15;
    }

    // delta_cache_references == 0: cache_miss_rate indefinido.
    {
        auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 1);
        auto curr = make_sample(1'100'000, 2'100'000, 500'000 /* sin cambio */, 50'500, 201'000, 1);
        auto result = build_cpu_features(prev, curr, 1'000'000);
        if (result.ok()) return 16;
        if (result.error != FeatureBuildError::kZeroCacheReferences) return 17;
    }

    return 0;
}
