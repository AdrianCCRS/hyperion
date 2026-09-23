#include "cpu_loop_tick.hpp"

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

CpuPhaseControllerConfig make_config() {
    CpuPhaseControllerConfig cfg{};
    cfg.compute_bound = {true, 3600000};
    cfg.memory_bound = {true, 800000};
    return cfg;
}

}  // namespace

int main() {
    auto prev = make_sample(1'000'000, 2'000'000, 500'000, 50'000, 200'000, 3'200'000);
    auto curr = make_sample(2'500'000, 3'800'000, 900'000, 130'000, 350'000, 2'900'000);

    // Confianza alta hacia memory_bound (p=0.95 >= threshold 0.85) -> actua.
    {
        std::vector<unsigned int> applied;
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });
        auto result = run_cpu_tick(prev, curr, 1'000'000,
                                    [](const FeatureVector&) { return 0.95f; }, 0.85f,
                                    /*gpu_active=*/false, controller);
        if (result.outcome != TickOutcome::kActed) return 1;
        if (!result.decision || result.decision->target_freq_khz != 800000) return 2;
        if (applied.size() != 1 || applied[0] != 800000) return 3;
    }

    // Confianza baja (p=0.6, confidence=0.6 < threshold 0.85) -> abstiene,
    // NUNCA llama al setter.
    {
        std::vector<unsigned int> applied;
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });
        auto result = run_cpu_tick(prev, curr, 1'000'000,
                                    [](const FeatureVector&) { return 0.6f; }, 0.85f,
                                    /*gpu_active=*/false, controller);
        if (result.outcome != TickOutcome::kAbstained) return 4;
        if (!applied.empty()) return 5;
        if (!result.p_memory_bound || *result.p_memory_bound != 0.6f) return 6;
    }

    // p bajo (compute_bound) con confianza alta (p=0.05, confidence=0.95) -> actua compute_bound.
    {
        std::vector<unsigned int> applied;
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });
        auto result = run_cpu_tick(prev, curr, 1'000'000,
                                    [](const FeatureVector&) { return 0.05f; }, 0.85f,
                                    /*gpu_active=*/false, controller);
        if (result.outcome != TickOutcome::kActed) return 7;
        if (!result.decision || result.decision->target_freq_khz != 3600000) return 8;
    }

    // Fallo en la construccion de variables (delta_t_ns invalido) -> nunca
    // llega a inferir ni a decidir, error explicito.
    {
        std::vector<unsigned int> applied;
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });
        bool predict_called = false;
        auto result = run_cpu_tick(prev, curr, /*delta_t_ns=*/0,
                                    [&](const FeatureVector&) { predict_called = true; return 0.5f; }, 0.85f,
                                    /*gpu_active=*/false, controller);
        if (result.outcome != TickOutcome::kFeatureBuildFailed) return 9;
        if (predict_called) return 10;  // no debe invocar el modelo si las features no se pudieron construir
        if (applied.size() != 0) return 11;
        if (result.feature_error != FeatureBuildError::kNegativeOrZeroDeltaT) return 12;
    }

    // Umbral en el borde exacto: confidence == threshold cuenta como
    // suficiente (q >= tau, no q > tau -- misma ecuacion del libro).
    {
        std::vector<unsigned int> applied;
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });
        auto result = run_cpu_tick(prev, curr, 1'000'000,
                                    [](const FeatureVector&) { return 0.85f; }, 0.85f,
                                    /*gpu_active=*/false, controller);
        if (result.outcome != TickOutcome::kActed) return 13;
    }

    return 0;
}
