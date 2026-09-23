#include "cpu_loop_consumer.hpp"

#include <vector>

using namespace hyperion::cpu_loop;
using telemetry::CpuSample;
using telemetry::Sample;
using telemetry::SampleTag;

namespace {

Sample cpu_sample(uint64_t ts, uint64_t instructions, uint64_t cycles, uint64_t cache_refs,
                   uint64_t cache_misses, uint64_t stalled_mem, uint64_t freq_khz) {
    Sample s{};
    s.tag = SampleTag::CPU;
    s.cpu.timestamp_ns = ts;
    s.cpu.instructions = instructions;
    s.cpu.cycles = cycles;
    s.cpu.cache_references = cache_refs;
    s.cpu.cache_misses = cache_misses;
    s.cpu.stalled_cycles_mem_any = stalled_mem;
    s.cpu.scaling_cur_freq_khz = freq_khz;
    return s;
}

Sample non_cpu_sample() {
    Sample s{};
    s.tag = SampleTag::ENERGY;
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
    // 3 muestras CPU consecutivas -> 2 ticks (la primera solo establece "prev").
    // Muestras intercaladas con una no-CPU, que debe ignorarse sin romper
    // la secuencia prev/curr.
    {
        telemetry::Collector::Ring ring;
        if (!ring.try_push(cpu_sample(0, 1'000'000, 2'000'000, 500'000, 50'000, 200'000, 3'200'000))) return 1;
        if (!ring.try_push(non_cpu_sample())) return 2;
        if (!ring.try_push(cpu_sample(1'000'000, 2'500'000, 3'800'000, 900'000, 130'000, 350'000, 2'900'000))) return 3;
        if (!ring.try_push(cpu_sample(2'000'000, 4'000'000, 5'600'000, 1'300'000, 210'000, 500'000, 800'000))) return 4;

        std::atomic<bool> stop{true};  // un solo drenaje: hay muestras ya en el ring, no hace falta bucle
        auto cfg = make_config();
        std::vector<unsigned int> applied;
        CpuPhaseController controller(cfg, [&applied](unsigned int khz) { applied.push_back(khz); return true; });

        std::vector<TickResult> ticks;
        run_consumer_loop(
            ring, stop,
            [](const FeatureVector&) { return 0.95f; },  // siempre memory_bound, alta confianza
            0.85f, []{ return false; }, controller,
            [&ticks](const TickResult& r) { ticks.push_back(r); });

        if (ticks.size() != 2) return 5;  // 3 muestras CPU -> 2 pares consecutivos
        if (ticks[0].outcome != TickOutcome::kActed) return 6;
        if (ticks[1].outcome != TickOutcome::kActed) return 7;
        // memory_bound en ambos ticks -> mismo nivel -> solo la primera escritura real.
        if (applied.size() != 1 || applied[0] != 800000) return 8;
    }

    // Ring vacio + stop ya activo: no debe llamar a predict_proba ni a on_tick.
    {
        telemetry::Collector::Ring ring;
        std::atomic<bool> stop{true};
        auto cfg = make_config();
        CpuPhaseController controller(cfg, [](unsigned int) { return true; });
        bool predict_called = false;
        int tick_count = 0;
        run_consumer_loop(
            ring, stop, [&](const FeatureVector&) { predict_called = true; return 0.5f; },
            0.85f, []{ return true; }, controller, [&](const TickResult&) { tick_count++; });
        if (predict_called) return 9;
        if (tick_count != 0) return 10;
    }

    // gpu_active() se consulta en cada tick, no una sola vez al inicio.
    {
        telemetry::Collector::Ring ring;
        ring.try_push(cpu_sample(0, 1'000'000, 2'000'000, 500'000, 50'000, 200'000, 3'200'000));
        ring.try_push(cpu_sample(1'000'000, 2'500'000, 3'800'000, 900'000, 130'000, 350'000, 2'900'000));
        std::atomic<bool> stop{true};
        auto cfg = make_config();
        cfg.gpu_active_floor_khz = 3200000;
        CpuPhaseController controller(cfg, [](unsigned int) { return true; });
        int gpu_active_calls = 0;
        std::vector<TickResult> ticks;
        run_consumer_loop(
            ring, stop, [](const FeatureVector&) { return 0.95f; }, 0.85f,
            [&gpu_active_calls]{ gpu_active_calls++; return true; }, controller,
            [&ticks](const TickResult& r) { ticks.push_back(r); });
        if (gpu_active_calls != 1) return 11;  // 2 muestras CPU -> 1 par -> 1 consulta
        if (ticks.size() != 1) return 12;
        if (!ticks[0].decision || !ticks[0].decision->gpu_floor_clamped) return 13;
    }

    return 0;
}
