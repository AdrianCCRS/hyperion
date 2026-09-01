#include "telemetry/collector.hpp"

#include <cstdio>
#include <thread>
#include <chrono>

/**
 * ARC-65: end-to-end check that Collector actually gates NVML reads to
 * gpu_interval_ns instead of the 1 ms CPU/RAPL tick -- not just that the code
 * compiles. Only meaningful in a TELEMETRY_WITH_GPU build (needs a real or
 * stubbed NVML at link time); skips (exit 77) otherwise, same convention as
 * perf_reader_pid_live_test for PMU-permission-gated tests.
 */
int main() {
    if (!telemetry::NvmlReader::compiled_with_gpu()) {
        return 77;
    }

    telemetry::Collector::Ring ring;
    telemetry::CollectorConfig cfg;
    cfg.enable_perf = false;
    cfg.enable_gpu = true;
    cfg.interval_ns = 1'000'000;       // 1 ms producer tick, same as production
    cfg.gpu_interval_ns = 50'000'000;  // 50 ms GPU cadence (shorter than the
                                       // 100 ms default so this test stays fast)

    telemetry::Collector collector(cfg, ring);
    collector.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    collector.stop();

    int gpu_samples = 0;
    while (auto sample = ring.try_pop()) {
        if (sample->tag == telemetry::SampleTag::GPU) {
            ++gpu_samples;
        }
    }

    // ~300ms / 50ms cadence => ~6 GPU samples expected. If the gate were
    // absent (sampling every 1ms tick like CPU/RAPL), this would be ~300.
    // Generous bounds to absorb scheduling jitter on a loaded CI machine.
    if (gpu_samples < 2) {
        std::fprintf(stderr, "too few GPU samples: %d (gate too aggressive?)\n", gpu_samples);
        return 1;
    }
    if (gpu_samples > 15) {
        std::fprintf(stderr, "too many GPU samples: %d (cadence gate not working -- "
                              "NVML is being read every 1ms tick again)\n", gpu_samples);
        return 2;
    }

    return 0;
}
