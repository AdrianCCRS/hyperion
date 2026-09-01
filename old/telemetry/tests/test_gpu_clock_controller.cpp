#include "telemetry/gpu_clock_controller.hpp"

#include <vector>

int main() {
    using telemetry::GpuClockController;
    using telemetry::GpuClockControllerConfig;
    using telemetry::GpuPhaseLabel;

    GpuClockControllerConfig config{};
    config.min_dwell_ns = 1'000'000'000;     // 1s floor for this test
    config.compute_bound_clock_mhz = 1410;
    config.memory_bound_clock_mhz = 900;

    std::vector<unsigned int> applied_clocks;
    auto setter = [&applied_clocks](unsigned int mhz) {
        applied_clocks.push_back(mhz);
        return true;
    };

    GpuClockController controller(config, setter);

    // First call: always applies, regardless of dwell.
    auto decision = controller.on_phase_begin(GpuPhaseLabel::ComputeBound, /*now_ns=*/0);
    if (decision.label != GpuPhaseLabel::ComputeBound) return 1;
    if (!decision.clock_changed) return 2;
    if (decision.applied_clock_mhz != 1410) return 3;
    if (applied_clocks.size() != 1 || applied_clocks[0] != 1410) return 4;

    // Second phase, same label -- nothing to do, even inside the dwell floor.
    decision = controller.on_phase_begin(GpuPhaseLabel::ComputeBound, /*now_ns=*/100);
    if (decision.clock_changed) return 5;
    if (applied_clocks.size() != 1) return 6;

    // Third phase, caller decided memory-bound, but still inside the 1s dwell
    // floor since the last (and only) change at t=0 -- must be suppressed.
    decision = controller.on_phase_begin(GpuPhaseLabel::MemoryBound, /*now_ns=*/500'000'000);
    if (decision.clock_changed) return 7;
    if (decision.dwell_remaining_ns != 500'000'000) return 8;
    if (decision.applied_clock_mhz != 1410) return 9; // still at the old clock
    if (applied_clocks.size() != 1) return 10;

    // Fourth phase, same memory-bound label, now past the dwell floor
    // (t=1_500_000_000 - 0 >= 1_000_000_000) -- must apply this time.
    decision = controller.on_phase_begin(GpuPhaseLabel::MemoryBound, /*now_ns=*/1'500'000'000);
    if (decision.label != GpuPhaseLabel::MemoryBound) return 11;
    if (!decision.clock_changed) return 12;
    if (decision.applied_clock_mhz != 900) return 13;
    if (applied_clocks.size() != 2 || applied_clocks[1] != 900) return 14;

    // A failing setter must not update controller state -- the next call
    // should see the clock as still unchanged and retry, not believe a
    // transition that never actually happened on the device.
    auto failing_setter = [](unsigned int) { return false; };
    GpuClockController flaky(config, failing_setter);
    decision = flaky.on_phase_begin(GpuPhaseLabel::ComputeBound, /*now_ns=*/0);
    if (!decision.clock_changed) return 15;
    if (!decision.clock_setter_failed) return 16;
    if (flaky.has_applied_once()) return 17;
    if (flaky.current_clock_mhz() != 0) return 18;

    return 0;
}
