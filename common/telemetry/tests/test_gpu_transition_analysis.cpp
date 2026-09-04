#include "telemetry/gpu_transition_analysis.hpp"

#include <vector>

// F1-GPU-002: pure-logic tests for the GPU clock-transition probe. No NVML,
// no GPU -- builds and runs on any machine, including the CPU-only CI path.
// Return code == first failing check id (0 == all passed).

using telemetry::gpu_transition::ClockReading;
using telemetry::gpu_transition::StabilityConfig;
using telemetry::gpu_transition::StabilityOutcome;
using telemetry::gpu_transition::detect_stability;
using telemetry::gpu_transition::compute_transition_metrics;
using telemetry::gpu_transition::compute_cadence_stats;
using telemetry::gpu_transition::analyze_signal_steps;

namespace {

ClockReading mk(int64_t t_ns, unsigned int mhz, bool valid = true,
                unsigned int util = 100, unsigned long long throttle = 0,
                bool throttle_valid = true) {
    ClockReading r;
    r.t_mono_ns = t_ns;
    r.graphics_clock_mhz = mhz;
    r.graphics_clock_valid = valid;
    r.sm_clock_mhz = mhz;
    r.sm_clock_valid = valid;
    r.util_pct = util;
    r.util_valid = true;
    r.throttle_reasons = throttle;
    r.throttle_valid = throttle_valid;
    return r;
}

StabilityConfig cfg_1400() {
    StabilityConfig c;
    c.target_mhz = 1400;
    c.tolerance_mhz = 30;
    c.required_consecutive = 3;
    c.require_active = true;
    c.active_util_threshold_pct = 5;
    return c;
}

}  // namespace

int main() {
    const int64_t NEVER = INT64_MAX;

    // 1. Clock converges and stays stable.
    {
        std::vector<ClockReading> r = {
            mk(1000, 900), mk(2000, 1100), mk(3000, 1360),  // 1360: |1360-1400|=40 > tol 30
            mk(4000, 1402), mk(5000, 1398), mk(6000, 1401), mk(7000, 1400),
        };
        auto s = detect_stability(r, /*t_request=*/500, cfg_1400(), NEVER);
        if (s.outcome != StabilityOutcome::Stable) return 1;
        // First run of 3 consecutive in-tolerance: idx 3,4,5 -> stable at idx 5.
        if (s.stable_start_index != 3) return 2;
        if (s.stable_end_index != 5) return 3;
        if (s.t_stable_ns != 6000) return 4;

        auto m = compute_transition_metrics(/*t_solicitud=*/500,
                                            /*t_command_return=*/900, r, s);
        if (!m.valid) return 5;
        if (m.command_latency_ns != 400) return 6;
        if (m.t_actuacion_ns != 6000 - 500) return 7;
        if (m.settle_after_command_ns != 6000 - 900) return 8;
        if (m.conservative_upper_bound_ns != m.t_actuacion_ns) return 9;
        if (m.optimistic_ns != 4000 - 500) return 10;  // first of the run
    }

    // 2. First touch of the target, then leaves tolerance, then converges later.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1405),   // touches
            mk(2000, 1200),   // leaves -> run resets
            mk(3000, 1398),   // run of 3 starts here
            mk(4000, 1401),
            mk(5000, 1399),
            mk(6000, 1400),
        };
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome != StabilityOutcome::Stable) return 11;
        if (s.stable_start_index != 2) return 12;
        if (s.t_stable_ns != 5000) return 13;  // idx 2,3,4 complete the run
    }

    // 3. Timeout: target is never held for 3 in a row, last reading past deadline.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1402), mk(2000, 1200), mk(3000, 1405),
            mk(4000, 1100), mk(5000, 1401),
        };
        auto s = detect_stability(r, 0, cfg_1400(), /*deadline=*/5000);
        if (s.outcome != StabilityOutcome::Timeout) return 14;
        auto m = compute_transition_metrics(0, 100, r, s);
        if (m.valid) return 15;  // no T_actuacion when not stable
        if (m.command_latency_ns != 100) return 16;  // latency still reported
    }

    // 3b. Not stable but last reading is BEFORE the deadline -> Pending, not Timeout.
    {
        std::vector<ClockReading> r = { mk(1000, 1402), mk(2000, 1200) };
        auto s = detect_stability(r, 0, cfg_1400(), /*deadline=*/9000);
        if (s.outcome != StabilityOutcome::Pending) return 17;
    }

    // 4. Irregular timestamps: cadence stats use REAL gaps, not a nominal rate.
    {
        std::vector<int64_t> ts = {0, 5'000'000, 9'000'000, 30'000'000,
                                   35'000'000, 40'000'000, 46'000'000};
        auto c = compute_cadence_stats(ts);
        if (c.n_intervals != 6) return 18;
        // deltas: 5,4,21,5,5,6 (ms). min 4ms, max 21ms.
        if (c.min_delta_ns != 4'000'000) return 19;
        if (c.max_delta_ns != 21'000'000) return 20;
        // sorted deltas: 4,5,5,5,6,21 -> p50 (index floor(0.5*5)=2) = 5ms.
        if (c.p50_delta_ns != 5'000'000) return 21;
        // Nearest-rank p95 includes the tail observation: ceil(.95*6)-1=5.
        if (c.p95_delta_ns != 21'000'000) return 22;
    }

    // 5. NVML data absent: invalid SM-clock readings never count toward stability.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1400, /*valid=*/false),
            mk(2000, 1400, /*valid=*/false),
            mk(3000, 1400, /*valid=*/false),
            mk(4000, 1400, /*valid=*/false),
        };
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome == StabilityOutcome::Stable) return 23;
        if (s.considered != 4) return 24;
    }

    // 5b. GPU idle (util below threshold) disqualifies otherwise-in-tolerance reads.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1400, true, /*util=*/0),
            mk(2000, 1400, true, /*util=*/0),
            mk(3000, 1400, true, /*util=*/0),
        };
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome == StabilityOutcome::Stable) return 25;
    }

    // 5c. Invalidating throttle bit (HW thermal slowdown = 0x40) disqualifies.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1400, true, 100, /*throttle=*/0x40),
            mk(2000, 1400, true, 100, /*throttle=*/0x40),
            mk(3000, 1400, true, 100, /*throttle=*/0x40),
            mk(4000, 1400, true, 100, /*throttle=*/0x00),  // clean
            mk(5000, 1400, true, 100, /*throttle=*/0x00),
            mk(6000, 1400, true, 100, /*throttle=*/0x00),
        };
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome != StabilityOutcome::Stable) return 26;
        if (s.stable_start_index != 3) return 27;  // only the clean run counts
    }

    // 5d. The `-lgc` lock's own "applications clocks setting" bit (0x2) must NOT
    // invalidate -- that is the state we asked for.
    {
        std::vector<ClockReading> r = {
            mk(1000, 1400, true, 100, /*throttle=*/0x2),
            mk(2000, 1400, true, 100, /*throttle=*/0x2),
            mk(3000, 1400, true, 100, /*throttle=*/0x2),
        };
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome != StabilityOutcome::Stable) return 28;
    }

    // 5e. -lgc constrains graphics clocks, so the default must not accept a
    // matching SM value when graphics itself is still outside tolerance.
    {
        std::vector<ClockReading> r = {mk(1000, 1200), mk(2000, 1200), mk(3000, 1200)};
        for (auto& reading : r) reading.sm_clock_mhz = 1400;
        auto s = detect_stability(r, 0, cfg_1400(), NEVER);
        if (s.outcome == StabilityOutcome::Stable) return 29;
    }

    // 6. Signal-step / redundancy analysis + conservative lower-bound framing.
    {
        //            t(ms):   0   5  10  15  20  25
        //            power : 100 100 100 110 110 120
        std::vector<int64_t> ts = {0, 5'000'000, 10'000'000, 15'000'000,
                                   20'000'000, 25'000'000};
        std::vector<long long> pw = {100, 100, 100, 110, 110, 120};
        std::vector<bool> ok = {true, true, true, true, true, true};
        auto st = analyze_signal_steps(ts, pw, ok);
        if (st.n_valid != 6) return 30;
        if (st.n_consecutive_changes != 2) return 31;   // 100->110, 110->120
        // 5 adjacent pairs, 3 unchanged -> redundancy 0.6
        if (!(st.redundancy_ratio > 0.59 && st.redundancy_ratio < 0.61)) return 32;
        // steps: [0..15]=15ms, [15..25]=10ms, trailing [25..25]=0 -> sorted 0,10,15
        if (st.median_step_duration_ns != 10'000'000) return 33;
        if (st.max_step_duration_ns != 15'000'000) return 34;
    }

    // 6b. All-invalid signal -> no stats, no divide-by-zero.
    {
        std::vector<int64_t> ts = {0, 1, 2};
        std::vector<long long> v = {1, 2, 3};
        std::vector<bool> ok = {false, false, false};
        auto st = analyze_signal_steps(ts, v, ok);
        if (st.n_valid != 0) return 35;
        if (st.n_consecutive_changes != 0) return 36;
    }

    // 7. Empty / single-reading inputs must not crash and must not be "stable".
    {
        std::vector<ClockReading> empty;
        auto s = detect_stability(empty, 0, cfg_1400(), NEVER);
        if (s.outcome == StabilityOutcome::Stable) return 37;
        auto c = compute_cadence_stats({});
        if (c.n_intervals != 0) return 38;
        auto c1 = compute_cadence_stats({42});
        if (c1.n_intervals != 0) return 39;
    }

    return 0;
}
