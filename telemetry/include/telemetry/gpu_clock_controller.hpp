#pragma once
#include "metrics.hpp"
#include <cstdint>
#include <functional>

/**
 * @file
 * @brief Phase-granularity GPU clock decision engine.
 *
 * This is deliberately NOT a per-sample (1 ms) controller like the CPU path.
 * Locking the GPU SM clock (nvmlDeviceSetGpuLockedClocks) has latency and cost
 * an order of magnitude above writing scaling_min_freq on CPU, so deciding
 * every window would spend more in transition overhead than it saves. The
 * unit of decision here is a *phase*: a maximal run of GPU kernels that share
 * an operational-intensity classification, bounded by whoever launches those
 * kernels (see gpu_phase_control design in
 * docs/retoma/pacca/Diseno_Politica_DVFS_CPU_GPU.md).
 *
 * Because GPU operational intensity is measured offline per kernel (ncu), not
 * from live PMU counters, this class takes a caller-supplied intensity value
 * per phase rather than reading any hardware counter itself. It owns only the
 * classify -> hysteresis -> dwell -> apply state machine, and delegates the
 * actual clock write through an injected callable so it can be unit tested on
 * any machine, with or without an NVIDIA GPU.
 */
namespace telemetry {

    enum class GpuPhaseLabel : uint8_t { ComputeBound, MemoryBound };

    struct GpuClockControllerConfig {
        /** i_ridge = P_pico / BW_pico for this GPU, from its own Roofline calibration. */
        double i_ridge_flops_per_byte;

        /**
         * Fractional hysteresis band around i_ridge (e.g. 0.15 = +/-15%).
         * A phase whose intensity falls inside the band keeps the previous
         * label instead of relabeling, so noise near the ridge point does not
         * by itself trigger a clock change.
         */
        double hysteresis_margin = 0.15;

        /**
         * Minimum time a locked clock must stay applied before it can be
         * changed again, regardless of what later phases request.
         *
         * This must be set from a *measured* clock transition latency on the
         * real device (T_transicion), not assumed from literature -- see the
         * open item in Diseno_Politica_DVFS_CPU_GPU.md section 5. A common
         * starting rule is min_dwell_ns >= 10 * T_transicion_ns.
         */
        int64_t min_dwell_ns;

        unsigned int compute_bound_clock_mhz;
        unsigned int memory_bound_clock_mhz;
    };

    /** Outcome of one on_phase_begin() call, for logging/CSV export. */
    struct GpuPhaseDecision {
        GpuPhaseLabel label;
        unsigned int target_clock_mhz;
        unsigned int applied_clock_mhz;  // what the GPU is actually locked to after this call
        bool clock_changed;              // true only if a new lock call was issued this time
        bool clock_setter_failed;        // true if clock_changed was attempted but the setter returned false
        int64_t dwell_remaining_ns;      // 0 unless a change was suppressed by the dwell floor
    };

    /**
     * @brief Classify + hysteresis + dwell-gated GPU clock decision engine.
     *
     * Pure state machine: no NVML/CUDA calls of its own. The constructor takes
     * a ClockSetter so tests can supply a fake and assert on call counts/args
     * without any GPU hardware, mirroring how orchestrator/campaign.py injects
     * apply_frequency() for the CPU path instead of calling freqctl directly.
     */
    class GpuClockController {
    public:
        using ClockSetter = std::function<bool(unsigned int mhz)>;

        GpuClockController(GpuClockControllerConfig config, ClockSetter set_clock)
            : config_(config), set_clock_(std::move(set_clock)) {}

        /**
         * @brief Called once per phase boundary, not once per sample.
         *
         * @param operational_intensity_flops_per_byte Static intensity of the
         *   kernel(s) about to run in this phase, from the offline ncu-derived
         *   table -- never computed from a live counter.
         * @param now_ns Monotonic timestamp of the phase boundary.
         */
        GpuPhaseDecision on_phase_begin(double operational_intensity_flops_per_byte, ns_t now_ns) {
            const GpuPhaseLabel label = classify(operational_intensity_flops_per_byte);
            const unsigned int desired_clock_mhz =
                (label == GpuPhaseLabel::ComputeBound) ? config_.compute_bound_clock_mhz
                                                        : config_.memory_bound_clock_mhz;

            GpuPhaseDecision decision{};
            decision.label = label;
            decision.target_clock_mhz = desired_clock_mhz;

            // First call ever: nothing to compare against, apply unconditionally.
            if (!has_applied_once_) {
                apply(decision, desired_clock_mhz, now_ns);
                current_label_ = label;
                return decision;
            }

            if (desired_clock_mhz == current_clock_mhz_) {
                // Already at the right clock (including: intensity stayed inside
                // the hysteresis band and kept the previous label). Nothing to do.
                decision.applied_clock_mhz = current_clock_mhz_;
                decision.clock_changed = false;
                decision.dwell_remaining_ns = 0;
                return decision;
            }

            const int64_t elapsed_ns = now_ns - last_change_ns_;
            if (elapsed_ns < config_.min_dwell_ns) {
                // Wants to change, but hasn't held the current clock long enough
                // to amortize the transition cost of the LAST change. Stay put.
                decision.applied_clock_mhz = current_clock_mhz_;
                decision.clock_changed = false;
                decision.dwell_remaining_ns = config_.min_dwell_ns - elapsed_ns;
                return decision;
            }

            apply(decision, desired_clock_mhz, now_ns);
            current_label_ = label;
            return decision;
        }

        GpuPhaseLabel current_label() const noexcept { return current_label_; }
        unsigned int current_clock_mhz() const noexcept { return current_clock_mhz_; }
        bool has_applied_once() const noexcept { return has_applied_once_; }

    private:
        GpuPhaseLabel classify(double intensity) const {
            const double lower = config_.i_ridge_flops_per_byte * (1.0 - config_.hysteresis_margin);
            const double upper = config_.i_ridge_flops_per_byte * (1.0 + config_.hysteresis_margin);
            if (intensity < lower) return GpuPhaseLabel::MemoryBound;
            if (intensity > upper) return GpuPhaseLabel::ComputeBound;
            // Inside the band: ambiguous by design, keep whatever label is
            // already active (defaults to ComputeBound before the first call,
            // which is discarded anyway since has_applied_once_ is false then).
            return current_label_;
        }

        void apply(GpuPhaseDecision& decision, unsigned int desired_clock_mhz, ns_t now_ns) {
            const bool ok = set_clock_(desired_clock_mhz);
            decision.clock_changed = true;
            decision.clock_setter_failed = !ok;
            if (ok) {
                current_clock_mhz_ = desired_clock_mhz;
                last_change_ns_ = now_ns;
                has_applied_once_ = true;
            }
            // On failure, deliberately do NOT update current_clock_mhz_/last_change_ns_:
            // the GPU is still at whatever clock it was at before, and the next
            // call should retry rather than believe a change that never happened.
            decision.applied_clock_mhz = current_clock_mhz_;
            decision.dwell_remaining_ns = 0;
        }

        GpuClockControllerConfig config_;
        ClockSetter set_clock_;
        GpuPhaseLabel current_label_ = GpuPhaseLabel::ComputeBound;
        unsigned int current_clock_mhz_ = 0;
        ns_t last_change_ns_ = 0;
        bool has_applied_once_ = false;
    };

}
