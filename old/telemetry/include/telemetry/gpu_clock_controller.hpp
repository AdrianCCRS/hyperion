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
 * every window would spend more in transition overhead than it saves.
 *
 * ARC-65: this class does NOT classify anything itself. Per the approved
 * thesis plan (docs/general/plan_trabajo_grado.md, sections 5.1/5.3), the
 * only telemetry source for GPU is NVML (utilization/memory/power), and
 * classification comes from a trained model consuming that telemetry live --
 * never from a static per-kernel operational-intensity table computed
 * offline with ncu. `ncu` still has a role (labeling the ground truth used
 * to TRAIN that model in Fase 1, see Diseno_Politica_DVFS_CPU_GPU.md section
 * 3), but it never reaches this class. The caller (either the offline
 * characterization campaign applying an ncu-derived label, or the live
 * daemon applying its model's prediction) decides the label; this class only
 * owns the hysteresis/dwell/apply state machine that turns "the label
 * changed" into "is it actually worth paying the clock transition cost right
 * now". Delegates the actual clock write through an injected callable so it
 * can be unit tested on any machine, with or without an NVIDIA GPU -- mirrors
 * how orchestrator/campaign.py injects apply_frequency() for the CPU path
 * instead of calling freqctl directly.
 */
namespace telemetry {

    enum class GpuPhaseLabel : uint8_t { ComputeBound, MemoryBound };

    struct GpuClockControllerConfig {
        /**
         * Minimum time a locked clock must stay applied before it can be
         * changed again, regardless of what later phases request.
         *
         * This must be set from a *measured* clock transition latency on the
         * real device (T_transicion), not assumed from literature -- see the
         * open item in Diseno_Politica_DVFS_CPU_GPU.md section 10. A common
         * starting rule is min_dwell_ns >= 10 * T_transicion_ns.
         */
        int64_t min_dwell_ns;

        unsigned int compute_bound_clock_mhz;
        unsigned int memory_bound_clock_mhz;
    };

    /** Outcome of one on_phase_begin() call, for logging/CSV export. */
    struct GpuPhaseDecision {
        GpuPhaseLabel label;              // echoes the label passed in, for logging convenience
        unsigned int target_clock_mhz;
        unsigned int applied_clock_mhz;   // what the GPU is actually locked to after this call
        bool clock_changed;               // true only if a new lock call was issued this time
        bool clock_setter_failed;         // true if clock_changed was attempted but the setter returned false
        int64_t dwell_remaining_ns;       // 0 unless a change was suppressed by the dwell floor
    };

    /**
     * @brief Hysteresis + dwell-gated GPU clock decision engine.
     *
     * Pure state machine: no NVML/CUDA calls, no classification logic of its
     * own. Tests supply a fake ClockSetter and assert on call counts/args
     * without any GPU hardware.
     */
    class GpuClockController {
    public:
        using ClockSetter = std::function<bool(unsigned int mhz)>;

        GpuClockController(GpuClockControllerConfig config, ClockSetter set_clock)
            : config_(config), set_clock_(std::move(set_clock)) {}

        /**
         * @brief Called once per phase boundary, not once per sample, with a
         * label that has ALREADY been decided by the caller (never computed
         * here -- see the file-level comment for why).
         *
         * @param label compute_bound/memory_bound for the phase about to run,
         *   from an ncu-derived offline ground truth (characterization
         *   campaign) or a live model prediction (daemon).
         * @param now_ns Monotonic timestamp of the phase boundary.
         */
        GpuPhaseDecision on_phase_begin(GpuPhaseLabel label, ns_t now_ns) {
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
                // Already at the right clock (including: same label as last time).
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
