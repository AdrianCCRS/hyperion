#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

/**
 * @file
 * @brief F1-GPU-002 pure analysis logic for the GPU clock-transition probe.
 *
 * This header has NO NVML/CUDA dependency on purpose. The probe executable
 * (experiments/gpu_clock_transition_probe.cpp) owns all device I/O; everything
 * that decides "is the SM clock stable at the target", "what is T_actuacion",
 * and "what cadence did we actually observe" lives here so it can be unit
 * tested on any machine, with or without a GPU -- mirrors how
 * gpu_clock_controller.hpp keeps its state machine free of NVML calls.
 *
 * Terminology (Seguimiento_Cambios_Plan_Director.md, F1-GPU-002; plan §2.4.1):
 *   t_solicitud      -- monotonic ns captured immediately BEFORE issuing
 *                       `nvidia-smi -lgc`.
 *   t_command_return -- monotonic ns when that command's process was reaped.
 *   t_estable        -- monotonic ns of the reading that COMPLETES the first
 *                       run of `required_consecutive` in-tolerance,
 *                       GPU-active, non-invalidated SM-clock readings.
 *   T_actuacion      = t_estable - t_solicitud.
 *
 * A measured T_actuacion is an OBSERVABLE UPPER BOUND, never an exact physical
 * latency: the SM clock is reported at MHz granularity and the probe polls at a
 * finite cadence, so the true settle instant lies somewhere in the last gap
 * before t_estable. The conservative bound (the larger number) is the one that
 * is safe to feed into min_dwell_ns / --t-transicion-gpu-ns.
 */
namespace telemetry::gpu_transition {

    /**
     * @brief One NVML poll, already reduced to plain data by the probe.
     *
     * *_valid mirrors the "not measured != real zero" contract used elsewhere
     * (nvml_reader.cpp, UncoreSnapshot::interval_valid). A reading whose
     * sm_clock_valid is false can never count toward stability.
     */
    struct ClockReading {
        /// Start of the polling bundle.  This is diagnostic only: it MUST NOT
        /// be used as the instant at which a clock value was observed.
        int64_t  t_poll_start_ns = 0;
        /// CLOCK_MONOTONIC timestamp taken immediately AFTER the graphics
        /// clock query returns.  It is deliberately the timestamp used by
        /// stability/transition calculations, so T_actuacion never predates
        /// the observation that confirmed it.
        int64_t  t_mono_ns = 0;
        unsigned int graphics_clock_mhz = 0;
        bool     graphics_clock_valid = false;
        unsigned int sm_clock_mhz = 0;
        bool     sm_clock_valid = false;
        unsigned int util_pct = 0;          ///< nvmlDeviceGetUtilizationRates().gpu
        bool     util_valid = false;
        unsigned int mem_util_pct = 0;
        bool     mem_util_valid = false;
        unsigned int power_mw = 0;
        bool     power_valid = false;
        unsigned int temperature_c = 0;
        bool     temperature_valid = false;
        unsigned long long energy_mj = 0;
        bool     energy_valid = false;
        unsigned long long throttle_reasons = 0;  ///< nvmlDeviceGetCurrentClocksThrottleReasons bitmask.
        bool     throttle_valid = false;
    };

    /**
     * @brief Clock-throttle bits that INVALIDATE a stability reading.
     *
     * Copied from nvml.h so this header stays NVML-free. Deliberately excludes
     * ApplicationsClocksSetting (0x2 -- that IS the `-lgc` lock we asked for),
     * SwPowerCap (0x4) and GpuIdle (0x1 -- idle is gated separately via util).
     * Kept: HW slowdown, SW/HW thermal slowdown, HW power-brake slowdown --
     * any of these means the clock we are reading is not the clock the driver
     * would hold for this lock under a clean load.
     */
    inline constexpr unsigned long long kDefaultInvalidatingThrottleMask =
        0x8ULL   /* nvmlClocksThrottleReasonHwSlowdown        */
      | 0x20ULL  /* nvmlClocksThrottleReasonSwThermalSlowdown  */
      | 0x40ULL  /* nvmlClocksThrottleReasonHwThermalSlowdown  */
      | 0x80ULL; /* nvmlClocksThrottleReasonHwPowerBrakeSlowdown*/

    struct StabilityConfig {
        enum class ClockDomain { Graphics, Sm };
        unsigned int target_mhz = 0;
        unsigned int tolerance_mhz = 0;
        int          required_consecutive = 3;
        bool         require_active = true;
        unsigned int active_util_threshold_pct = 5;
        unsigned long long invalidating_throttle_mask = kDefaultInvalidatingThrottleMask;
        /// -lgc locks the graphics clock.  Graphics is therefore the safe
        /// default for validating actuation; SM remains an exported signal.
        ClockDomain clock_domain = ClockDomain::Graphics;
    };

    enum class StabilityOutcome { Stable, Timeout, Pending };

    struct StabilityResult {
        StabilityOutcome outcome = StabilityOutcome::Pending;
        /// Monotonic ns of the reading that completed the qualifying run.
        int64_t t_stable_ns = 0;
        /// Index (into the filtered-from-request view? no -- into the ORIGINAL
        /// vector) of the first reading of the qualifying run, and of the one
        /// that completed it.
        std::size_t stable_start_index = 0;
        std::size_t stable_end_index = 0;
        /// How many readings at or after t_request were examined.
        std::size_t considered = 0;
    };

    /** @brief True if one reading qualifies as "at target, right now". */
    inline bool reading_qualifies(const ClockReading& r, const StabilityConfig& cfg) {
        const bool clock_valid = cfg.clock_domain == StabilityConfig::ClockDomain::Graphics
            ? r.graphics_clock_valid : r.sm_clock_valid;
        const unsigned int clock_mhz = cfg.clock_domain == StabilityConfig::ClockDomain::Graphics
            ? r.graphics_clock_mhz : r.sm_clock_mhz;
        if (!clock_valid) return false;
        const long diff = static_cast<long>(clock_mhz) - static_cast<long>(cfg.target_mhz);
        if (std::abs(diff) > static_cast<long>(cfg.tolerance_mhz)) return false;
        if (cfg.require_active) {
            if (!r.util_valid) return false;
            if (r.util_pct < cfg.active_util_threshold_pct) return false;
        }
        if (r.throttle_valid && (r.throttle_reasons & cfg.invalidating_throttle_mask) != 0ULL) {
            return false;
        }
        return true;
    }

    /**
     * @brief Scan readings from t_request_ns onward for the first run of
     * `cfg.required_consecutive` qualifying readings.
     *
     * @param readings   full poll trace, assumed sorted by t_mono_ns.
     * @param t_request_ns readings strictly before this are ignored.
     * @param deadline_ns if stability is not reached and the last considered
     *   reading is at or past this, the outcome is Timeout; otherwise Pending
     *   (more data could still have arrived). Pass INT64_MAX to never time out.
     *
     * "First touch then leave tolerance" resets the run: a lone in-tolerance
     * reading followed by an out-of-tolerance one does not count.
     */
    inline StabilityResult detect_stability(const std::vector<ClockReading>& readings,
                                            int64_t t_request_ns,
                                            const StabilityConfig& cfg,
                                            int64_t deadline_ns) {
        StabilityResult res;
        const int need = cfg.required_consecutive < 1 ? 1 : cfg.required_consecutive;
        int run = 0;
        std::size_t run_start = 0;
        int64_t last_considered_ns = t_request_ns;
        for (std::size_t i = 0; i < readings.size(); ++i) {
            const ClockReading& r = readings[i];
            if (r.t_mono_ns < t_request_ns) continue;
            ++res.considered;
            last_considered_ns = r.t_mono_ns;
            if (reading_qualifies(r, cfg)) {
                if (run == 0) run_start = i;
                ++run;
                if (run >= need) {
                    res.outcome = StabilityOutcome::Stable;
                    res.t_stable_ns = r.t_mono_ns;
                    res.stable_start_index = run_start;
                    res.stable_end_index = i;
                    return res;
                }
            } else {
                run = 0;
            }
        }
        res.outcome = (last_considered_ns >= deadline_ns && deadline_ns != INT64_MAX)
                          ? StabilityOutcome::Timeout
                          : StabilityOutcome::Pending;
        return res;
    }

    // ---------------------------------------------------------------------
    // Transition metrics
    // ---------------------------------------------------------------------

    struct TransitionMetrics {
        int64_t command_latency_ns = 0;          ///< t_command_return - t_solicitud
        int64_t t_actuacion_ns = 0;              ///< t_estable - t_solicitud  (observed)
        int64_t settle_after_command_ns = 0;     ///< t_estable - t_command_return
        /// Conservative upper bound safe for min_dwell_ns: equals the observed
        /// t_actuacion_ns (we know the clock was NOT confirmed stable any
        /// earlier than t_estable).
        int64_t conservative_upper_bound_ns = 0;
        /// Optimistic reading of the same event: time to the FIRST reading of
        /// the qualifying run. Reported for context only, never for dwell.
        int64_t optimistic_ns = 0;
        bool valid = false;                      ///< false unless stability was reached.
    };

    inline TransitionMetrics compute_transition_metrics(int64_t t_solicitud_ns,
                                                        int64_t t_command_return_ns,
                                                        const std::vector<ClockReading>& readings,
                                                        const StabilityResult& stability) {
        TransitionMetrics m;
        m.command_latency_ns = t_command_return_ns - t_solicitud_ns;
        if (stability.outcome != StabilityOutcome::Stable) {
            return m;  // valid stays false; caller reports the failure reason.
        }
        m.t_actuacion_ns = stability.t_stable_ns - t_solicitud_ns;
        m.settle_after_command_ns = stability.t_stable_ns - t_command_return_ns;
        m.conservative_upper_bound_ns = m.t_actuacion_ns;
        if (stability.stable_start_index < readings.size()) {
            m.optimistic_ns = readings[stability.stable_start_index].t_mono_ns - t_solicitud_ns;
        } else {
            m.optimistic_ns = m.t_actuacion_ns;
        }
        m.valid = true;
        return m;
    }

    // ---------------------------------------------------------------------
    // Cadence analysis (F1-GPU-002 requirement 5)
    // ---------------------------------------------------------------------

    struct CadenceStats {
        std::size_t n_intervals = 0;
        int64_t p50_delta_ns = 0;
        int64_t p95_delta_ns = 0;
        int64_t min_delta_ns = 0;
        int64_t max_delta_ns = 0;
    };

    /** @brief Nearest-rank percentile of a copy of `values` (values may be unsorted). */
    inline int64_t percentile_nearest_rank(std::vector<int64_t> values, double q) {
        if (values.empty()) return 0;
        std::sort(values.begin(), values.end());
        if (q <= 0.0) return values.front();
        if (q >= 1.0) return values.back();
        const auto index = static_cast<std::size_t>(std::ceil(q * values.size())) - 1;
        return values[std::min(index, values.size() - 1)];
    }

    /**
     * @brief p50/p95 of the REAL inter-reading gaps, not the requested cadence.
     * The probe asks for a fixed interval but scheduling jitter and NVML call
     * cost make the delivered cadence irregular; this reports what happened.
     */
    inline CadenceStats compute_cadence_stats(const std::vector<int64_t>& timestamps_ns) {
        CadenceStats s;
        if (timestamps_ns.size() < 2) return s;
        std::vector<int64_t> deltas;
        deltas.reserve(timestamps_ns.size() - 1);
        for (std::size_t i = 1; i < timestamps_ns.size(); ++i) {
            deltas.push_back(timestamps_ns[i] - timestamps_ns[i - 1]);
        }
        s.n_intervals = deltas.size();
        s.min_delta_ns = *std::min_element(deltas.begin(), deltas.end());
        s.max_delta_ns = *std::max_element(deltas.begin(), deltas.end());
        s.p50_delta_ns = percentile_nearest_rank(deltas, 0.50);
        s.p95_delta_ns = percentile_nearest_rank(deltas, 0.95);
        return s;
    }

    struct SignalStepStats {
        std::size_t n_valid = 0;
        /// Consecutive value changes. This is a LOWER BOUND on the number of
        /// physical sensor updates: two real updates that return the same
        /// value are counted as zero changes. Never report this as a
        /// confirmed physical refresh rate.
        std::size_t n_consecutive_changes = 0;
        /// Redundancy: fraction of adjacent valid pairs that did NOT change.
        double redundancy_ratio = 0.0;
        int64_t median_step_duration_ns = 0;  ///< median time a value was held before it changed
        int64_t max_step_duration_ns = 0;
        /// n_consecutive_changes / observed_span_seconds -- an OBSERVED rate /
        /// lower bound, explicitly not a physical guarantee.
        double observed_update_rate_hz_lower_bound = 0.0;
    };

    /**
     * @brief Step/redundancy analysis for one scalar NVML signal.
     * @param ts_ns    poll timestamps (sorted).
     * @param values   signal value per poll (any integer signal: power_mw,
     *                 util_pct, sm_clock_mhz, ...).
     * @param valid    per-poll validity; invalid samples are skipped entirely.
     */
    inline SignalStepStats analyze_signal_steps(const std::vector<int64_t>& ts_ns,
                                                const std::vector<long long>& values,
                                                const std::vector<bool>& valid) {
        SignalStepStats s;
        const std::size_t n = std::min(ts_ns.size(), std::min(values.size(), valid.size()));
        // Compact to valid-only (timestamp, value) pairs.
        std::vector<int64_t> vt;
        std::vector<long long> vv;
        vt.reserve(n); vv.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (valid[i]) { vt.push_back(ts_ns[i]); vv.push_back(values[i]); }
        }
        s.n_valid = vv.size();
        if (vv.size() < 2) return s;

        std::vector<int64_t> step_durations;
        std::size_t adjacent_pairs = vv.size() - 1;
        std::size_t unchanged_pairs = 0;
        int64_t current_step_start = vt.front();
        for (std::size_t i = 1; i < vv.size(); ++i) {
            if (vv[i] != vv[i - 1]) {
                ++s.n_consecutive_changes;
                step_durations.push_back(vt[i] - current_step_start);
                current_step_start = vt[i];
            } else {
                ++unchanged_pairs;
            }
        }
        // Trailing (still-open) step.
        step_durations.push_back(vt.back() - current_step_start);

        s.redundancy_ratio = adjacent_pairs > 0
            ? static_cast<double>(unchanged_pairs) / static_cast<double>(adjacent_pairs)
            : 0.0;
        if (!step_durations.empty()) {
            std::sort(step_durations.begin(), step_durations.end());
            s.median_step_duration_ns = step_durations[step_durations.size() / 2];
            s.max_step_duration_ns = step_durations.back();
        }
        const double span_s = static_cast<double>(vt.back() - vt.front()) / 1e9;
        s.observed_update_rate_hz_lower_bound =
            span_s > 0.0 ? static_cast<double>(s.n_consecutive_changes) / span_s : 0.0;
        return s;
    }

}  // namespace telemetry::gpu_transition
