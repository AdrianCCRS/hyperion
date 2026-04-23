#pragma once

#include "telemetry/metrics.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace telemetry::experiment {

    /** @brief Basic descriptive statistics used in benchmark summaries. */
    struct Stats {
        double mean = 0.0;
        double sd = 0.0;
        double cv_pct = 0.0;
    };

    /**
     * @brief RAPL ranges needed to export wrap-aware energy deltas.
     *
     * Zero means the corresponding max_energy_range_uj was not available. If a
     * wrap is detected with a zero range, the produced RaplDelta is invalid.
     */
    struct RaplExportConfig {
        uint64_t pkg_max_range_uj = 0;
        uint64_t dram_max_range_uj = 0;
    };

    /**
     * @brief Stateful tracker for consecutive RAPL samples during export.
     *
     * The repetition id is part of the state so deltas never cross experiment
     * repetition boundaries.
     */
    struct RaplDeltaState {
        int repetition = -1;
        bool have_previous = false;
        EnergySnapshot previous{};
    };

    /** @brief Result of differencing two RAPL snapshots. */
    struct RaplDelta {
        uint64_t pkg_delta_uj = 0;
        uint64_t dram_delta_uj = 0;
        bool valid = false;
    };

    /** @brief Current CLOCK_MONOTONIC timestamp in nanoseconds. */
    uint64_t now_ns() noexcept;

    /** @brief Parse CPU lists such as "2,4-6" into explicit CPU ids. */
    std::vector<int> parse_cpu_list(const std::string& text);

    /** @brief Format explicit CPU ids as a comma-separated list. */
    std::string format_cpu_list(const std::vector<int>& cpus);

    /** @brief Return true when cpu is present in cpus. */
    bool contains_cpu(const std::vector<int>& cpus, int cpu) noexcept;

    /** @brief Return the dataset label assigned to a supported kernel name. */
    const char* kernel_label(const std::string& kernel) noexcept;

    /** @brief Return true if the kernel is implemented by the workload binary. */
    bool is_supported_kernel(const std::string& kernel) noexcept;

    /** @brief Compute mean, population standard deviation, and CV percent. */
    Stats compute_stats(const std::vector<double>& values) noexcept;

    /** @brief Compute percentage overhead relative to a baseline duration. */
    double overhead_percent(double baseline, double telemetry) noexcept;

    /** @brief Escape a string for the small JSON writer used by the launcher. */
    std::string json_escape(const std::string& text);

    /**
     * @brief Compute the next RAPL delta for export.
     *
     * The first ENERGY sample of each repetition is invalid because there is no
     * previous snapshot. Wrap-around is accepted only when the corresponding
     * max range is known.
     */
    RaplDelta next_rapl_delta(int repetition,
                              const EnergySnapshot& current,
                              const RaplExportConfig& config,
                              RaplDeltaState& state) noexcept;

}
