#pragma once

#include "telemetry/metrics.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace telemetry::experiment {

    struct Stats {
        double mean = 0.0;
        double sd = 0.0;
        double cv_pct = 0.0;
    };

    struct RaplExportConfig {
        uint64_t pkg_max_range_uj = 0;
        uint64_t dram_max_range_uj = 0;
    };

    struct RaplDeltaState {
        int repetition = -1;
        bool have_previous = false;
        EnergySnapshot previous{};
    };

    struct RaplDelta {
        uint64_t pkg_delta_uj = 0;
        uint64_t dram_delta_uj = 0;
        bool valid = false;
    };

    uint64_t now_ns() noexcept;
    std::vector<int> parse_cpu_list(const std::string& text);
    std::string format_cpu_list(const std::vector<int>& cpus);
    bool contains_cpu(const std::vector<int>& cpus, int cpu) noexcept;
    const char* kernel_label(const std::string& kernel) noexcept;
    bool is_supported_kernel(const std::string& kernel) noexcept;
    Stats compute_stats(const std::vector<double>& values) noexcept;
    double overhead_percent(double baseline, double telemetry) noexcept;
    std::string json_escape(const std::string& text);
    RaplDelta next_rapl_delta(int repetition,
                              const EnergySnapshot& current,
                              const RaplExportConfig& config,
                              RaplDeltaState& state) noexcept;

}
