#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace telemetry::experiment {

    struct Stats {
        double mean = 0.0;
        double sd = 0.0;
        double cv_pct = 0.0;
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

}
