#include "telemetry/experiment_utils.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>
#include <stdexcept>

namespace telemetry::experiment {

    uint64_t now_ns() noexcept {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + ts.tv_nsec;
    }

    namespace {
        int parse_cpu_token(const std::string& token) {
            if(token.empty()) throw std::invalid_argument("empty CPU token");
            size_t pos = 0;
            const int value = std::stoi(token, &pos);
            if(pos != token.size() || value < 0) {
                throw std::invalid_argument("invalid CPU token: " + token);
            }
            return value;
        }
    }

    std::vector<int> parse_cpu_list(const std::string& text) {
        if(text.empty()) throw std::invalid_argument("CPU list must not be empty");

        std::vector<int> cpus;
        size_t start = 0;
        while(start <= text.size()) {
            const size_t comma = text.find(',', start);
            const std::string token = text.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start
            );
            const size_t dash = token.find('-');
            if(dash == std::string::npos) {
                cpus.push_back(parse_cpu_token(token));
            } else {
                const int first = parse_cpu_token(token.substr(0, dash));
                const int last = parse_cpu_token(token.substr(dash + 1));
                if(last < first) {
                    throw std::invalid_argument("CPU range end is smaller than start");
                }
                for(int cpu = first; cpu <= last; ++cpu) {
                    cpus.push_back(cpu);
                }
            }
            if(comma == std::string::npos) break;
            start = comma + 1;
        }

        std::vector<int> sorted = cpus;
        std::sort(sorted.begin(), sorted.end());
        if(std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("CPU list contains duplicates");
        }
        return cpus;
    }

    std::string format_cpu_list(const std::vector<int>& cpus) {
        std::string out;
        for(size_t i = 0; i < cpus.size(); ++i) {
            if(i != 0) out += ",";
            out += std::to_string(cpus[i]);
        }
        return out;
    }

    bool contains_cpu(const std::vector<int>& cpus, int cpu) noexcept {
        return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
    }

    const char* kernel_label(const std::string& kernel) noexcept {
        if(kernel == "gemm_naive") return "compute_bound";
        if(kernel == "stream_triad") return "memory_bound";
        if(kernel == "reduction") return "memory_bound";
        if(kernel == "stencil_2d") return "cache_sensitive";
        return "unknown";
    }

    bool is_supported_kernel(const std::string& kernel) noexcept {
        return kernel_label(kernel)[0] != 'u';
    }

    Stats compute_stats(const std::vector<double>& values) noexcept {
        Stats stats{};
        if(values.empty()) return stats;

        stats.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                     static_cast<double>(values.size());
        double variance = 0.0;
        for(double value : values) {
            const double diff = value - stats.mean;
            variance += diff * diff;
        }
        stats.sd = std::sqrt(variance / static_cast<double>(values.size()));
        stats.cv_pct = stats.mean == 0.0 ? 0.0 : 100.0 * stats.sd / stats.mean;
        return stats;
    }

    double overhead_percent(double baseline, double telemetry) noexcept {
        if(baseline == 0.0) return 0.0;
        return 100.0 * (telemetry - baseline) / baseline;
    }

    std::string json_escape(const std::string& text) {
        std::string out;
        out.reserve(text.size());
        for(char c : text) {
            switch(c) {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
            }
        }
        return out;
    }

    namespace {
        bool rapl_wrap_without_valid_range(uint64_t previous_uj,
                                           uint64_t current_uj,
                                           uint64_t max_range_uj) noexcept {
            return current_uj < previous_uj &&
                   (max_range_uj == 0 || previous_uj > max_range_uj);
        }

        uint64_t rapl_delta_uj(uint64_t previous_uj,
                               uint64_t current_uj,
                               uint64_t max_range_uj) noexcept {
            if(current_uj >= previous_uj) return current_uj - previous_uj;
            return (max_range_uj - previous_uj) + current_uj;
        }
    }

    RaplDelta next_rapl_delta(int repetition,
                              const EnergySnapshot& current,
                              const RaplExportConfig& config,
                              RaplDeltaState& state) noexcept {
        RaplDelta delta{};

        if(state.repetition != repetition) {
            state.repetition = repetition;
            state.have_previous = false;
            state.previous = {};
        }

        if(state.have_previous) {
            const bool pkg_invalid = rapl_wrap_without_valid_range(
                state.previous.pkg_uj,
                current.pkg_uj,
                config.pkg_max_range_uj
            );
            const bool dram_invalid = rapl_wrap_without_valid_range(
                state.previous.dram_uj,
                current.dram_uj,
                config.dram_max_range_uj
            );

            if(!pkg_invalid && !dram_invalid) {
                delta.pkg_delta_uj = rapl_delta_uj(
                    state.previous.pkg_uj,
                    current.pkg_uj,
                    config.pkg_max_range_uj
                );
                delta.dram_delta_uj = rapl_delta_uj(
                    state.previous.dram_uj,
                    current.dram_uj,
                    config.dram_max_range_uj
                );
                delta.valid = true;
            }
        }

        state.previous = current;
        state.have_previous = true;
        return delta;
    }

}
