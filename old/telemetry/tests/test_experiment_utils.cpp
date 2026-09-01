#include "telemetry/experiment_utils.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    const auto cpus = telemetry::experiment::parse_cpu_list("2,4-6");
    if(cpus.size() != 4) return 1;
    if(cpus[0] != 2 || cpus[1] != 4 || cpus[2] != 5 || cpus[3] != 6) return 1;
    if(telemetry::experiment::format_cpu_list(cpus) != "2,4,5,6") return 1;
    if(!telemetry::experiment::contains_cpu(cpus, 5)) return 1;
    if(telemetry::experiment::contains_cpu(cpus, 3)) return 1;

    bool threw = false;
    try {
        (void)telemetry::experiment::parse_cpu_list("1,1");
    } catch(const std::invalid_argument&) {
        threw = true;
    }
    if(!threw) return 1;

    if(!telemetry::experiment::is_supported_kernel("stream_triad")) return 1;
    if(!telemetry::experiment::is_supported_kernel("gemm_naive")) return 1;
    if(telemetry::experiment::is_supported_kernel("fft")) return 1;
    if(std::string(telemetry::experiment::kernel_label("stencil_2d")) != "cache_sensitive") return 1;

    const std::vector<double> values{1.0, 2.0, 3.0};
    const auto stats = telemetry::experiment::compute_stats(values);
    if(std::fabs(stats.mean - 2.0) > 1e-12) return 1;
    if(stats.sd <= 0.0) return 1;
    if(std::fabs(telemetry::experiment::overhead_percent(100.0, 110.0) - 10.0) > 1e-12) return 1;
    if(telemetry::experiment::json_escape("a\"b\\c") != "a\\\"b\\\\c") return 1;

    telemetry::experiment::RaplExportConfig rapl_config{};
    rapl_config.pkg_max_range_uj = 100;
    rapl_config.dram_max_range_uj = 1000;
    telemetry::experiment::RaplDeltaState rapl_state{};

    auto delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{10, 90, 900},
        rapl_config,
        rapl_state
    );
    if(delta.valid) return 1;

    delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{20, 5, 950},
        rapl_config,
        rapl_state
    );
    if(!delta.valid || delta.pkg_delta_uj != 15 || delta.dram_delta_uj != 50) return 1;

    delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{30, 8, 20},
        rapl_config,
        rapl_state
    );
    if(!delta.valid || delta.pkg_delta_uj != 3 || delta.dram_delta_uj != 70) return 1;

    delta = telemetry::experiment::next_rapl_delta(
        2,
        telemetry::EnergySnapshot{40, 12, 30},
        rapl_config,
        rapl_state
    );
    if(delta.valid) return 1;

    telemetry::experiment::RaplExportConfig no_range_config{};
    telemetry::experiment::RaplDeltaState no_range_state{};
    delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{50, 90, 0},
        no_range_config,
        no_range_state
    );
    if(delta.valid) return 1;

    delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{60, 95, 0},
        no_range_config,
        no_range_state
    );
    if(!delta.valid || delta.pkg_delta_uj != 5 || delta.dram_delta_uj != 0) return 1;

    delta = telemetry::experiment::next_rapl_delta(
        1,
        telemetry::EnergySnapshot{70, 5, 0},
        no_range_config,
        no_range_state
    );
    if(delta.valid) return 1;

    return 0;
}
