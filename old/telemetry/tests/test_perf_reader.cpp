#include "telemetry/perf_reader.hpp"

int main() {
    telemetry::PerfReader reader;
    telemetry::CpuSample sample{};

    if(reader.is_open()) return 1;
    if(reader.read(sample)) return 1;
    if(reader.has_stalled_cycles_backend()) return 1;
    if(reader.has_l2_lines_in_all()) return 1;
    if(reader.has_fp_arith()) return 1; // ARC-97: false before open(), same as the other optional counters

    reader.enable();
    reader.disable();
    reader.close();
    if(reader.is_open()) return 1;

    if(telemetry::detail::scale_perf_count(10, 100, 100) != 10) return 1;
    if(telemetry::detail::scale_perf_count(10, 200, 100) != 20) return 1;
    if(telemetry::detail::scale_perf_count(10, 200, 0) != 0) return 1;

    return 0;
}
