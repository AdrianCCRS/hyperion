#include "telemetry/uncore_reader.hpp"

int main() {
    // --- parse_uncore_event_format (unchanged from ARC-116, still used to
    // validate sysfs event strings before trusting them in a perf argv) ---
    uint64_t config = 0;
    if(!telemetry::detail::parse_uncore_event_format("event=0x04,umask=0x03\n", config)) return 1;
    if(config != (0x04u | (0x03u << 8))) return 1;
    if(telemetry::detail::parse_uncore_event_format("umask=0x03", config)) return 1;

    // --- build_perf_stat_event_list ---
    {
        std::vector<telemetry::detail::UncoreBoxEvents> boxes = {
            {"uncore_imc_0", "event=0x04,umask=0x0f", "event=0x04,umask=0x30"},
            {"uncore_imc_1", "event=0x04,umask=0x0f", "event=0x04,umask=0x30"},
        };
        const std::string list = telemetry::detail::build_perf_stat_event_list(boxes);
        const std::string expected =
            "uncore_imc_0/event=0x04,umask=0x0f/,uncore_imc_0/event=0x04,umask=0x30/,"
            "uncore_imc_1/event=0x04,umask=0x0f/,uncore_imc_1/event=0x04,umask=0x30/";
        if(list != expected) return 1;

        if(!telemetry::detail::build_perf_stat_event_list({}).empty()) return 1;
    }

    // --- parse_perf_stat_csv_line ---
    {
        // A realistic `perf stat -I <ms> -x;` line.
        auto line = telemetry::detail::parse_perf_stat_csv_line(
            "1.000234567;10245;;uncore_imc_0/event=0x04,umask=0x0f/;1000198123;100.00;;", ';');
        if(!line.valid) return 1;
        if(line.value != 10245) return 1;
        if(line.interval_time_s < 1.0002 || line.interval_time_s > 1.0003) return 1;

        // "<not counted>" (permission/scheduling failure) must never read as 0.
        auto not_counted = telemetry::detail::parse_perf_stat_csv_line(
            "2.000123456;<not counted>;;uncore_imc_1/event=0x04,umask=0x30/;0;0.00;;", ';');
        if(not_counted.valid) return 1;
        // interval time is still parsed even when the value is not -- needed
        // for bucket-boundary detection regardless of which term failed.
        if(not_counted.interval_time_s < 1.999 || not_counted.interval_time_s > 2.001) return 1;

        // A comma inside the raw event descriptor must not break parsing --
        // this is exactly why ';' is used as the field separator, not ','.
        auto with_comma_event = telemetry::detail::parse_perf_stat_csv_line(
            "3.000000000;500;;uncore_imc_2/event=0x04,umask=0x0f/;1000000000;100.00;;", ';');
        if(!with_comma_event.valid || with_comma_event.value != 500) return 1;

        // Malformed line (no fields at all).
        auto malformed = telemetry::detail::parse_perf_stat_csv_line("", ';');
        if(malformed.valid) return 1;
    }

    // --- UncoreReader end-to-end: this dev machine has no uncore_imc
    // access (either no perf, or perf without CAP_PERFMON/root) --
    // open() must degrade to unavailable, never throw or hang. ---
    {
        telemetry::UncoreReader reader;
        reader.open();
        telemetry::UncoreSnapshot sample{};
        if(!reader.is_open()) {
            if(reader.read(sample)) return 1;
            if(reader.box_count() != 0) return 1;
        }
        reader.close();
        if(reader.is_open()) return 1;
    }

    return 0;
}
