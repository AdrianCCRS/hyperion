#include "telemetry/cpu_freq_reader.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {
    bool write_file(const std::filesystem::path& path, const char* value) {
        std::ofstream out(path);
        out << value;
        return static_cast<bool>(out);
    }
}

int main() {
    // Empty path (ARC-135 default: disabled) never opens.
    telemetry::CpuFreqReader disabled("");
    disabled.open();
    if(disabled.is_open()) return 1;
    uint64_t out = 999;
    if(disabled.read(out)) return 1;
    if(out != 999) return 1; // untouched on failure

    const auto base = std::filesystem::temp_directory_path() /
                      ("telemetry-cpufreq-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(base);
    const auto path = base / "scaling_cur_freq";
    if(!write_file(path, "2200000\n")) return 1;

    telemetry::CpuFreqReader reader(path.string());
    reader.open();
    if(!reader.is_open()) return 1;

    uint64_t khz = 0;
    if(!reader.read(khz)) return 1;
    if(khz != 2200000) return 1;

    // Re-read after the file changes -- the reader must rewind, not cache.
    if(!write_file(path, "800000\n")) return 1;
    if(!reader.read(khz)) return 1;
    if(khz != 800000) return 1;

    reader.close();
    if(reader.is_open()) return 1;
    if(reader.read(khz)) return 1;

    // Missing file: open() degrades to is_open()==false, never throws.
    telemetry::CpuFreqReader missing((base / "does_not_exist").string());
    missing.open();
    if(missing.is_open()) return 1;

    std::filesystem::remove_all(base);
    return 0;
}
