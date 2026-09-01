#include "telemetry/rapl_reader.hpp"

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
    uint64_t parsed = 0;
    if(!telemetry::detail::parse_uint64("123\n", parsed) || parsed != 123) return 1;
    if(telemetry::detail::parse_uint64("12x\n", parsed)) return 1;

    if(telemetry::detail::rapl_delta_uj(100, 150, 1000) != 50) return 1;
    if(telemetry::detail::rapl_delta_uj(90, 5, 100) != 15) return 1;
    if(telemetry::detail::rapl_delta_uj(90, 5, 0) != 0) return 1;

    const auto base = std::filesystem::temp_directory_path() /
                      ("telemetry-rapl-test-" + std::to_string(::getpid()));
    const auto pkg = base / "pkg";
    const auto dram = base / "dram";
    std::filesystem::create_directories(pkg);
    std::filesystem::create_directories(dram);

    if(!write_file(pkg / "energy_uj", "1000\n")) return 1;
    if(!write_file(pkg / "max_energy_range_uj", "5000\n")) return 1;
    if(!write_file(dram / "energy_uj", "250\n")) return 1;

    telemetry::RaplReader reader(pkg.string(), dram.string());
    reader.open();
    if(!reader.is_open()) return 1;
    if(reader.max_range_uj() != 5000) return 1;

    telemetry::EnergySnapshot sample{};
    if(!reader.read(sample)) return 1;
    if(sample.pkg_uj != 1000) return 1;
    if(sample.dram_uj != 250) return 1;

    reader.close();
    if(reader.is_open()) return 1;
    if(reader.read(sample)) return 1;

    std::filesystem::remove_all(base);
    return 0;
}
