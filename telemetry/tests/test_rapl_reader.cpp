#include "telemetry/rapl_reader.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

    // --- Multi-socket: dos paquetes, sin DRAM en ninguno ---------------
    {
        const auto base2 = std::filesystem::temp_directory_path() /
                           ("telemetry-rapl-test2-" + std::to_string(::getpid()));
        const auto pkg0 = base2 / "pkg0";
        const auto pkg1 = base2 / "pkg1";
        std::filesystem::create_directories(pkg0);
        std::filesystem::create_directories(pkg1);
        if(!write_file(pkg0 / "energy_uj", "1000\n")) return 1;
        if(!write_file(pkg0 / "max_energy_range_uj", "5000\n")) return 1;
        if(!write_file(pkg1 / "energy_uj", "2000\n")) return 1;
        if(!write_file(pkg1 / "max_energy_range_uj", "6000\n")) return 1;

        telemetry::RaplReader multi(pkg0.string() + "," + pkg1.string());
        multi.open();
        if(!multi.is_open()) return 1;
        if(multi.package_count() != 2) return 1;
        if(multi.max_range_uj() != 5000 + 6000) return 1;

        telemetry::EnergySnapshot s1{};
        if(!multi.read(s1)) return 1;
        // Primera lectura: cada paquete arranca su total logico EN su valor
        // crudo (no en cero) -- un solo socket queda identico al
        // comportamiento anterior desde la primera lectura.
        if(s1.pkg_uj != 1000 + 2000) return 1;
        if(s1.dram_uj != 0) return 1;

        // Paquete 0 avanza normal (sin wrap); paquete 1 SI envuelve su
        // contador (2000 -> 500 < previo, con max_range=6000).
        if(!write_file(pkg0 / "energy_uj", "1300\n")) return 1;  // +300
        if(!write_file(pkg1 / "energy_uj", "500\n")) return 1;   // wrap: (6000-2000)+500=4500

        telemetry::EnergySnapshot s2{};
        if(!multi.read(s2)) return 1;
        const uint64_t expected_pkg_total = (1000 + 300) + (2000 + 4500);
        if(s2.pkg_uj != expected_pkg_total) return 1;
        // Confirma que el total es MONOTONO creciente pese al wrap fisico
        // de un solo paquete -- exactamente lo que next_rapl_delta aguas
        // abajo necesita para no requerir logica especial de multi-socket.
        if(s2.pkg_uj <= s1.pkg_uj) return 1;

        multi.close();
        std::filesystem::remove_all(base2);
    }

    // --- Multi-socket: pkg_paths y dram_paths con conteos distintos ----
    {
        const auto base3 = std::filesystem::temp_directory_path() /
                           ("telemetry-rapl-test3-" + std::to_string(::getpid()));
        const auto pkg0 = base3 / "pkg0";
        const auto pkg1 = base3 / "pkg1";
        const auto dram0 = base3 / "dram0";
        std::filesystem::create_directories(pkg0);
        std::filesystem::create_directories(pkg1);
        std::filesystem::create_directories(dram0);
        write_file(pkg0 / "energy_uj", "1\n");
        write_file(pkg1 / "energy_uj", "1\n");
        write_file(dram0 / "energy_uj", "1\n");

        telemetry::RaplReader mismatched(pkg0.string() + "," + pkg1.string(), dram0.string());
        bool threw = false;
        try { mismatched.open(); } catch(const std::exception&) { threw = true; }
        if(!threw) return 1;  // cobertura parcial de DRAM debe rechazarse, no degradar en silencio

        std::filesystem::remove_all(base3);
    }

    return 0;
}
