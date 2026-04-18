#pragma once
#include "metrics.hpp"
#include <cstdint>
#include <string>

namespace telemetry {
    namespace detail {
        bool parse_uint64(const char* text, uint64_t& out) noexcept;
        uint64_t rapl_delta_uj(uint64_t previous_uj,
                               uint64_t current_uj,
                               uint64_t max_range_uj) noexcept;
    }

    class RaplReader {
    public:
    //Domain paths for RAPL
    explicit RaplReader(std::string pkg_path, std::string dram_path = "");
    ~RaplReader();

    void open(); //opens file descriptors; throws on failure (e.g., permission issues)
    void close() noexcept;

    bool is_open() const noexcept { return pkg_fd_ >= 0; }

    //Read current snapshot of energy counters.
    //Compute deltas in consumer thread by differencing consecutive two snapshots.
    bool read(EnergySnapshot& out) noexcept;
    uint64_t max_range_uj() const noexcept { return max_range_uj_; }

    private:
    std::string pkg_path_;
    std::string dram_path_;
    int pkg_fd_ = -1;
    int dram_fd_ = -1;
    uint64_t max_range_uj_ = 0;

    static uint64_t read_energy_uj(int fd) noexcept;
    };
}
