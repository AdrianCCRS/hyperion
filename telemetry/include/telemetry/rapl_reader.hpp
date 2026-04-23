#pragma once
#include "metrics.hpp"
#include <cstdint>
#include <string>

namespace telemetry {
    namespace detail {
        /** @brief Parse a sysfs unsigned integer without throwing. */
        bool parse_uint64(const char* text, uint64_t& out) noexcept;

        /**
         * @brief Compute a RAPL delta with optional wrap-around range.
         *
         * Returns zero when current < previous and the max range is unknown or
         * inconsistent. Experiment export code adds a validity flag around this
         * primitive so invalid deltas do not enter datasets silently.
         */
        uint64_t rapl_delta_uj(uint64_t previous_uj,
                               uint64_t current_uj,
                               uint64_t max_range_uj) noexcept;
    }

    /**
     * @brief Low-overhead reader for Intel RAPL sysfs energy counters.
     *
     * The reader opens energy_uj files once and reuses file descriptors. read()
     * returns raw cumulative microjoule snapshots; consumers/exporters should
     * compute deltas and handle wrap-around outside the producer hot path.
     */
    class RaplReader {
    public:
        /**
         * @param pkg_path RAPL package domain directory containing energy_uj.
         * @param dram_path Optional RAPL DRAM domain directory.
         */
        explicit RaplReader(std::string pkg_path, std::string dram_path = "");
        ~RaplReader();

        /** @brief Open RAPL energy file descriptors. Throws on package failure. */
        void open();

        /** @brief Close every opened descriptor. */
        void close() noexcept;

        /** @return true when package energy_uj is open. */
        bool is_open() const noexcept { return pkg_fd_ >= 0; }

        /**
         * @brief Read raw package and optional DRAM energy snapshots.
         *
         * This function avoids throwing and is intended for the producer loop.
         */
        bool read(EnergySnapshot& out) noexcept;

        /** @return package max_energy_range_uj read during open(), or zero. */
        uint64_t max_range_uj() const noexcept { return max_range_uj_; }

    private:
        std::string pkg_path_;
        std::string dram_path_;
        int pkg_fd_ = -1;
        int dram_fd_ = -1;
        uint64_t max_range_uj_ = 0;

        /** Read one sysfs energy counter from an already-open descriptor. */
        static uint64_t read_energy_uj(int fd) noexcept;
    };
}
