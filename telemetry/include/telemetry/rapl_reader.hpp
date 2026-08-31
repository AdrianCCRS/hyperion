#pragma once
#include "metrics.hpp"
#include <cstdint>
#include <string>
#include <vector>

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

        /** @brief Split a comma-separated path list. Empty input -> empty vector. */
        std::vector<std::string> split_comma_paths(const std::string& value);
    }

    /**
     * @brief Low-overhead reader for Intel RAPL sysfs energy counters.
     *
     * Accepts one or several comma-separated package/DRAM domain directories
     * (one per socket actually delegated to the job -- see
     * ``orchestrator.environment.EnvironmentProfile.delegated_numa_nodes``).
     * Each package is tracked and unwrapped INDEPENDENTLY internally (its own
     * previous raw sample and its own ``max_energy_range_uj``), because two
     * sockets under similar-but-not-identical load wrap their hardware
     * counters at different times -- summing raw cumulative values first and
     * wrap-correcting the sum afterwards, with a single shared max range,
     * would silently misattribute a wrap in one package as a wrap in the
     * combined total. read() therefore returns an ALREADY-UNWRAPPED, ever
     * -increasing logical total (sum of each package's own corrected delta
     * accumulated since open()), not a raw hardware snapshot. Every existing
     * consumer of EnergySnapshot only ever computes ``current - previous``
     * downstream (see experiment_utils.cpp::next_rapl_delta), so this stays
     * a drop-in-compatible value: for a single package it is numerically
     * equivalent to the previous raw-snapshot behaviour from the first read
     * onward, and for several packages it is the only way to keep that
     * subtraction correct.
     */
    class RaplReader {
    public:
        /**
         * @param pkg_paths One or more (comma-separated) RAPL package domain
         *     directories, each containing energy_uj.
         * @param dram_paths Optional, one or more (comma-separated) RAPL DRAM
         *     domain directories. Must be empty or match pkg_paths in count.
         */
        explicit RaplReader(std::string pkg_paths, std::string dram_paths = "");
        ~RaplReader();

        /** @brief Open RAPL energy file descriptors. Throws on package failure. */
        void open();

        /** @brief Close every opened descriptor. */
        void close() noexcept;

        /** @return true when at least the package energy_uj files are open. */
        bool is_open() const noexcept { return !pkg_states_.empty() && pkg_states_.front().fd >= 0; }

        /**
         * @brief Read an already-unwrapped, monotonic energy snapshot summed
         * across every configured package/DRAM domain.
         *
         * This function avoids throwing and is intended for the producer loop.
         */
        bool read(EnergySnapshot& out) noexcept;

        /** @return sum of max_energy_range_uj across package domains, or zero.
         * Diagnostic only -- read()'s own values never wrap, this is not
         * required for correct delta computation downstream. */
        uint64_t max_range_uj() const noexcept { return max_range_uj_; }

        /** @return number of package domains configured (sockets covered). */
        std::size_t package_count() const noexcept { return pkg_states_.size(); }

    private:
        struct PackageState {
            std::string path;
            int fd = -1;
            uint64_t max_range_uj = 0;
            uint64_t previous_raw_uj = 0;
            uint64_t logical_total_uj = 0;
            bool have_previous = false;
        };

        std::string pkg_paths_raw_;
        std::string dram_paths_raw_;
        std::vector<PackageState> pkg_states_;
        std::vector<PackageState> dram_states_;
        uint64_t max_range_uj_ = 0;

        /** Read one sysfs energy counter from an already-open descriptor. */
        static uint64_t read_energy_uj(int fd) noexcept;

        /** Poll one package: read raw, wrap-correct against its own previous
         * sample, accumulate into its own logical_total_uj. */
        static void poll_package(PackageState& state) noexcept;
    };
}
