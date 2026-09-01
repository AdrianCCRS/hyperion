#pragma once
#include <cstdint>
#include <string>

namespace telemetry {
    /**
     * @brief Low-overhead reader for cpufreq's scaling_cur_freq sysfs file.
     *
     * ARC-135: reads the CURRENT frequency of one representative delegated
     * CPU on every producer tick, the same cadence as the PMU counters --
     * unlike the single post-hoc Python snapshot it replaces (taken after
     * the workload process already exited, campaign.py), this samples
     * during the actual measured execution, once per CPU window, exactly
     * like RaplReader/UncoreReader already do for their own signals.
     */
    class CpuFreqReader {
    public:
        /** @param sysfs_path Full path to one CPU's cpufreq/scaling_cur_freq. */
        explicit CpuFreqReader(std::string sysfs_path);
        ~CpuFreqReader();

        /** @brief Open the sysfs file descriptor. Never throws; degrades to is_open()==false. */
        void open() noexcept;

        /** @brief Close the descriptor, if open. */
        void close() noexcept;

        /** @return true while the sysfs file descriptor is open. */
        bool is_open() const noexcept { return fd_ >= 0; }

        /**
         * @brief Read the current frequency in kHz.
         * @return true on a successful parse; out is left untouched on failure.
         */
        bool read(uint64_t& khz_out) noexcept;

    private:
        std::string sysfs_path_;
        int fd_ = -1;
    };
}
