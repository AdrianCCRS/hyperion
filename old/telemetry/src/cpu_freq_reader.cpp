#include "telemetry/cpu_freq_reader.hpp"
#include "telemetry/rapl_reader.hpp" // detail::parse_uint64
#include <fcntl.h>
#include <unistd.h>

namespace telemetry {

    CpuFreqReader::CpuFreqReader(std::string sysfs_path)
        : sysfs_path_(std::move(sysfs_path)) {}

    CpuFreqReader::~CpuFreqReader() {
        close();
    }

    void CpuFreqReader::open() noexcept {
        if(is_open() || sysfs_path_.empty()) return;
        fd_ = ::open(sysfs_path_.c_str(), O_RDONLY);
        // Degrades to is_open()==false on failure, same contract as every
        // other optional backend in this codebase (RaplReader is the
        // exception -- it throws on package failure -- but this signal is
        // purely diagnostic, never required for a run to be measurable).
    }

    void CpuFreqReader::close() noexcept {
        if(fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool CpuFreqReader::read(uint64_t& khz_out) noexcept {
        if(fd_ < 0) return false;
        if(::lseek(fd_, 0, SEEK_SET) < 0) return false;
        char buf[32];
        ssize_t n = ::read(fd_, buf, sizeof(buf) - 1);
        if(n <= 0) return false;
        buf[n] = '\0';
        uint64_t parsed = 0;
        if(!detail::parse_uint64(buf, parsed)) return false;
        khz_out = parsed;
        return true;
    }
}
