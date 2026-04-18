#include "telemetry/rapl_reader.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <ctime>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace telemetry {
    namespace detail {
        bool parse_uint64(const char* text, uint64_t& out) noexcept {
            if(text == nullptr) return false;

            errno = 0;
            char* end = nullptr;
            unsigned long long value = std::strtoull(text, &end, 10);
            if(end == text || errno == ERANGE) return false;

            while(*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r'){
                ++end;
            }
            if(*end != '\0') return false;

            out = static_cast<uint64_t>(value);
            return true;
        }

        uint64_t rapl_delta_uj(uint64_t previous_uj,
                               uint64_t current_uj,
                               uint64_t max_range_uj) noexcept {
            if(current_uj >= previous_uj) return current_uj - previous_uj;
            if(max_range_uj == 0 || previous_uj > max_range_uj) return 0;
            return (max_range_uj - previous_uj) + current_uj;
        }
    }

    namespace {
        std::runtime_error errno_error(const char* context, const std::string& path) {
            return std::runtime_error(std::string(context) + ": " + path + ": " + std::strerror(errno));
        }
    }

    //Keep the file descriptors open for the lifetime of the reader.
    //Re-opening on every sample adds significant overhead and variability.

    RaplReader::RaplReader(std::string pkg_path, std::string dram_path)
        : pkg_path_(std::move(pkg_path)), dram_path_(std::move(dram_path)) {}

    RaplReader::~RaplReader(){
        close();
    }

    void RaplReader::open() {
        if(is_open()) return;
        max_range_uj_ = 0;

        auto make_path = [](const std::string& base) {
            return base + "/energy_uj";
        }; //Lambda to construct the full path to the energy file (should be used once at initialization)
        const std::string pkg_energy_path = make_path(pkg_path_);
        pkg_fd_ = ::open(pkg_energy_path.c_str(), O_RDONLY);
        if(pkg_fd_ < 0) throw errno_error("Failed to open RAPL package energy file", pkg_energy_path);
        if(!dram_path_.empty()){
            const std::string dram_energy_path = make_path(dram_path_);
            dram_fd_ = ::open(dram_energy_path.c_str(), O_RDONLY);
            if(dram_fd_ < 0){
                close();
                throw errno_error("Failed to open RAPL DRAM energy file", dram_energy_path);
            }
        }
        //Read max range for wrap detection
        char buf[32]; int fd;
        fd = ::open((pkg_path_ + "/max_energy_range_uj").c_str(), O_RDONLY);
        if(fd >= 0){
            ssize_t n = ::read(fd, buf, sizeof(buf)-1); //Number of bytes read
            if(n > 0){
                buf[n] = '\0'; //Null-terminate
                uint64_t parsed = 0;
                if(detail::parse_uint64(buf, parsed)){
                    max_range_uj_ = parsed;
                }
            }
            ::close(fd);
        }
    }

    void RaplReader::close() noexcept {
        if(pkg_fd_ >= 0){
            ::close(pkg_fd_);
            pkg_fd_ = -1;
        }
        if(dram_fd_ >= 0){
            ::close(dram_fd_);
            dram_fd_ = -1;
        }
    }

    uint64_t RaplReader::read_energy_uj(int fd) noexcept {
        if(fd < 0) return 0;

        char buf[32];
        if(lseek(fd, 0, SEEK_SET) < 0) return 0; //We make rewind the file to ensure we read the latest value (since these files are updated by the kernel)
        //Mantaining the file open and just rewinding is much faster than closing and reopening on every read.
        ssize_t n = ::read(fd, buf, sizeof(buf)-1);
        if(n <= 0) return 0; //On error, return 0 (could also throw, but we want to be resilient to transient read errors)
        buf[n] = '\0'; //Null-terminate
        uint64_t parsed = 0;
        return detail::parse_uint64(buf, parsed) ? parsed : 0;
    }

    bool RaplReader::read(EnergySnapshot& out) noexcept {
        if(!is_open()) return false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        EnergySnapshot sample{};
        sample.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;
        sample.pkg_uj = read_energy_uj(pkg_fd_);
        sample.dram_uj = dram_fd_ >= 0 ? read_energy_uj(dram_fd_) : 0;
        out = sample;
        return true;
    }

    
}
