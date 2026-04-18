#include "telemetry/rapl_reader.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <utility>

namespace telemetry {
    //Keep the file descriptors open for the lifetime of the reader.
    //Re-opening on every sample adds significant overhead and variability.

    RaplReader::RaplReader(std::string pkg_path, std::string dram_path)
        : pkg_path_(std::move(pkg_path)), dram_path_(std::move(dram_path)) {}

    RaplReader::~RaplReader(){
        close();
    }

    void RaplReader::open() {
        if(is_open()) return;

        auto make_path = [](const std::string& base) {
            return base + "/energy_uj";
        }; //Lambda to construct the full path to the energy file (should be used once at initialization)
        pkg_fd_ = ::open(make_path(pkg_path_).c_str(), O_RDONLY);
        if(pkg_fd_ < 0) throw std::runtime_error("Failed to open RAPL package energy file");
        if(!dram_path_.empty()){
            dram_fd_ = ::open(make_path(dram_path_).c_str(), O_RDONLY);
            if(dram_fd_ < 0){
                close();
                throw std::runtime_error("Failed to open RAPL DRAM energy file");
            }
        }
        //Read max range for wrap detection
        char buf[32]; int fd;
        fd = ::open((pkg_path_ + "/max_energy_range_uj").c_str(), O_RDONLY);
        if(fd >= 0){
            ssize_t n = ::read(fd, buf, sizeof(buf)-1); //Number of bytes read
            if(n > 0){
                buf[n] = '\0'; //Null-terminate
                max_range_uj_ = std::stoull(buf); //Convert to uint64_t
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
        char buf[32];
        lseek(fd, 0, SEEK_SET); //We make rewind the file to ensure we read the latest value (since these files are updated by the kernel)
        //Mantaining the file open and just rewinding is much faster than closing and reopening on every read.
        ssize_t n = ::read(fd, buf, sizeof(buf)-1);
        if(n <= 0) return 0; //On error, return 0 (could also throw, but we want to be resilient to transient read errors)
        buf[n] = '\0'; //Null-terminate
        try {
            return std::stoull(buf); //Convert to uint64_t
        } catch (...) {
            return 0;
        }
    }

    bool RaplReader::read(EnergySnapshot& out) noexcept {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        out.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;
        out.pkg_uj = read_energy_uj(pkg_fd_);
        out.dram_uj = dram_fd_ >= 0 ? read_energy_uj(dram_fd_) : 0;
        return true;
    }

    
}
