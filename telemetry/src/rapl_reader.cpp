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

/**
 * @file rapl_reader.cpp
 * @brief Low-overhead sysfs reader for Intel RAPL energy counters.
 *
 * The producer stores raw cumulative energy snapshots. Delta computation and
 * overflow handling are intentionally left to the export/consumer side.
 */
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

        std::vector<std::string> split_comma_paths(const std::string& value) {
            std::vector<std::string> parts;
            std::size_t start = 0;
            while(start <= value.size()){
                std::size_t comma = value.find(',', start);
                std::string part = value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if(!part.empty()) parts.push_back(std::move(part));
                if(comma == std::string::npos) break;
                start = comma + 1;
            }
            return parts;
        }
    }

    namespace {
        std::runtime_error errno_error(const char* context, const std::string& path) {
            return std::runtime_error(std::string(context) + ": " + path + ": " + std::strerror(errno));
        }
    }

    //Keep the file descriptors open for the lifetime of the reader.
    //Re-opening on every sample adds significant overhead and variability.

    RaplReader::RaplReader(std::string pkg_paths, std::string dram_paths)
        : pkg_paths_raw_(std::move(pkg_paths)), dram_paths_raw_(std::move(dram_paths)) {}

    RaplReader::~RaplReader(){
        close();
    }

    void RaplReader::open() {
        if(is_open()) return;
        max_range_uj_ = 0;

        auto make_path = [](const std::string& base) {
            return base + "/energy_uj";
        };

        const std::vector<std::string> pkg_paths = detail::split_comma_paths(pkg_paths_raw_);
        if(pkg_paths.empty()){
            throw std::runtime_error("RaplReader: no RAPL package path configured");
        }
        const std::vector<std::string> dram_paths = detail::split_comma_paths(dram_paths_raw_);
        if(!dram_paths.empty() && dram_paths.size() != pkg_paths.size()){
            throw std::runtime_error(
                "RaplReader: dram_paths count (" + std::to_string(dram_paths.size()) +
                ") must match pkg_paths count (" + std::to_string(pkg_paths.size()) +
                ") when DRAM domains are requested -- partial coverage would "
                "silently undercount energy on the sockets missing a DRAM path");
        }

        // One socket wraps its energy_uj independently of the others (see the
        // class comment) -- each package keeps its own fd, previous sample and
        // max_energy_range_uj, opened once here and reused every read().
        for(const auto& path : pkg_paths){
            PackageState state{};
            state.path = path;
            const std::string energy_path = make_path(path);
            state.fd = ::open(energy_path.c_str(), O_RDONLY);
            if(state.fd < 0){
                close();
                throw errno_error("Failed to open RAPL package energy file", energy_path);
            }
            char buf[32];
            int range_fd = ::open((path + "/max_energy_range_uj").c_str(), O_RDONLY);
            if(range_fd >= 0){
                ssize_t n = ::read(range_fd, buf, sizeof(buf)-1);
                if(n > 0){
                    buf[n] = '\0';
                    uint64_t parsed = 0;
                    if(detail::parse_uint64(buf, parsed)) state.max_range_uj = parsed;
                }
                ::close(range_fd);
            }
            max_range_uj_ += state.max_range_uj;
            pkg_states_.push_back(std::move(state));
        }
        for(const auto& path : dram_paths){
            PackageState state{};
            state.path = path;
            const std::string energy_path = make_path(path);
            state.fd = ::open(energy_path.c_str(), O_RDONLY);
            if(state.fd < 0){
                close();
                throw errno_error("Failed to open RAPL DRAM energy file", energy_path);
            }
            dram_states_.push_back(std::move(state));
        }
    }

    void RaplReader::close() noexcept {
        for(auto& state : pkg_states_){
            if(state.fd >= 0){ ::close(state.fd); state.fd = -1; }
        }
        for(auto& state : dram_states_){
            if(state.fd >= 0){ ::close(state.fd); state.fd = -1; }
        }
        pkg_states_.clear();
        dram_states_.clear();
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

    void RaplReader::poll_package(PackageState& state) noexcept {
        const uint64_t raw = read_energy_uj(state.fd);
        if(state.have_previous){
            const uint64_t delta = detail::rapl_delta_uj(state.previous_raw_uj, raw, state.max_range_uj);
            state.logical_total_uj += delta;
        } else {
            state.have_previous = true;
            // First sample of this package's lifetime: the logical total
            // starts AT the raw value, not at zero, so a single-package
            // reader is numerically identical to the previous raw-passthrough
            // behaviour from the very first read() -- and a multi-package sum
            // is a genuine "cumulative energy since open()" per socket, just
            // unwrapped independently before summing.
            state.logical_total_uj = raw;
        }
        state.previous_raw_uj = raw;
    }

    bool RaplReader::read(EnergySnapshot& out) noexcept {
        if(!is_open()) return false;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        EnergySnapshot sample{};
        sample.timestamp_ns = ts.tv_sec * 1'000'000'000ULL + ts.tv_nsec;

        uint64_t pkg_total = 0;
        for(auto& state : pkg_states_){
            poll_package(state);
            pkg_total += state.logical_total_uj;
        }
        uint64_t dram_total = 0;
        for(auto& state : dram_states_){
            poll_package(state);
            dram_total += state.logical_total_uj;
        }
        sample.pkg_uj = pkg_total;
        sample.dram_uj = dram_total;
        out = sample;
        return true;
    }
}
