#include "telemetry/uncore_reader.hpp"
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <sched.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

/**
 * @file uncore_reader.cpp
 * @brief `perf` CLI-backed reader for uncore_imc CAS_COUNT_READ/WRITE.
 *
 * See uncore_reader.hpp for why this shells out to `perf stat` instead of
 * calling perf_event_open() directly (ARC-118, superseding the ARC-116
 * design that never worked under this cluster's CAP_PERFMON grant, which is
 * scoped to the `perf` binary's file capability, not to arbitrary processes).
 */
namespace telemetry {
    namespace detail {
        bool parse_uncore_event_format(const std::string& text, uint64_t& config) noexcept {
            bool has_event = false;
            uint64_t event = 0, umask = 0, cmask = 0;
            bool edge = false, inv = false;

            std::stringstream ss(text);
            std::string term;
            while(std::getline(ss, term, ',')) {
                const auto eq = term.find('=');
                if(eq == std::string::npos) continue;
                std::string key = term.substr(0, eq);
                std::string value = term.substr(eq + 1);
                const auto trim = [](std::string& s) {
                    while(!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
                    size_t start = 0;
                    while(start < s.size() && s[start] == ' ') ++start;
                    s = s.substr(start);
                };
                trim(key);
                trim(value);
                if(value.empty()) continue;

                errno = 0;
                char* end = nullptr;
                unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
                if(end == value.c_str() || errno == ERANGE) continue;

                if(key == "event") { event = parsed; has_event = true; }
                else if(key == "umask") { umask = parsed; }
                else if(key == "cmask") { cmask = parsed; }
                else if(key == "edge") { edge = parsed != 0; }
                else if(key == "inv") { inv = parsed != 0; }
            }
            if(!has_event) return false;

            config = (event & 0xFFu)
                   | ((umask & 0xFFu) << 8)
                   | (edge ? (1ull << 18) : 0)
                   | (inv ? (1ull << 23) : 0)
                   | ((cmask & 0xFFu) << 24);
            return true;
        }

        std::string build_perf_stat_event_list(const std::vector<UncoreBoxEvents>& boxes) {
            std::string out;
            for(const auto& box : boxes) {
                if(!out.empty()) out += ',';
                out += box.pmu_name + "/" + box.read_format + "/,"
                     + box.pmu_name + "/" + box.write_format + "/";
            }
            return out;
        }

        PerfStatCsvLine parse_perf_stat_csv_line(const std::string& line, char sep) noexcept {
            PerfStatCsvLine result{};
            std::vector<std::string> fields;
            size_t start = 0;
            while(true) {
                const size_t pos = line.find(sep, start);
                fields.push_back(line.substr(start, pos == std::string::npos ? std::string::npos : pos - start));
                if(pos == std::string::npos) break;
                start = pos + 1;
            }
            // interval-time,value,unit,event,time-running,percent-running[,...]
            if(fields.size() < 2) return result;

            errno = 0;
            char* end = nullptr;
            const double time_s = std::strtod(fields[0].c_str(), &end);
            if(end == fields[0].c_str() || errno == ERANGE) return result;
            result.interval_time_s = time_s;

            // "<not counted>"/"<not supported>"/empty -- a real perf failure
            // mode (event rejected, still no permission, box busy), never
            // treated as a numeric zero.
            std::string value_field = fields[1];
            size_t first = value_field.find_first_not_of(' ');
            if(first == std::string::npos) return result;
            value_field = value_field.substr(first);
            if(value_field.empty() || value_field[0] == '<') return result;

            errno = 0;
            end = nullptr;
            unsigned long long parsed = std::strtoull(value_field.c_str(), &end, 10);
            if(end == value_field.c_str() || errno == ERANGE) return result;

            result.value = parsed;
            result.valid = true;
            return result;
        }
    }

    namespace {
        bool read_sysfs_text(const std::string& path, std::string& out) noexcept {
            std::ifstream in(path);
            if(!in.is_open()) return false;
            std::getline(in, out);
            return !out.empty();
        }

        // ARC-118: discovery of real, configurable uncore_imc_<N> boxes --
        // same sysfs scan and the same "free_running" exclusion learned on
        // pacca in ARC-117 (those boxes expose a completely different,
        // non-configurable event format and would poison the event list).
        std::vector<detail::UncoreBoxEvents> discover_uncore_boxes() {
            std::vector<detail::UncoreBoxEvents> boxes;
            static const char* kDevicesRoot = "/sys/bus/event_source/devices";
            DIR* dir = ::opendir(kDevicesRoot);
            if(dir == nullptr) return boxes;

            struct dirent* entry;
            while((entry = ::readdir(dir)) != nullptr) {
                const std::string name = entry->d_name;
                if(name.rfind("uncore_imc", 0) != 0) continue;
                if(name.find("free_running") != std::string::npos) continue;

                const std::string pmu_dir = std::string(kDevicesRoot) + "/" + name;
                std::string read_fmt, write_fmt;
                uint64_t dummy_config = 0;
                const bool ok =
                    read_sysfs_text(pmu_dir + "/events/cas_count_read", read_fmt) &&
                    read_sysfs_text(pmu_dir + "/events/cas_count_write", write_fmt) &&
                    detail::parse_uncore_event_format(read_fmt, dummy_config) &&
                    detail::parse_uncore_event_format(write_fmt, dummy_config);
                if(!ok) continue; // best-effort per box at discovery time; a box this malformed is skipped, not fatal

                boxes.push_back({name, read_fmt, write_fmt});
            }
            ::closedir(dir);
            return boxes;
        }
    }

    UncoreReader::UncoreReader(long interval_ms, int pin_cpu)
        // perf stat -I has no reliable sub-10ms guarantee across builds;
        // floor it rather than pass through a value that might silently
        // get rejected or rounded by perf itself.
        : interval_ms_(interval_ms < 10 ? 10 : interval_ms), pin_cpu_(pin_cpu) {}

    UncoreReader::~UncoreReader() {
        close();
    }

    void UncoreReader::open() noexcept {
        if(is_open()) return;

        const std::vector<detail::UncoreBoxEvents> boxes = discover_uncore_boxes();
        if(boxes.empty()) return;

        const std::string event_list = detail::build_perf_stat_event_list(boxes);
        if(event_list.empty()) return;

        term_is_write_.clear();
        term_is_write_.reserve(boxes.size() * 2);
        for(size_t i = 0; i < boxes.size(); ++i) {
            term_is_write_.push_back(false); // read term first
            term_is_write_.push_back(true);  // write term second
        }

        int pipe_fds[2];
        if(::pipe(pipe_fds) != 0) return;

        const std::string interval_str = std::to_string(interval_ms_);
        struct timespec launch_ts;
        clock_gettime(CLOCK_MONOTONIC, &launch_ts);
        const pid_t pid = ::fork();
        if(pid < 0) {
            ::close(pipe_fds[0]);
            ::close(pipe_fds[1]);
            return;
        }

        if(pid == 0) {
            // ARC-131: pin the child away from the workload's own
            // measurement CPUs BEFORE exec, if requested -- see the
            // constructor doc for why (scheduling contention with
            // FP_ARITH_INST_RETIRED, found empirically on paccaA100).
            // Best-effort: a failed sched_setaffinity here is not fatal to
            // uncore measurement itself, just loses the isolation.
            if(pin_cpu_ >= 0) {
                cpu_set_t cpu_set;
                CPU_ZERO(&cpu_set);
                CPU_SET(static_cast<unsigned>(pin_cpu_), &cpu_set);
                ::sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
            }
            // Child: perf stat writes its periodic report to stderr by
            // default (no --log-fd redirection needed, avoids depending on
            // an option that may not exist in every perf build).
            ::close(pipe_fds[0]);
            if(::dup2(pipe_fds[1], STDERR_FILENO) < 0) _exit(127);
            ::close(pipe_fds[1]);
            int devnull = ::open("/dev/null", O_RDONLY);
            if(devnull >= 0) { ::dup2(devnull, STDIN_FILENO); ::close(devnull); }

            // Semicolon field separator: perf's own raw event syntax
            // ("event=0x04,umask=0x0f") contains commas, which would
            // collide with a comma field separator (ARC-118).
            std::vector<char*> argv = {
                const_cast<char*>("perf"),
                const_cast<char*>("stat"),
                const_cast<char*>("-a"),
                const_cast<char*>("-I"),
                const_cast<char*>(interval_str.c_str()),
                const_cast<char*>("-x"),
                const_cast<char*>(";"),
                const_cast<char*>("-e"),
                const_cast<char*>(event_list.c_str()),
                nullptr,
            };
            ::execvp("perf", argv.data());
            _exit(127); // perf missing / exec failed
        }

        // Parent.
        ::close(pipe_fds[1]);
        const int flags = ::fcntl(pipe_fds[0], F_GETFL, 0);
        if(flags >= 0) ::fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK);

        // Give perf a moment to either start counting or fail (missing
        // binary, EACCES still in effect, malformed event list) before
        // trusting this backend. A short, bounded wait -- this only runs
        // once per Collector::start(), not on the sampling hot path.
        struct timespec wait_ts{0, 300'000'000L}; // 300ms
        ::nanosleep(&wait_ts, nullptr);

        int status = 0;
        const pid_t reaped = ::waitpid(pid, &status, WNOHANG);
        if(reaped == pid) {
            // Child already exited -- perf failed immediately (not found,
            // still no permission, bad event syntax). Degrade to
            // unavailable, same contract as every other optional backend.
            ::close(pipe_fds[0]);
            return;
        }

        pipe_fd_ = pipe_fds[0];
        child_pid_ = pid;
        box_count_ = boxes.size();
        launch_time_ns_ = static_cast<long long>(launch_ts.tv_sec) * 1'000'000'000LL + launch_ts.tv_nsec;
    }

    void UncoreReader::close() noexcept {
        if(child_pid_ > 0) {
            ::kill(child_pid_, SIGTERM);
            int status = 0;
            // Bounded wait: if perf does not exit promptly, do not block
            // the caller (Collector::stop()/close_readers() must be able
            // to return) -- escalate to SIGKILL and reap without blocking further.
            for(int i = 0; i < 20; ++i) {
                if(::waitpid(child_pid_, &status, WNOHANG) == child_pid_) { child_pid_ = -1; break; }
                struct timespec t{0, 10'000'000L}; // 10ms
                ::nanosleep(&t, nullptr);
            }
            if(child_pid_ > 0) {
                ::kill(child_pid_, SIGKILL);
                ::waitpid(child_pid_, &status, 0);
                child_pid_ = -1;
            }
        }
        if(pipe_fd_ >= 0) {
            ::close(pipe_fd_);
            pipe_fd_ = -1;
        }
        term_is_write_.clear();
        box_count_ = 0;
        line_buffer_.clear();
        bucket_time_s_ = -1.0;
        bucket_seen_ = 0;
        bucket_read_interval_ = bucket_write_interval_ = 0;
        bucket_has_data_ = false;
        bucket_any_valid_ = false;
        pending_.clear();
    }

    void UncoreReader::fold_line(const std::string& line) {
        const detail::PerfStatCsvLine parsed = detail::parse_perf_stat_csv_line(line, ';');

        if(bucket_has_data_ && parsed.interval_time_s != bucket_time_s_) {
            // A new interval started: the previous bucket (already a
            // per-interval delta, ARC-119 -- see UncoreSnapshot doc) is
            // complete and stands on its own, no differencing against any
            // other reading.
            //
            // ARC-120: `perf stat` keeps running even when EVERY term comes
            // back "<not counted>"/"<not supported>" (confirmed live on
            // pacca under the still-unresolved CAP_PERFMON gap, ARC-117) --
            // is_open() alone cannot see that, so bucket_any_valid_ is what
            // tells the difference between a real zero and a fully-invalid
            // interval downstream.
            UncoreSnapshot snapshot{};
            snapshot.timestamp_ns = static_cast<uint64_t>(
                launch_time_ns_ + static_cast<long long>(bucket_time_s_ * 1e9)
            );
            snapshot.cas_count_read_interval = bucket_read_interval_;
            snapshot.cas_count_write_interval = bucket_write_interval_;
            snapshot.interval_valid = bucket_any_valid_;
            pending_.push_back(snapshot);
            bucket_seen_ = 0;
            bucket_read_interval_ = bucket_write_interval_ = 0;
            bucket_any_valid_ = false;
        }
        bucket_time_s_ = parsed.interval_time_s;
        bucket_has_data_ = true;

        if(!term_is_write_.empty() && parsed.valid) {
            const size_t index = bucket_seen_ % term_is_write_.size();
            if(term_is_write_[index]) bucket_write_interval_ += parsed.value;
            else bucket_read_interval_ += parsed.value;
            bucket_any_valid_ = true;
        }
        ++bucket_seen_;
    }

    bool UncoreReader::read(UncoreSnapshot& out) noexcept {
        if(!is_open()) return false;

        char buf[4096];
        while(true) {
            const ssize_t n = ::read(pipe_fd_, buf, sizeof(buf));
            if(n > 0) {
                line_buffer_.append(buf, static_cast<size_t>(n));
                continue;
            }
            if(n == 0) {
                // perf exited and closed the pipe -- backend is gone.
                close();
                return false;
            }
            // n < 0: EAGAIN/EWOULDBLOCK means no more data right now, stop draining.
            break;
        }

        size_t pos;
        while((pos = line_buffer_.find('\n')) != std::string::npos) {
            const std::string line = line_buffer_.substr(0, pos);
            line_buffer_.erase(0, pos + 1);
            if(!line.empty()) fold_line(line);
        }

        if(pending_.empty()) return false;
        out = pending_.front();
        pending_.pop_front();
        return true;
    }
}
