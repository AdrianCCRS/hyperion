#include "telemetry/collector.hpp"
#include "telemetry/experiment_utils.hpp"
#include "telemetry/rapl_reader.hpp"

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sched.h>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

/**
 * @file telemetry_kernel_launcher.cpp
 * @brief Manual experiment runner for CPU multithreaded telemetry captures.
 *
 * The launcher owns orchestration, not kernel work. For each repetition it runs
 * one baseline child and then one telemetry child. Every child stops itself
 * with SIGSTOP right after fork; the parent opens perf on that exact PID with
 * inherit=1 (optionally also moving it into a delegated cgroup for isolation)
 * before sending SIGCONT, so no workload instruction runs unmeasured. All
 * samples are exported after the measured window finishes.
 */
namespace {
    namespace fs = std::filesystem;

    /**
     * @brief Complete launcher configuration parsed from CLI arguments.
     *
     * Names deliberately separate measurement scope (perf_cpus) from scheduling
     * policy (pin_workload_cpus, pin_workers, collector_cpu, consumer_cpu).
     */
    struct Options {
        std::string kernel = "stream_triad";
        bool kernel_explicit = false;
        size_t size = 1'000'000;
        int iterations = 10;
        int warmup = 1;
        int threads = 1;
        int repetitions = 1;
        std::vector<int> perf_cpus;
        std::vector<int> pin_workload_cpus;
        bool pin_workers = false;
        int collector_cpu = -1;
        int consumer_cpu = -1;
        std::string cgroup_path;
        long interval_ns = 1'000'000;
        bool enable_perf = true;
        std::string rapl_pkg_path;
        std::string rapl_dram_path;
        fs::path output_dir = "runs";
        std::string run_id;
        fs::path workload_bin;
        // External binary mode: when exec_path is non-empty, the launcher
        // measures a real binary instead of the synthetic telemetry_kernel_workload.
        // The fork+exec target is replaced; the rest of the measurement pipeline
        // (cgroup, affinity, perf by PID/cgroup, RAPL, file export) is agnostic.
        fs::path exec_path;
        std::vector<std::string> exec_argv;
    };

    /** @brief Result captured from one child workload process. */
    struct ChildResult {
        uint64_t elapsed_ns = 0;
        int exit_code = -1;
        std::string output;
        pid_t pid = -1;
        // ARC-50: per-node capability, only meaningful when collect=true.
        // false for the baseline child (no collector) and for external-mode
        // runs where collection failed to start.
        bool stalled_cycles_backend_available = false;
        bool l2_lines_in_all_available = false; // ARC-63, same semantics as above
    };

    /** @brief Sample plus repetition id, used to avoid cross-run deltas. */
    struct RecordedSample {
        int repetition = 0;
        telemetry::Sample sample{};
    };

    /** @brief Aggregated RAPL export diagnostics for metadata/summary files. */
    struct RaplSummary {
        uint64_t pkg_max_range_uj = 0;
        uint64_t dram_max_range_uj = 0;
        uint64_t pkg_total_delta_uj = 0;
        uint64_t dram_total_delta_uj = 0;
        uint64_t energy_delta_count = 0;
    };

    [[noreturn]] void usage(const char* argv0) {
        std::fprintf(stderr,
                     "usage: %s (synthetic)  --kernel <name> --size <N> --iterations <N> "
                     "--warmup <N> --threads <N> --repetitions <N> "
                     "--perf-cpus <list> "
                     "[--pin-workload-cpus <list> --pin-workers] "
                     "--collector-cpu <cpu> --consumer-cpu <cpu> "
                     "--cgroup-path <path> --output-dir <dir> --run-id <id>\n"
                     "       %s (external)   --exec <path> [--exec-args <string>] "
                     "--repetitions <N> "
                     "--perf-cpus <list> "
                     "[--pin-workload-cpus <list>] "
                     "--collector-cpu <cpu> --consumer-cpu <cpu> "
                     "--cgroup-path <path> --output-dir <dir> --run-id <id>\n",
                     argv0,
                     argv0);
        std::exit(2);
    }

    fs::path default_workload_path(const char* argv0) {
        fs::path self = fs::absolute(argv0);
        return self.parent_path() / "telemetry_kernel_workload";
    }

    // Simple whitespace tokenizer for --exec-args. Quoting/escaping is not
    // supported on purpose: external dataset binaries are expected to expose
    // flag-style arguments that do not need shell parsing.
    std::vector<std::string> split_whitespace(const std::string& text) {
        std::vector<std::string> result;
        size_t i = 0;
        while(i < text.size()) {
            while(i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            if(i >= text.size()) break;
            const size_t begin = i;
            while(i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            result.push_back(text.substr(begin, i - begin));
        }
        return result;
    }

    Options parse_args(int argc, char** argv) {
        Options opt;
        opt.workload_bin = default_workload_path(argv[0]);
        opt.run_id = "run_" + std::to_string(telemetry::experiment::now_ns());

        // Keep parsing explicit instead of adding a dependency on an argument
        // library. This binary is part of the experimental harness and should
        // remain easy to build on restricted HPC nodes.
        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_value = [&]() -> const char* {
                if(i + 1 >= argc) usage(argv[0]);
                return argv[++i];
            };

            if(arg == "--kernel") {
                opt.kernel = need_value();
                opt.kernel_explicit = true;
            } else if(arg == "--size") {
                opt.size = static_cast<size_t>(std::stoull(need_value()));
            } else if(arg == "--iterations") {
                opt.iterations = std::stoi(need_value());
            } else if(arg == "--warmup") {
                opt.warmup = std::stoi(need_value());
            } else if(arg == "--threads") {
                opt.threads = std::stoi(need_value());
            } else if(arg == "--repetitions") {
                opt.repetitions = std::stoi(need_value());
            } else if(arg == "--perf-cpus") {
                opt.perf_cpus = telemetry::experiment::parse_cpu_list(need_value());
            } else if(arg == "--workload-cpus") {
                opt.perf_cpus = telemetry::experiment::parse_cpu_list(need_value());
            } else if(arg == "--pin-workload-cpus") {
                opt.pin_workload_cpus = telemetry::experiment::parse_cpu_list(need_value());
            } else if(arg == "--pin-workers") {
                opt.pin_workers = true;
            } else if(arg == "--collector-cpu") {
                opt.collector_cpu = std::stoi(need_value());
            } else if(arg == "--consumer-cpu") {
                opt.consumer_cpu = std::stoi(need_value());
            } else if(arg == "--cgroup-path") {
                opt.cgroup_path = need_value();
            } else if(arg == "--interval-ns") {
                opt.interval_ns = std::stol(need_value());
            } else if(arg == "--no-perf") {
                opt.enable_perf = false;
            } else if(arg == "--rapl-pkg") {
                opt.rapl_pkg_path = need_value();
            } else if(arg == "--rapl-dram") {
                opt.rapl_dram_path = need_value();
            } else if(arg == "--output-dir") {
                opt.output_dir = need_value();
            } else if(arg == "--run-id") {
                opt.run_id = need_value();
            } else if(arg == "--workload-bin") {
                opt.workload_bin = need_value();
            } else if(arg == "--exec") {
                opt.exec_path = need_value();
            } else if(arg == "--exec-args") {
                opt.exec_argv = split_whitespace(need_value());
            } else if(arg == "--help") {
                usage(argv[0]);
            } else {
                usage(argv[0]);
            }
        }

        // External binary mode: the synthetic kernel parameters do not apply.
        // The default kernel label becomes the exec basename so the dataset
        // metadata still identifies the run. All other knobs (cgroup, perf,
        // RAPL, affinity, output) keep their existing semantics.
        if(!opt.exec_path.empty()) {
            if(!opt.kernel_explicit) {
                opt.kernel = opt.exec_path.filename().string();
            }
            opt.size = 0;
            opt.iterations = 0;
            opt.warmup = 0;
            opt.threads = 0;
            // --pin-workers targets the synthetic workload's thread pool and
            // has no meaning for an external binary. It is silently ignored.
            opt.pin_workers = false;
        } else {
            if(!telemetry::experiment::is_supported_kernel(opt.kernel)) {
                throw std::invalid_argument("unsupported kernel: " + opt.kernel);
            }
            if(opt.size == 0) throw std::invalid_argument("--size must be positive");
            if(opt.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
            if(opt.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
            if(opt.threads <= 0) throw std::invalid_argument("--threads must be positive");
            if(opt.pin_workers && opt.pin_workload_cpus.empty()) {
                throw std::invalid_argument("--pin-workers requires --pin-workload-cpus");
            }
            if(opt.pin_workers && static_cast<size_t>(opt.threads) > opt.pin_workload_cpus.size()) {
                throw std::invalid_argument("--threads must not exceed --pin-workload-cpus count when --pin-workers is used");
            }
        }
        if(opt.repetitions <= 0) throw std::invalid_argument("--repetitions must be positive");
        if(opt.interval_ns <= 0) throw std::invalid_argument("--interval-ns must be positive");
        // --cgroup-path is optional (CPP-05): perf now attaches by PID with
        // inherit=1, never through a cgroup. When present it is only used to
        // move the measured child into a delegated cgroup as an additional
        // isolation mechanism.
        if(opt.enable_perf && opt.perf_cpus.empty()) {
            throw std::invalid_argument("--perf-cpus is required when perf is enabled");
        }
        return opt;
    }

    void set_affinity(pid_t pid, const std::vector<int>& cpus) {
        if(cpus.empty()) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        for(int cpu : cpus) CPU_SET(cpu, &set);
        if(::sched_setaffinity(pid, sizeof(set), &set) != 0) {
            throw std::runtime_error(std::string("sched_setaffinity failed: ") + std::strerror(errno));
        }
    }

    void set_current_thread_affinity(int cpu) {
        if(cpu < 0) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if(rc != 0) {
            throw std::runtime_error(std::string("pthread_setaffinity_np failed: ") + std::strerror(rc));
        }
    }

    void write_all(int fd, const char* data, size_t size) {
        size_t written = 0;
        while(written < size) {
            const ssize_t n = ::write(fd, data + written, size - written);
            if(n <= 0) throw std::runtime_error("write failed");
            written += static_cast<size_t>(n);
        }
    }

    void move_pid_to_cgroup(pid_t pid, const std::string& cgroup_path) {
        if(cgroup_path.empty()) return;
        const fs::path procs = fs::path(cgroup_path) / "cgroup.procs";
        // The launcher only joins a pre-created/delegated cgroup. It does not
        // create hierarchy state or require global cgroup privileges.
        std::ofstream out(procs);
        if(!out.is_open()) {
            throw std::runtime_error("failed to open cgroup.procs: " + procs.string());
        }
        out << pid;
        if(!out) {
            throw std::runtime_error("failed to move child into cgroup: " + procs.string());
        }
    }

    uint64_t parse_elapsed_ns(const std::string& output) {
        const std::string key = "elapsed_ns=";
        const size_t pos = output.find(key);
        if(pos == std::string::npos) {
            throw std::runtime_error("workload did not report elapsed_ns");
        }
        const size_t begin = pos + key.size();
        size_t end = begin;
        while(end < output.size() && output[end] >= '0' && output[end] <= '9') ++end;
        return std::stoull(output.substr(begin, end - begin));
    }

    uint64_t read_rapl_max_range_uj(const std::string& domain_path) {
        if(domain_path.empty()) return 0;

        std::ifstream in(fs::path(domain_path) / "max_energy_range_uj");
        std::string text;
        if(!(in >> text)) return 0;

        uint64_t parsed = 0;
        return telemetry::detail::parse_uint64(text.c_str(), parsed) ? parsed : 0;
    }

    telemetry::experiment::RaplExportConfig read_rapl_export_config(const Options& opt) {
        telemetry::experiment::RaplExportConfig config{};
        config.pkg_max_range_uj = read_rapl_max_range_uj(opt.rapl_pkg_path);
        config.dram_max_range_uj = read_rapl_max_range_uj(opt.rapl_dram_path);
        return config;
    }

    RaplSummary compute_rapl_summary(const Options& opt,
                                     const std::vector<RecordedSample>& samples) {
        const telemetry::experiment::RaplExportConfig config = read_rapl_export_config(opt);
        RaplSummary summary{};
        summary.pkg_max_range_uj = config.pkg_max_range_uj;
        summary.dram_max_range_uj = config.dram_max_range_uj;

        telemetry::experiment::RaplDeltaState state{};
        for(const auto& record : samples) {
            if(record.sample.tag != telemetry::SampleTag::ENERGY) continue;
            // Reuse the same export delta helper used by CSV generation so
            // metadata totals follow exactly the same wrap/validity rules.
            const telemetry::experiment::RaplDelta delta =
                telemetry::experiment::next_rapl_delta(
                    record.repetition,
                    record.sample.energy,
                    config,
                    state
                );
            if(!delta.valid) continue;

            summary.pkg_total_delta_uj += delta.pkg_delta_uj;
            summary.dram_total_delta_uj += delta.dram_delta_uj;
            ++summary.energy_delta_count;
        }
        return summary;
    }

    std::vector<std::string> build_workload_args(const Options& opt, int ready_fd, int go_fd) {
        if(!opt.exec_path.empty()) {
            // External binary mode: the harness only sets up the measured process.
            // The ready/go handshake does not apply because the external binary
            // does not know the protocol; elapsed time is taken from the parent.
            std::vector<std::string> args;
            args.reserve(1 + opt.exec_argv.size());
            args.push_back(opt.exec_path.string());
            for(const auto& a : opt.exec_argv) args.push_back(a);
            (void)ready_fd;
            (void)go_fd;
            return args;
        }
        std::vector<std::string> args = {
            opt.workload_bin.string(),
            "--kernel", opt.kernel,
            "--size", std::to_string(opt.size),
            "--iterations", std::to_string(opt.iterations),
            "--warmup", std::to_string(opt.warmup),
            "--threads", std::to_string(opt.threads),
            "--ready-fd", std::to_string(ready_fd),
            "--go-fd", std::to_string(go_fd)
        };
        if(opt.pin_workers && !opt.pin_workload_cpus.empty()) {
            // Worker pinning is delegated to the child so std::thread native
            // handles are available when affinity is applied.
            args.push_back("--worker-cpus");
            args.push_back(telemetry::experiment::format_cpu_list(opt.pin_workload_cpus));
        }
        return args;
    }

    void drain_samples(telemetry::Collector::Ring& ring,
                       std::atomic<bool>& stop,
                       std::vector<RecordedSample>& samples,
                       int consumer_cpu,
                       int repetition) {
        set_current_thread_affinity(consumer_cpu);
        while(!stop.load(std::memory_order_relaxed)) {
            while(auto sample = ring.try_pop()) {
                samples.push_back(RecordedSample{repetition, *sample});
            }
            ring.flush_consumer();

            // Batch drain and sleep briefly. Disk export is deferred until the
            // workload is finished so the measured window does not write files.
            struct timespec t{0, 100'000};
            nanosleep(&t, nullptr);
        }
        while(auto sample = ring.try_pop()) {
            samples.push_back(RecordedSample{repetition, *sample});
        }
        ring.flush_consumer();
    }

    ChildResult run_child(const Options& opt,
                          bool collect,
                          std::vector<RecordedSample>& samples,
                          uint64_t reserve_samples,
                          uint64_t& push_retries,
                          int repetition) {
        // The ready/go pipes delimit the measured region for the synthetic
        // workload on top of the SIGSTOP/SIGCONT handshake below: the child
        // reports ready after setup and warmup; the parent sends go once perf
        // is armed. External binaries do not speak this protocol, so the
        // pipes (and the wait/send steps in the parent) are skipped in --exec
        // mode; SIGCONT alone releases straight into execv().
        const bool use_ready_go = opt.exec_path.empty();
        int ready_pipe[2] = {-1, -1};
        int go_pipe[2] = {-1, -1};
        int stdout_pipe[2];
        if(::pipe(stdout_pipe) != 0) {
            throw std::runtime_error("pipe failed");
        }
        if(use_ready_go) {
            if(::pipe(ready_pipe) != 0 || ::pipe(go_pipe) != 0) {
                ::close(stdout_pipe[0]);
                ::close(stdout_pipe[1]);
                throw std::runtime_error("pipe failed");
            }
        }

        const pid_t pid = ::fork();
        if(pid < 0) throw std::runtime_error("fork failed");

        if(pid == 0) {
            // Stop immediately, before touching any fd or doing any setup, so
            // the parent can open perf on this exact PID (stop->open->resume,
            // Guia_Maestra_Fase1_DVFS.md section 3.1/4.2) before a single
            // instruction of the measured workload runs.
            ::raise(SIGSTOP);
            try {
                if(use_ready_go) {
                    ::close(ready_pipe[0]);
                    ::close(go_pipe[1]);
                }
                ::close(stdout_pipe[0]);
                if(::dup2(stdout_pipe[1], STDOUT_FILENO) < 0) _exit(126);
                set_affinity(0, opt.pin_workload_cpus);

                // Replace the child with the measured binary. For the synthetic
                // workload this is opt.workload_bin; for --exec mode it is the
                // external binary at opt.exec_path. The ready/go fds are unused
                // in external mode and ignored by build_workload_args.
                std::vector<std::string> args = build_workload_args(opt, ready_pipe[1], go_pipe[0]);
                std::vector<char*> argv;
                argv.reserve(args.size() + 1);
                for(auto& arg : args) argv.push_back(arg.data());
                argv.push_back(nullptr);
                const char* target = use_ready_go ? opt.workload_bin.c_str() : opt.exec_path.c_str();
                ::execv(target, argv.data());
            } catch(...) {
            }
            _exit(127);
        }

        if(use_ready_go) {
            ::close(ready_pipe[1]);
            ::close(go_pipe[0]);
        }
        ::close(stdout_pipe[1]);

        // Confirm the child actually reached the stopped state before doing
        // anything else with it. If it died before raise(SIGSTOP) could take
        // effect (e.g. fork-side resource exhaustion), fail fast instead of
        // opening perf on a PID that no longer means what we think it means.
        {
            int wstatus = 0;
            if(::waitpid(pid, &wstatus, WUNTRACED) < 0 || !WIFSTOPPED(wstatus)) {
                if(!WIFEXITED(wstatus) && !WIFSIGNALED(wstatus)) {
                    ::kill(pid, SIGKILL);
                    int ignored = 0;
                    ::waitpid(pid, &ignored, 0);
                }
                if(use_ready_go) { ::close(ready_pipe[0]); ::close(go_pipe[1]); }
                ::close(stdout_pipe[0]);
                throw std::runtime_error("child did not reach stopped state for perf setup");
            }
        }

        telemetry::Collector::Ring ring;
        telemetry::CollectorConfig cfg;
        cfg.enable_perf = opt.enable_perf;
        cfg.interval_ns = opt.interval_ns;
        cfg.producer_cpu = opt.collector_cpu;
        // perf_cgroup_path is intentionally left empty: measurement always
        // attaches by PID with inherit=1 (CPP-01/CPP-05). --cgroup-path, if
        // given, only isolates the child via move_pid_to_cgroup below.
        cfg.target_pid = pid;
        cfg.perf_cpus = opt.perf_cpus;
        cfg.rapl_pkg_path = opt.rapl_pkg_path;
        cfg.rapl_dram_path = opt.rapl_dram_path;
        telemetry::Collector collector(cfg, ring);

        std::atomic<bool> stop_consumer{false};
        std::thread consumer;
        // Wall-clock start for the measured window in --exec mode. Captured
        // immediately before the collector starts so fork/cgroup placement
        // stay outside the measured region; SIGCONT (and thus execv) only
        // happens after this point, mirroring the synthetic path.
        uint64_t wall_start_ns = 0;
        uint64_t wall_end_ns = 0;
        // ARC-50: must be read before collector.stop() -- stop() closes the
        // perf reader, after which has_stalled_cycles_backend() always
        // reports false regardless of what actually happened during the run.
        bool stalled_cycles_backend_available = false;
        bool l2_lines_in_all_available = false; // ARC-63, same must-read-before-stop() constraint

        try {
            // The child is still stopped here: cgroup placement and perf
            // setup both happen before it can run a single instruction.
            move_pid_to_cgroup(pid, opt.cgroup_path);

            if(collect) {
                // Reserve before the consumer starts so vector growth does not
                // happen while the telemetry child is executing.
                samples.reserve(samples.size() + static_cast<size_t>(reserve_samples));
                consumer = std::thread(drain_samples,
                                       std::ref(ring),
                                       std::ref(stop_consumer),
                                       std::ref(samples),
                                       opt.consumer_cpu,
                                       repetition);
                // Start collection (opens+arms perf on the stopped pid) before
                // releasing the measured workload.
                wall_start_ns = telemetry::experiment::now_ns();
                collector.start();
                stalled_cycles_backend_available = collector.has_stalled_cycles_backend();
                l2_lines_in_all_available = collector.has_l2_lines_in_all();
            }

            // Release the child unconditionally: it stopped itself right
            // after fork regardless of whether this repetition collects
            // telemetry (baseline runs still need to resume).
            if(::kill(pid, SIGCONT) != 0) {
                throw std::runtime_error(std::string("SIGCONT failed: ") + std::strerror(errno));
            }

            if(use_ready_go) {
                char ready = 0;
                if(::read(ready_pipe[0], &ready, 1) != 1 || ready != 'R') {
                    throw std::runtime_error("workload failed before ready signal");
                }
                const char go = 'G';
                write_all(go_pipe[1], &go, 1);
            }
        } catch(...) {
            ::kill(pid, SIGKILL);
            int ignored = 0;
            ::waitpid(pid, &ignored, 0);
            if(collect) {
                collector.stop();
                stop_consumer.store(true, std::memory_order_relaxed);
                if(consumer.joinable()) consumer.join();
            }
            if(use_ready_go) ::close(ready_pipe[0]);
            if(use_ready_go) ::close(go_pipe[1]);
            ::close(stdout_pipe[0]);
            throw;
        }

        std::string output;
        char buffer[4096];
        // Capture stdout completely. On success it contains elapsed_ns (synthetic)
        // or the external binary's own diagnostics; on failure it helps diagnose.
        while(true) {
            const ssize_t n = ::read(stdout_pipe[0], buffer, sizeof(buffer));
            if(n > 0) output.append(buffer, static_cast<size_t>(n));
            else break;
        }

        int status = 0;
        ::waitpid(pid, &status, 0);
        wall_end_ns = telemetry::experiment::now_ns();

        if(collect) {
            collector.stop();
            push_retries = collector.push_retries();
            stop_consumer.store(true, std::memory_order_relaxed);
            if(consumer.joinable()) consumer.join();
        }

        if(use_ready_go) ::close(ready_pipe[0]);
        if(use_ready_go) ::close(go_pipe[1]);
        ::close(stdout_pipe[0]);

        ChildResult result;
        result.output = output;
        result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        result.pid = pid;
        result.stalled_cycles_backend_available = stalled_cycles_backend_available;
        result.l2_lines_in_all_available = l2_lines_in_all_available;
        if(use_ready_go) {
            if(result.exit_code == 0) result.elapsed_ns = parse_elapsed_ns(output);
        } else if(collect && result.exit_code == 0) {
            // External binary: trust the parent's wall clock across the same
            // window the synthetic path covers with the child's own timer.
            result.elapsed_ns = wall_end_ns - wall_start_ns;
        }
        return result;
    }

    telemetry::experiment::Stats sampling_jitter(const std::vector<RecordedSample>& samples) {
        std::vector<double> intervals;
        int current_repetition = -1;
        uint64_t previous = 0;
        for(const auto& record : samples) {
            const auto& sample = record.sample;
            if(sample.tag != telemetry::SampleTag::CPU) continue;
            if(record.repetition != current_repetition) {
                // Do not compute an interval across independent telemetry runs.
                current_repetition = record.repetition;
                previous = 0;
            }
            if(previous != 0) {
                intervals.push_back(static_cast<double>(sample.cpu.timestamp_ns - previous));
            }
            previous = sample.cpu.timestamp_ns;
        }
        return telemetry::experiment::compute_stats(intervals);
    }

    double perf_running_ratio_min(const std::vector<RecordedSample>& samples) {
        double min_ratio = 0.0;
        bool saw_ratio = false;
        for(const auto& record : samples) {
            const auto& sample = record.sample;
            if(sample.tag == telemetry::SampleTag::CPU && sample.cpu.time_enabled_ns != 0) {
                const double ratio = static_cast<double>(sample.cpu.time_running_ns) /
                                     static_cast<double>(sample.cpu.time_enabled_ns);
                if(!saw_ratio || ratio < min_ratio) {
                    min_ratio = ratio;
                    saw_ratio = true;
                }
            }
        }
        return saw_ratio ? min_ratio : 0.0;
    }

    const char* tag_name(telemetry::SampleTag tag) {
        switch(tag) {
            case telemetry::SampleTag::CPU: return "CPU";
            case telemetry::SampleTag::ENERGY: return "ENERGY";
            case telemetry::SampleTag::GPU: return "GPU";
        }
        return "UNKNOWN";
    }

    // The dataset label is normally derived from the synthetic kernel mapping.
    // External binaries (or any kernel outside the synthetic set) fall back to
    // the kernel name itself so metadata still carries a meaningful identifier
    // instead of the generic "unknown" bucket.
    std::string dataset_label(const std::string& kernel) {
        const char* mapped = telemetry::experiment::kernel_label(kernel);
        if(mapped[0] != 'u' || mapped[1] != 'n') return std::string(mapped);
        return kernel;
    }

    void write_samples_csv(const fs::path& path,
                           const Options& opt,
                           const std::vector<RecordedSample>& samples,
                           bool stalled_cycles_backend_available,
                           bool l2_lines_in_all_available) {
        const telemetry::experiment::RaplExportConfig rapl_config = read_rapl_export_config(opt);
        telemetry::experiment::RaplDeltaState rapl_state{};

        // CSV is deliberately rectangular: every row has the same columns and
        // unused fields remain empty. This makes downstream ML ingestion simple.
        std::ofstream out(path);
        out << "run_id,repetition,kernel,label,timestamp_ns,tag,instructions,cycles,"
               "cache_references,cache_misses,stalled_cycles_backend,l2_lines_in_all,time_enabled_ns,time_running_ns,"
               "pkg_uj,dram_uj,pkg_delta_uj,dram_delta_uj,energy_delta_valid,"
               "gpu_power_mw,gpu_util_pct\n";
        const std::string label = dataset_label(opt.kernel);

        auto write_prefix = [&](const RecordedSample& record,
                                uint64_t timestamp,
                                const char* tag) {
            out << opt.run_id << ','
                << record.repetition << ','
                << opt.kernel << ','
                << label << ','
                << timestamp << ','
                << tag;
        };
        auto empty_field = [&]() { out << ','; };
        auto value_field = [&](auto value) { out << ',' << value; };

        for(const auto& record : samples) {
            const auto& sample = record.sample;
            if(sample.tag == telemetry::SampleTag::CPU) {
                write_prefix(record, sample.cpu.timestamp_ns, tag_name(sample.tag));
                value_field(sample.cpu.instructions);
                value_field(sample.cpu.cycles);
                value_field(sample.cpu.cache_references);
                value_field(sample.cpu.cache_misses);
                // ARC-50: empty (not "0"), a real 0 stalls reading is
                // indistinguishable from "not measured" otherwise --
                // postprocess.py relies on this to tell node-level
                // unavailability apart from a genuine zero delta.
                if(stalled_cycles_backend_available) {
                    value_field(sample.cpu.stalled_cycles_backend);
                } else {
                    empty_field();
                }
                // ARC-63: same empty-not-zero rule as stalled_cycles_backend.
                if(l2_lines_in_all_available) {
                    value_field(sample.cpu.l2_lines_in_all);
                } else {
                    empty_field();
                }
                value_field(sample.cpu.time_enabled_ns);
                value_field(sample.cpu.time_running_ns);
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                out << '\n';
            } else if(sample.tag == telemetry::SampleTag::ENERGY) {
                const telemetry::experiment::RaplDelta delta =
                    telemetry::experiment::next_rapl_delta(
                        record.repetition,
                        sample.energy,
                        rapl_config,
                        rapl_state
                    );
                // ENERGY rows keep raw cumulative counters and export derived
                // deltas with an explicit validity bit for wrap/first samples.
                write_prefix(record, sample.energy.timestamp_ns, tag_name(sample.tag));
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                value_field(sample.energy.pkg_uj);
                value_field(sample.energy.dram_uj);
                value_field(delta.pkg_delta_uj);
                value_field(delta.dram_delta_uj);
                value_field(delta.valid ? 1 : 0);
                empty_field();
                empty_field();
                out << '\n';
            } else {
                write_prefix(record, sample.gpu.timestamp_ns, tag_name(sample.tag));
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                empty_field();
                value_field(sample.gpu.power_mw);
                value_field(sample.gpu.util_pct);
                out << '\n';
            }
        }
    }

    template <typename T>
    void write_json_array(std::ofstream& out, const std::vector<T>& values) {
        out << '[';
        for(size_t i = 0; i < values.size(); ++i) {
            if(i != 0) out << ',';
            out << values[i];
        }
        out << ']';
    }

    void write_metadata_json(const fs::path& path,
                             const Options& opt,
                             const std::vector<uint64_t>& baseline_elapsed_ns,
                             const std::vector<uint64_t>& telemetry_elapsed_ns,
                             const std::vector<double>& overheads,
                             const std::vector<RecordedSample>& samples,
                             const std::vector<uint64_t>& push_retries_by_repetition,
                             const std::vector<pid_t>& measured_pids) {
        const auto jitter = sampling_jitter(samples);
        const double ratio = perf_running_ratio_min(samples);
        const RaplSummary rapl_summary = compute_rapl_summary(opt, samples);

        // Metadata contains both per-repetition raw values and aggregate
        // statistics so a dataset pipeline can choose its own analysis level.
        std::vector<double> baseline_values;
        std::vector<double> telemetry_values;
        baseline_values.reserve(baseline_elapsed_ns.size());
        telemetry_values.reserve(telemetry_elapsed_ns.size());
        for(uint64_t value : baseline_elapsed_ns) baseline_values.push_back(static_cast<double>(value));
        for(uint64_t value : telemetry_elapsed_ns) telemetry_values.push_back(static_cast<double>(value));
        const auto baseline_stats = telemetry::experiment::compute_stats(baseline_values);
        const auto telemetry_stats = telemetry::experiment::compute_stats(telemetry_values);
        const auto overhead_stats = telemetry::experiment::compute_stats(overheads);
        const uint64_t push_retries_total = std::accumulate(push_retries_by_repetition.begin(),
                                                            push_retries_by_repetition.end(),
                                                            uint64_t{0});

        std::ofstream out(path);
        out << "{\n";
        out << "  \"run_id\": \"" << telemetry::experiment::json_escape(opt.run_id) << "\",\n";
        out << "  \"kernel\": \"" << telemetry::experiment::json_escape(opt.kernel) << "\",\n";
        out << "  \"label\": \"" << dataset_label(opt.kernel) << "\",\n";
        out << "  \"size\": " << opt.size << ",\n";
        out << "  \"iterations\": " << opt.iterations << ",\n";
        out << "  \"warmup\": " << opt.warmup << ",\n";
        out << "  \"threads\": " << opt.threads << ",\n";
        out << "  \"repetitions\": " << opt.repetitions << ",\n";
        out << "  \"interval_ns\": " << opt.interval_ns << ",\n";
        out << "  \"enable_perf\": " << (opt.enable_perf ? "true" : "false") << ",\n";
        out << "  \"perf_attach_mode\": \"pid_inherit\",\n";
        out << "  \"measured_pids\": ";
        write_json_array(out, measured_pids);
        out << ",\n";
        out << "  \"perf_cpus\": \"" << telemetry::experiment::format_cpu_list(opt.perf_cpus) << "\",\n";
        out << "  \"pin_workload_cpus\": \"" << telemetry::experiment::format_cpu_list(opt.pin_workload_cpus) << "\",\n";
        out << "  \"pin_workers\": " << (opt.pin_workers ? "true" : "false") << ",\n";
        out << "  \"collector_cpu\": " << opt.collector_cpu << ",\n";
        out << "  \"consumer_cpu\": " << opt.consumer_cpu << ",\n";
        out << "  \"cgroup_path\": \"" << telemetry::experiment::json_escape(opt.cgroup_path) << "\",\n";
        out << "  \"baseline_elapsed_ns_mean\": " << baseline_stats.mean << ",\n";
        out << "  \"baseline_elapsed_ns_sd\": " << baseline_stats.sd << ",\n";
        out << "  \"telemetry_elapsed_ns_mean\": " << telemetry_stats.mean << ",\n";
        out << "  \"telemetry_elapsed_ns_sd\": " << telemetry_stats.sd << ",\n";
        out << "  \"overhead_pct_mean\": " << overhead_stats.mean << ",\n";
        out << "  \"overhead_pct_sd\": " << overhead_stats.sd << ",\n";
        out << "  \"baseline_elapsed_ns_values\": ";
        write_json_array(out, baseline_elapsed_ns);
        out << ",\n";
        out << "  \"telemetry_elapsed_ns_values\": ";
        write_json_array(out, telemetry_elapsed_ns);
        out << ",\n";
        out << "  \"overhead_pct_values\": ";
        write_json_array(out, overheads);
        out << ",\n";
        out << "  \"sampling_interval_mean_ns\": " << jitter.mean << ",\n";
        out << "  \"sampling_interval_sd_ns\": " << jitter.sd << ",\n";
        out << "  \"sampling_interval_cv_pct\": " << jitter.cv_pct << ",\n";
        out << "  \"push_retries\": " << push_retries_total << ",\n";
        out << "  \"push_retries_by_repetition\": ";
        write_json_array(out, push_retries_by_repetition);
        out << ",\n";
        out << "  \"perf_running_ratio_min\": " << ratio << ",\n";
        out << "  \"rapl_pkg_max_range_uj\": " << rapl_summary.pkg_max_range_uj << ",\n";
        out << "  \"rapl_dram_max_range_uj\": " << rapl_summary.dram_max_range_uj << ",\n";
        out << "  \"rapl_pkg_total_delta_uj\": " << rapl_summary.pkg_total_delta_uj << ",\n";
        out << "  \"rapl_dram_total_delta_uj\": " << rapl_summary.dram_total_delta_uj << ",\n";
        out << "  \"rapl_energy_delta_count\": " << rapl_summary.energy_delta_count << ",\n";
        out << "  \"samples_collected\": " << samples.size() << "\n";
        out << "}\n";
    }

    void write_summary(const fs::path& path,
                       const Options& opt,
                       const std::vector<uint64_t>& baseline_elapsed_ns,
                       const std::vector<uint64_t>& telemetry_elapsed_ns,
                       const std::vector<double>& overheads,
                       const std::vector<RecordedSample>& samples,
                       const std::vector<uint64_t>& push_retries_by_repetition) {
        const auto jitter = sampling_jitter(samples);
        const RaplSummary rapl_summary = compute_rapl_summary(opt, samples);
        std::vector<double> baseline_values;
        std::vector<double> telemetry_values;
        baseline_values.reserve(baseline_elapsed_ns.size());
        telemetry_values.reserve(telemetry_elapsed_ns.size());
        for(uint64_t value : baseline_elapsed_ns) baseline_values.push_back(static_cast<double>(value));
        for(uint64_t value : telemetry_elapsed_ns) telemetry_values.push_back(static_cast<double>(value));
        const auto baseline_stats = telemetry::experiment::compute_stats(baseline_values);
        const auto telemetry_stats = telemetry::experiment::compute_stats(telemetry_values);
        const auto overhead_stats = telemetry::experiment::compute_stats(overheads);
        const uint64_t push_retries_total = std::accumulate(push_retries_by_repetition.begin(),
                                                            push_retries_by_repetition.end(),
                                                            uint64_t{0});
        std::ofstream out(path);
        out << "run_id=" << opt.run_id << "\n";
        out << "kernel=" << opt.kernel << "\n";
        out << "label=" << dataset_label(opt.kernel) << "\n";
        out << "repetitions=" << opt.repetitions << "\n";
        out << "baseline_elapsed_ns_mean=" << baseline_stats.mean << "\n";
        out << "baseline_elapsed_ns_sd=" << baseline_stats.sd << "\n";
        out << "telemetry_elapsed_ns_mean=" << telemetry_stats.mean << "\n";
        out << "telemetry_elapsed_ns_sd=" << telemetry_stats.sd << "\n";
        out << "overhead_pct_mean=" << overhead_stats.mean << "\n";
        out << "overhead_pct_sd=" << overhead_stats.sd << "\n";
        out << "sampling_cv_pct=" << jitter.cv_pct << "\n";
        out << "perf_running_ratio_min=" << perf_running_ratio_min(samples) << "\n";
        out << "rapl_pkg_total_delta_uj=" << rapl_summary.pkg_total_delta_uj << "\n";
        out << "rapl_dram_total_delta_uj=" << rapl_summary.dram_total_delta_uj << "\n";
        out << "rapl_energy_delta_count=" << rapl_summary.energy_delta_count << "\n";
        out << "push_retries=" << push_retries_total << "\n";
        out << "samples_collected=" << samples.size() << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        // Store timing arrays separately from samples because baseline runs do
        // not collect telemetry rows. The repetition index links telemetry rows
        // to the corresponding telemetry elapsed value. External binary mode
        // leaves the baseline arrays empty: there is no uninstrumented twin
        // of a real binary to compare against.
        std::vector<uint64_t> baseline_elapsed_ns;
        std::vector<uint64_t> telemetry_elapsed_ns;
        std::vector<double> overheads;
        std::vector<uint64_t> push_retries_by_repetition;
        std::vector<RecordedSample> samples;
        std::vector<pid_t> measured_pids;
        const bool external_mode = !opt.exec_path.empty();
        // ARC-50: a per-node capability fact, expected identical across every
        // repetition of the same run on the same machine/kernel -- OR'd
        // across repetitions defensively rather than assumed from the first.
        bool stalled_cycles_backend_available = false;
        bool l2_lines_in_all_available = false; // ARC-63, same semantics as above

        telemetry_elapsed_ns.reserve(static_cast<size_t>(opt.repetitions));
        push_retries_by_repetition.reserve(static_cast<size_t>(opt.repetitions));
        if(!external_mode) {
            baseline_elapsed_ns.reserve(static_cast<size_t>(opt.repetitions));
            overheads.reserve(static_cast<size_t>(opt.repetitions));
        }

        for(int repetition = 1; repetition <= opt.repetitions; ++repetition) {
            if(external_mode) {
                // No baseline child: the real binary is its own reference.
                // The reserve is a coarse heuristic since the binary's runtime
                // is not known a priori; it only affects when the samples vector
                // reallocates, not correctness.
                const uint64_t expected_samples = 4096;
                uint64_t push_retries = 0;
                const ChildResult telemetry = run_child(opt,
                                                        true,
                                                        samples,
                                                        expected_samples,
                                                        push_retries,
                                                        repetition);
                if(telemetry.exit_code != 0) {
                    std::fprintf(stderr, "external workload failed: repetition=%d exit=%d\n%s",
                                 repetition,
                                 telemetry.exit_code,
                                 telemetry.output.c_str());
                    return 1;
                }
                // CAL-02/CAL-03/POST-09 parse BW_pico/P_pico/FLOPs from the
                // measured binary's OWN stdout (never a PMU counter). Before
                // this, run_child() captured it into ChildResult::output but
                // only ever printed it on the failure path above -- on
                // success it was silently discarded, so the orchestrator's
                // stdout.txt (which is this process's own stdout, captured
                // by runner.py) never actually contained the child's program
                // output to regex-match against. Found running the real
                // orchestrator against felix for the first time (F4.4).
                std::fputs(telemetry.output.c_str(), stdout);
                telemetry_elapsed_ns.push_back(telemetry.elapsed_ns);
                push_retries_by_repetition.push_back(push_retries);
                measured_pids.push_back(telemetry.pid);
                stalled_cycles_backend_available =
                    stalled_cycles_backend_available || telemetry.stalled_cycles_backend_available;
                l2_lines_in_all_available =
                    l2_lines_in_all_available || telemetry.l2_lines_in_all_available;
                continue;
            }

            std::vector<RecordedSample> discarded;
            uint64_t ignored_push_retries = 0;
            // Baseline and telemetry run sequentially, never concurrently.
            // This avoids mutual interference and makes overhead interpretation
            // simpler, at the cost of requiring repetitions to handle noise.
            const ChildResult baseline = run_child(opt,
                                                   false,
                                                   discarded,
                                                   0,
                                                   ignored_push_retries,
                                                   repetition);
            if(baseline.exit_code != 0) {
                std::fprintf(stderr, "baseline workload failed: repetition=%d exit=%d\n%s",
                             repetition,
                             baseline.exit_code,
                             baseline.output.c_str());
                return 1;
            }

            const uint64_t expected_samples =
                std::max<uint64_t>(
                    1024,
                    // A conservative reserve: CPU + RAPL + optional GPU rows
                    // can produce multiple samples per sampling tick.
                    baseline.elapsed_ns / static_cast<uint64_t>(opt.interval_ns) * 3 + 1024
                );
            uint64_t push_retries = 0;
            const ChildResult telemetry = run_child(opt,
                                                    true,
                                                    samples,
                                                    expected_samples,
                                                    push_retries,
                                                    repetition);
            if(telemetry.exit_code != 0) {
                std::fprintf(stderr, "telemetry workload failed: repetition=%d exit=%d\n%s",
                             repetition,
                             telemetry.exit_code,
                             telemetry.output.c_str());
                return 1;
            }
            // See the matching comment in the external-mode branch above.
            std::fputs(telemetry.output.c_str(), stdout);

            baseline_elapsed_ns.push_back(baseline.elapsed_ns);
            telemetry_elapsed_ns.push_back(telemetry.elapsed_ns);
            overheads.push_back(telemetry::experiment::overhead_percent(
                static_cast<double>(baseline.elapsed_ns),
                static_cast<double>(telemetry.elapsed_ns)
            ));
            push_retries_by_repetition.push_back(push_retries);
            measured_pids.push_back(telemetry.pid);
            stalled_cycles_backend_available =
                stalled_cycles_backend_available || telemetry.stalled_cycles_backend_available;
            l2_lines_in_all_available =
                l2_lines_in_all_available || telemetry.l2_lines_in_all_available;
        }

        const fs::path run_dir = opt.output_dir / opt.run_id;
        fs::create_directories(run_dir);
        write_samples_csv(run_dir / "samples.csv", opt, samples, stalled_cycles_backend_available, l2_lines_in_all_available);
        write_metadata_json(run_dir / "metadata.json",
                            opt,
                            baseline_elapsed_ns,
                            telemetry_elapsed_ns,
                            overheads,
                            samples,
                            push_retries_by_repetition,
                            measured_pids);
        write_summary(run_dir / "summary.txt",
                      opt,
                      baseline_elapsed_ns,
                      telemetry_elapsed_ns,
                      overheads,
                      samples,
                      push_retries_by_repetition);

        std::vector<double> baseline_values;
        std::vector<double> telemetry_values;
        baseline_values.reserve(baseline_elapsed_ns.size());
        telemetry_values.reserve(telemetry_elapsed_ns.size());
        for(uint64_t value : baseline_elapsed_ns) baseline_values.push_back(static_cast<double>(value));
        for(uint64_t value : telemetry_elapsed_ns) telemetry_values.push_back(static_cast<double>(value));
        const auto baseline_stats = telemetry::experiment::compute_stats(baseline_values);
        const auto telemetry_stats = telemetry::experiment::compute_stats(telemetry_values);
        const auto overhead_stats = telemetry::experiment::compute_stats(overheads);
        const auto jitter = sampling_jitter(samples);
        const uint64_t push_retries_total = std::accumulate(push_retries_by_repetition.begin(),
                                                            push_retries_by_repetition.end(),
                                                            uint64_t{0});
        std::printf("run_dir=%s\n", run_dir.c_str());
        std::printf("repetitions=%d baseline_mean_ns=%.2f telemetry_mean_ns=%.2f overhead_mean=%.2f%% overhead_sd=%.2f%%\n",
                    opt.repetitions,
                    baseline_stats.mean,
                    telemetry_stats.mean,
                    overhead_stats.mean,
                    overhead_stats.sd);
        std::printf("sampling_cv=%.2f%% perf_running_ratio_min=%.4f push_retries=%llu samples=%llu\n",
                    jitter.cv_pct,
                    perf_running_ratio_min(samples),
                    static_cast<unsigned long long>(push_retries_total),
                    static_cast<unsigned long long>(samples.size()));
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "telemetry_kernel_launcher: %s\n", e.what());
        return 1;
    }
}
