#include "telemetry/collector.hpp"
#include "telemetry/experiment_utils.hpp"

#include <atomic>
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
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
    namespace fs = std::filesystem;

    struct Options {
        std::string kernel = "stream_triad";
        size_t size = 1'000'000;
        int iterations = 10;
        int warmup = 1;
        int threads = 1;
        std::vector<int> workload_cpus;
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
    };

    struct ChildResult {
        uint64_t elapsed_ns = 0;
        int exit_code = -1;
        std::string output;
    };

    [[noreturn]] void usage(const char* argv0) {
        std::fprintf(stderr,
                     "usage: %s --kernel <name> --size <N> --iterations <N> "
                     "--warmup <N> --threads <N> --workload-cpus <list> "
                     "--collector-cpu <cpu> --consumer-cpu <cpu> "
                     "--cgroup-path <path> --output-dir <dir> --run-id <id>\n",
                     argv0);
        std::exit(2);
    }

    fs::path default_workload_path(const char* argv0) {
        fs::path self = fs::absolute(argv0);
        return self.parent_path() / "telemetry_kernel_workload";
    }

    Options parse_args(int argc, char** argv) {
        Options opt;
        opt.workload_bin = default_workload_path(argv[0]);
        opt.run_id = "run_" + std::to_string(telemetry::experiment::now_ns());

        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_value = [&]() -> const char* {
                if(i + 1 >= argc) usage(argv[0]);
                return argv[++i];
            };

            if(arg == "--kernel") {
                opt.kernel = need_value();
            } else if(arg == "--size") {
                opt.size = static_cast<size_t>(std::stoull(need_value()));
            } else if(arg == "--iterations") {
                opt.iterations = std::stoi(need_value());
            } else if(arg == "--warmup") {
                opt.warmup = std::stoi(need_value());
            } else if(arg == "--threads") {
                opt.threads = std::stoi(need_value());
            } else if(arg == "--workload-cpus") {
                opt.workload_cpus = telemetry::experiment::parse_cpu_list(need_value());
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
            } else if(arg == "--help") {
                usage(argv[0]);
            } else {
                usage(argv[0]);
            }
        }

        if(!telemetry::experiment::is_supported_kernel(opt.kernel)) {
            throw std::invalid_argument("unsupported kernel: " + opt.kernel);
        }
        if(opt.size == 0) throw std::invalid_argument("--size must be positive");
        if(opt.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
        if(opt.warmup < 0) throw std::invalid_argument("--warmup must be non-negative");
        if(opt.threads <= 0) throw std::invalid_argument("--threads must be positive");
        if(opt.interval_ns <= 0) throw std::invalid_argument("--interval-ns must be positive");
        if(opt.enable_perf && opt.cgroup_path.empty()) {
            throw std::invalid_argument("--cgroup-path is required when perf is enabled");
        }
        if(opt.enable_perf && opt.workload_cpus.empty()) {
            throw std::invalid_argument("--workload-cpus is required when perf is enabled");
        }
        if(opt.collector_cpu >= 0 &&
           telemetry::experiment::contains_cpu(opt.workload_cpus, opt.collector_cpu)) {
            throw std::invalid_argument("--collector-cpu must not be inside --workload-cpus");
        }
        if(opt.consumer_cpu >= 0 &&
           telemetry::experiment::contains_cpu(opt.workload_cpus, opt.consumer_cpu)) {
            throw std::invalid_argument("--consumer-cpu must not be inside --workload-cpus");
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

    std::vector<std::string> build_workload_args(const Options& opt, int ready_fd, int go_fd) {
        return {
            opt.workload_bin.string(),
            "--kernel", opt.kernel,
            "--size", std::to_string(opt.size),
            "--iterations", std::to_string(opt.iterations),
            "--warmup", std::to_string(opt.warmup),
            "--threads", std::to_string(opt.threads),
            "--ready-fd", std::to_string(ready_fd),
            "--go-fd", std::to_string(go_fd)
        };
    }

    void drain_samples(telemetry::Collector::Ring& ring,
                       std::atomic<bool>& stop,
                       std::vector<telemetry::Sample>& samples,
                       int consumer_cpu) {
        set_current_thread_affinity(consumer_cpu);
        while(!stop.load(std::memory_order_relaxed)) {
            while(auto sample = ring.try_pop()) {
                samples.push_back(*sample);
            }
            ring.flush_consumer();
            struct timespec t{0, 100'000};
            nanosleep(&t, nullptr);
        }
        while(auto sample = ring.try_pop()) {
            samples.push_back(*sample);
        }
        ring.flush_consumer();
    }

    ChildResult run_child(const Options& opt,
                          bool collect,
                          std::vector<telemetry::Sample>& samples,
                          uint64_t reserve_samples,
                          uint64_t& push_retries) {
        int ready_pipe[2];
        int go_pipe[2];
        int stdout_pipe[2];
        if(::pipe(ready_pipe) != 0 || ::pipe(go_pipe) != 0 || ::pipe(stdout_pipe) != 0) {
            throw std::runtime_error("pipe failed");
        }

        const pid_t pid = ::fork();
        if(pid < 0) throw std::runtime_error("fork failed");

        if(pid == 0) {
            try {
                ::close(ready_pipe[0]);
                ::close(go_pipe[1]);
                ::close(stdout_pipe[0]);
                if(::dup2(stdout_pipe[1], STDOUT_FILENO) < 0) _exit(126);
                set_affinity(0, opt.workload_cpus);

                std::vector<std::string> args = build_workload_args(opt, ready_pipe[1], go_pipe[0]);
                std::vector<char*> argv;
                argv.reserve(args.size() + 1);
                for(auto& arg : args) argv.push_back(arg.data());
                argv.push_back(nullptr);
                ::execv(opt.workload_bin.c_str(), argv.data());
            } catch(...) {
            }
            _exit(127);
        }

        ::close(ready_pipe[1]);
        ::close(go_pipe[0]);
        ::close(stdout_pipe[1]);

        telemetry::Collector::Ring ring;
        telemetry::CollectorConfig cfg;
        cfg.enable_perf = opt.enable_perf;
        cfg.interval_ns = opt.interval_ns;
        cfg.producer_cpu = opt.collector_cpu;
        cfg.perf_cgroup_path = opt.cgroup_path;
        cfg.perf_cpus = opt.workload_cpus;
        cfg.rapl_pkg_path = opt.rapl_pkg_path;
        cfg.rapl_dram_path = opt.rapl_dram_path;
        telemetry::Collector collector(cfg, ring);

        std::atomic<bool> stop_consumer{false};
        std::thread consumer;

        try {
            move_pid_to_cgroup(pid, opt.cgroup_path);

            char ready = 0;
            if(::read(ready_pipe[0], &ready, 1) != 1 || ready != 'R') {
                throw std::runtime_error("workload failed before ready signal");
            }

            if(collect) {
                samples.reserve(static_cast<size_t>(reserve_samples));
                consumer = std::thread(drain_samples,
                                       std::ref(ring),
                                       std::ref(stop_consumer),
                                       std::ref(samples),
                                       opt.consumer_cpu);
                collector.start();
            }

            const char go = 'G';
            write_all(go_pipe[1], &go, 1);
        } catch(...) {
            ::kill(pid, SIGKILL);
            int ignored = 0;
            ::waitpid(pid, &ignored, 0);
            if(collect) {
                collector.stop();
                stop_consumer.store(true, std::memory_order_relaxed);
                if(consumer.joinable()) consumer.join();
            }
            ::close(ready_pipe[0]);
            ::close(go_pipe[1]);
            ::close(stdout_pipe[0]);
            throw;
        }

        std::string output;
        char buffer[4096];
        while(true) {
            const ssize_t n = ::read(stdout_pipe[0], buffer, sizeof(buffer));
            if(n > 0) output.append(buffer, static_cast<size_t>(n));
            else break;
        }

        int status = 0;
        ::waitpid(pid, &status, 0);

        if(collect) {
            collector.stop();
            push_retries = collector.push_retries();
            stop_consumer.store(true, std::memory_order_relaxed);
            if(consumer.joinable()) consumer.join();
        }

        ::close(ready_pipe[0]);
        ::close(go_pipe[1]);
        ::close(stdout_pipe[0]);

        ChildResult result;
        result.output = output;
        result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        if(result.exit_code == 0) result.elapsed_ns = parse_elapsed_ns(output);
        return result;
    }

    telemetry::experiment::Stats sampling_jitter(const std::vector<telemetry::Sample>& samples) {
        std::vector<double> intervals;
        uint64_t previous = 0;
        for(const auto& sample : samples) {
            if(sample.tag != telemetry::SampleTag::CPU) continue;
            if(previous != 0) {
                intervals.push_back(static_cast<double>(sample.cpu.timestamp_ns - previous));
            }
            previous = sample.cpu.timestamp_ns;
        }
        return telemetry::experiment::compute_stats(intervals);
    }

    double perf_running_ratio(const std::vector<telemetry::Sample>& samples) {
        for(auto it = samples.rbegin(); it != samples.rend(); ++it) {
            if(it->tag == telemetry::SampleTag::CPU && it->cpu.time_enabled_ns != 0) {
                return static_cast<double>(it->cpu.time_running_ns) /
                       static_cast<double>(it->cpu.time_enabled_ns);
            }
        }
        return 0.0;
    }

    const char* tag_name(telemetry::SampleTag tag) {
        switch(tag) {
            case telemetry::SampleTag::CPU: return "CPU";
            case telemetry::SampleTag::ENERGY: return "ENERGY";
            case telemetry::SampleTag::GPU: return "GPU";
        }
        return "UNKNOWN";
    }

    void write_samples_csv(const fs::path& path,
                           const Options& opt,
                           const std::vector<telemetry::Sample>& samples) {
        std::ofstream out(path);
        out << "run_id,kernel,label,timestamp_ns,tag,instructions,cycles,"
               "cache_references,cache_misses,time_enabled_ns,time_running_ns,"
               "pkg_uj,dram_uj,gpu_power_mw,gpu_util_pct\n";
        const char* label = telemetry::experiment::kernel_label(opt.kernel);
        for(const auto& sample : samples) {
            out << opt.run_id << ','
                << opt.kernel << ','
                << label << ',';
            if(sample.tag == telemetry::SampleTag::CPU) {
                out << sample.cpu.timestamp_ns << ','
                    << tag_name(sample.tag) << ','
                    << sample.cpu.instructions << ','
                    << sample.cpu.cycles << ','
                    << sample.cpu.cache_references << ','
                    << sample.cpu.cache_misses << ','
                    << sample.cpu.time_enabled_ns << ','
                    << sample.cpu.time_running_ns << ",,,,,\n";
            } else if(sample.tag == telemetry::SampleTag::ENERGY) {
                out << sample.energy.timestamp_ns << ','
                    << tag_name(sample.tag) << ",,,,,,,"
                    << sample.energy.pkg_uj << ','
                    << sample.energy.dram_uj << ",,\n";
            } else {
                out << sample.gpu.timestamp_ns << ','
                    << tag_name(sample.tag) << ",,,,,,,,,"
                    << sample.gpu.power_mw << ','
                    << sample.gpu.util_pct << '\n';
            }
        }
    }

    void write_metadata_json(const fs::path& path,
                             const Options& opt,
                             const ChildResult& baseline,
                             const ChildResult& telemetry,
                             const std::vector<telemetry::Sample>& samples,
                             uint64_t push_retries) {
        const auto jitter = sampling_jitter(samples);
        const double ratio = perf_running_ratio(samples);
        const double overhead = telemetry::experiment::overhead_percent(
            static_cast<double>(baseline.elapsed_ns),
            static_cast<double>(telemetry.elapsed_ns)
        );

        std::ofstream out(path);
        out << "{\n";
        out << "  \"run_id\": \"" << telemetry::experiment::json_escape(opt.run_id) << "\",\n";
        out << "  \"kernel\": \"" << telemetry::experiment::json_escape(opt.kernel) << "\",\n";
        out << "  \"label\": \"" << telemetry::experiment::kernel_label(opt.kernel) << "\",\n";
        out << "  \"size\": " << opt.size << ",\n";
        out << "  \"iterations\": " << opt.iterations << ",\n";
        out << "  \"warmup\": " << opt.warmup << ",\n";
        out << "  \"threads\": " << opt.threads << ",\n";
        out << "  \"interval_ns\": " << opt.interval_ns << ",\n";
        out << "  \"enable_perf\": " << (opt.enable_perf ? "true" : "false") << ",\n";
        out << "  \"workload_cpus\": \"" << telemetry::experiment::format_cpu_list(opt.workload_cpus) << "\",\n";
        out << "  \"collector_cpu\": " << opt.collector_cpu << ",\n";
        out << "  \"consumer_cpu\": " << opt.consumer_cpu << ",\n";
        out << "  \"cgroup_path\": \"" << telemetry::experiment::json_escape(opt.cgroup_path) << "\",\n";
        out << "  \"baseline_elapsed_ns\": " << baseline.elapsed_ns << ",\n";
        out << "  \"telemetry_elapsed_ns\": " << telemetry.elapsed_ns << ",\n";
        out << "  \"overhead_pct\": " << overhead << ",\n";
        out << "  \"sampling_interval_mean_ns\": " << jitter.mean << ",\n";
        out << "  \"sampling_interval_sd_ns\": " << jitter.sd << ",\n";
        out << "  \"sampling_interval_cv_pct\": " << jitter.cv_pct << ",\n";
        out << "  \"push_retries\": " << push_retries << ",\n";
        out << "  \"perf_running_ratio\": " << ratio << ",\n";
        out << "  \"samples_collected\": " << samples.size() << "\n";
        out << "}\n";
    }

    void write_summary(const fs::path& path,
                       const Options& opt,
                       const ChildResult& baseline,
                       const ChildResult& telemetry,
                       const std::vector<telemetry::Sample>& samples,
                       uint64_t push_retries) {
        const auto jitter = sampling_jitter(samples);
        const double overhead = telemetry::experiment::overhead_percent(
            static_cast<double>(baseline.elapsed_ns),
            static_cast<double>(telemetry.elapsed_ns)
        );
        std::ofstream out(path);
        out << "run_id=" << opt.run_id << "\n";
        out << "kernel=" << opt.kernel << "\n";
        out << "label=" << telemetry::experiment::kernel_label(opt.kernel) << "\n";
        out << "baseline_elapsed_ns=" << baseline.elapsed_ns << "\n";
        out << "telemetry_elapsed_ns=" << telemetry.elapsed_ns << "\n";
        out << "overhead_pct=" << overhead << "\n";
        out << "sampling_cv_pct=" << jitter.cv_pct << "\n";
        out << "perf_running_ratio=" << perf_running_ratio(samples) << "\n";
        out << "push_retries=" << push_retries << "\n";
        out << "samples_collected=" << samples.size() << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);

        std::vector<telemetry::Sample> discarded;
        uint64_t ignored_push_retries = 0;
        const ChildResult baseline = run_child(opt, false, discarded, 0, ignored_push_retries);
        if(baseline.exit_code != 0) {
            std::fprintf(stderr, "baseline workload failed: exit=%d\n%s",
                         baseline.exit_code,
                         baseline.output.c_str());
            return 1;
        }

        const uint64_t expected_samples =
            std::max<uint64_t>(1024, baseline.elapsed_ns / static_cast<uint64_t>(opt.interval_ns) * 3 + 1024);
        std::vector<telemetry::Sample> samples;
        uint64_t push_retries = 0;
        const ChildResult telemetry = run_child(opt, true, samples, expected_samples, push_retries);
        if(telemetry.exit_code != 0) {
            std::fprintf(stderr, "telemetry workload failed: exit=%d\n%s",
                         telemetry.exit_code,
                         telemetry.output.c_str());
            return 1;
        }

        const fs::path run_dir = opt.output_dir / opt.run_id;
        fs::create_directories(run_dir);
        write_samples_csv(run_dir / "samples.csv", opt, samples);
        write_metadata_json(run_dir / "metadata.json", opt, baseline, telemetry, samples, push_retries);
        write_summary(run_dir / "summary.txt", opt, baseline, telemetry, samples, push_retries);

        const double overhead = telemetry::experiment::overhead_percent(
            static_cast<double>(baseline.elapsed_ns),
            static_cast<double>(telemetry.elapsed_ns)
        );
        const auto jitter = sampling_jitter(samples);
        std::printf("run_dir=%s\n", run_dir.c_str());
        std::printf("baseline_elapsed_ns=%llu telemetry_elapsed_ns=%llu overhead=%.2f%%\n",
                    static_cast<unsigned long long>(baseline.elapsed_ns),
                    static_cast<unsigned long long>(telemetry.elapsed_ns),
                    overhead);
        std::printf("sampling_cv=%.2f%% perf_running_ratio=%.4f push_retries=%llu samples=%llu\n",
                    jitter.cv_pct,
                    perf_running_ratio(samples),
                    static_cast<unsigned long long>(push_retries),
                    static_cast<unsigned long long>(samples.size()));
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "telemetry_kernel_launcher: %s\n", e.what());
        return 1;
    }
}
