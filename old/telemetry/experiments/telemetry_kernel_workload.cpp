#include "telemetry/experiment_utils.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

/**
 * @file telemetry_kernel_workload.cpp
 * @brief CPU multithreaded kernels executed under launcher control.
 *
 * This binary should stay focused on workload execution. The launcher handles
 * cgroups, collector lifecycle, consumer lifecycle, and persistence. The child
 * prepares memory and worker threads before the measured window begins.
 */
namespace {
    // Prevent the compiler from removing whole kernels whose results are only
    // used to keep the workload observable.
    volatile double sink = 0.0;

    /** @brief Workload CLI/configuration parsed from launcher arguments. */
    struct Options {
        std::string kernel = "stream_triad";
        size_t size = 1'000'000;
        int iterations = 10;
        int warmup = 1;
        int threads = 1;
        std::vector<int> worker_cpus;
        int ready_fd = -1;
        int go_fd = -1;
    };

    [[noreturn]] void usage(const char* argv0) {
        std::fprintf(stderr,
                     "usage: %s --kernel <name> --size <N> --iterations <N> "
                     "--warmup <N> --threads <N> [--worker-cpus list] "
                     "[--ready-fd fd --go-fd fd]\n",
                     argv0);
        std::exit(2);
    }

    Options parse_args(int argc, char** argv) {
        Options opt;
        for(int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_value = [&](const char* name) -> const char* {
                if(i + 1 >= argc) usage(argv[0]);
                (void)name;
                return argv[++i];
            };

            if(arg == "--kernel") {
                opt.kernel = need_value("--kernel");
            } else if(arg == "--size") {
                opt.size = static_cast<size_t>(std::stoull(need_value("--size")));
            } else if(arg == "--iterations") {
                opt.iterations = std::stoi(need_value("--iterations"));
            } else if(arg == "--warmup") {
                opt.warmup = std::stoi(need_value("--warmup"));
            } else if(arg == "--threads") {
                opt.threads = std::stoi(need_value("--threads"));
            } else if(arg == "--worker-cpus") {
                opt.worker_cpus = telemetry::experiment::parse_cpu_list(need_value("--worker-cpus"));
            } else if(arg == "--ready-fd") {
                opt.ready_fd = std::stoi(need_value("--ready-fd"));
            } else if(arg == "--go-fd") {
                opt.go_fd = std::stoi(need_value("--go-fd"));
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
        if(!opt.worker_cpus.empty() && static_cast<size_t>(opt.threads) > opt.worker_cpus.size()) {
            throw std::invalid_argument("--threads must not exceed --worker-cpus count");
        }
        return opt;
    }

    /**
     * @brief Minimal fixed-size worker pool for CPU kernels.
     *
     * Workers are created before the measured region. Each run() call publishes
     * one Task plus an element count, and workers split the range by worker id.
     */
    class ThreadPool {
    public:
        /** @brief Per-kernel range task executed by each worker. */
        class Task {
        public:
            virtual ~Task() = default;
            virtual void run(size_t worker, size_t begin, size_t end) noexcept = 0;
        };

        ThreadPool(int threads, std::vector<int> worker_cpus)
            : worker_cpus_(std::move(worker_cpus)) {
            if(!worker_cpus_.empty() && worker_cpus_.size() < static_cast<size_t>(threads)) {
                throw std::invalid_argument("ThreadPool requires one worker CPU per thread");
            }
            workers_.reserve(static_cast<size_t>(threads));
            for(size_t worker = 0; worker < static_cast<size_t>(threads); ++worker) {
                spawn_worker(worker);
            }
        }

        ~ThreadPool() {
            stop_and_join();
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        size_t worker_count() const noexcept {
            return workers_.size();
        }

        void run(size_t count, Task& task) {
            if(count == 0) return;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                count_ = count;
                task_ = &task;
                remaining_ = workers_.size();
                ++generation_;
            }
            cv_.notify_all();

            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this]() { return remaining_ == 0; });
            task_ = nullptr;
        }

    private:
        std::vector<std::thread> workers_;
        std::vector<int> worker_cpus_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::condition_variable done_cv_;
        Task* task_ = nullptr;
        size_t count_ = 0;
        size_t remaining_ = 0;
        uint64_t generation_ = 0;
        bool stopping_ = false;

        void spawn_worker(size_t worker) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, worker);
            if(!worker_cpus_.empty()) {
                // Worker pinning is optional and controlled by the launcher.
                // It is applied after std::thread creation through native_handle.
                const int cpu = worker_cpus_[worker];
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(cpu, &set);
                const int rc = pthread_setaffinity_np(workers_.back().native_handle(), sizeof(set), &set);
                if(rc != 0) {
                    stop_and_join();
                    throw std::runtime_error(std::string("pthread_setaffinity_np worker failed: ") +
                                             std::strerror(rc));
                }
            }
        }

        void stop_and_join() noexcept {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
                ++generation_;
            }
            cv_.notify_all();
            for(auto& worker : workers_) {
                if(worker.joinable()) worker.join();
            }
        }

        void worker_loop(size_t worker) {
            uint64_t seen_generation = 0;
            while(true) {
                Task* task = nullptr;
                size_t count = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this, seen_generation]() {
                        return stopping_ || generation_ != seen_generation;
                    });
                    if(stopping_) return;
                    seen_generation = generation_;
                    task = task_;
                    count = count_;
                }

                const size_t workers = workers_.size();
                // Static contiguous partitioning keeps each task simple and
                // deterministic for benchmark comparisons.
                const size_t begin = worker * count / workers;
                const size_t end = (worker + 1) * count / workers;
                if(begin < end) {
                    task->run(worker, begin, end);
                }

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --remaining_;
                    if(remaining_ == 0) {
                        done_cv_.notify_one();
                    }
                }
            }
        }
    };

    /** @brief Common interface for all CPU kernels used by the workload. */
    class Kernel {
    public:
        virtual ~Kernel() = default;
        /** Reset mutable buffers after warmup and before the measured window. */
        virtual void prepare_for_measurement() = 0;
        /** Execute the measured kernel body for the requested iterations. */
        virtual void run(int iterations) = 0;
        /** Return one value so the compiler cannot discard the computation. */
        virtual double result() const noexcept = 0;
    };

    /** @brief STREAM triad style memory-bandwidth kernel. */
    class StreamTriadKernel final : public Kernel {
    public:
        StreamTriadKernel(size_t n, int threads, const std::vector<int>& worker_cpus)
            : n_(n), pool_(threads, worker_cpus), a_(n, 1.0), b_(n, 2.0), c_(n, 0.0) {}

        void prepare_for_measurement() override {
            std::fill(c_.begin(), c_.end(), 0.0);
        }

        void run(int iterations) override {
            struct TriadTask final : ThreadPool::Task {
                StreamTriadKernel* kernel;
                explicit TriadTask(StreamTriadKernel* k) : kernel(k) {}
                void run(size_t, size_t begin, size_t end) noexcept override {
                    constexpr double scalar = 3.0;
                    for(size_t i = begin; i < end; ++i) {
                        kernel->c_[i] = kernel->a_[i] + scalar * kernel->b_[i];
                    }
                }
            } task{this};

            for(int iter = 0; iter < iterations; ++iter) {
                pool_.run(n_, task);
            }
        }

        double result() const noexcept override {
            return c_[n_ / 2];
        }

    private:
        size_t n_;
        ThreadPool pool_;
        std::vector<double> a_;
        std::vector<double> b_;
        std::vector<double> c_;
    };

    /** @brief Parallel reduction kernel dominated by linear memory reads. */
    class ReductionKernel final : public Kernel {
    public:
        ReductionKernel(size_t n, int threads, const std::vector<int>& worker_cpus)
            : n_(n),
              pool_(threads, worker_cpus),
              values_(n, 1.0),
              partial_(pool_.worker_count(), 0.0) {}

        void prepare_for_measurement() override {
            std::fill(partial_.begin(), partial_.end(), 0.0);
        }

        void run(int iterations) override {
            struct ReductionTask final : ThreadPool::Task {
                ReductionKernel* kernel;
                explicit ReductionTask(ReductionKernel* k) : kernel(k) {}
                void run(size_t worker, size_t begin, size_t end) noexcept override {
                    double sum = 0.0;
                    for(size_t i = begin; i < end; ++i) {
                        sum += kernel->values_[i];
                    }
                    kernel->partial_[worker] = sum;
                }
            } task{this};

            for(int iter = 0; iter < iterations; ++iter) {
                std::fill(partial_.begin(), partial_.end(), 0.0);
                pool_.run(n_, task);
            }
        }

        double result() const noexcept override {
            double total = 0.0;
            for(double value : partial_) total += value;
            return total;
        }

    private:
        size_t n_;
        ThreadPool pool_;
        std::vector<double> values_;
        std::vector<double> partial_;
    };

    /** @brief 2D stencil kernel with local-neighbor cache-sensitive access. */
    class Stencil2DKernel final : public Kernel {
    public:
        Stencil2DKernel(size_t n, int threads, const std::vector<int>& worker_cpus)
            : n_(n),
              pool_(threads, worker_cpus),
              current_(n * n, 1.0),
              next_(n * n, 0.0) {
            if(n < 3) throw std::invalid_argument("stencil_2d requires --size >= 3");
        }

        void prepare_for_measurement() override {
            std::fill(current_.begin(), current_.end(), 1.0);
            std::fill(next_.begin(), next_.end(), 0.0);
        }

        void run(int iterations) override {
            struct StencilTask final : ThreadPool::Task {
                Stencil2DKernel* kernel;
                explicit StencilTask(Stencil2DKernel* k) : kernel(k) {}
                void run(size_t, size_t begin, size_t end) noexcept override {
                    const size_t n = kernel->n_;
                    for(size_t row_index = begin; row_index < end; ++row_index) {
                        const size_t r = row_index + 1;
                        for(size_t c = 1; c + 1 < n; ++c) {
                            const size_t idx = r * n + c;
                            kernel->next_[idx] = 0.25 * (
                                kernel->current_[idx - 1] +
                                kernel->current_[idx + 1] +
                                kernel->current_[idx - n] +
                                kernel->current_[idx + n]
                            );
                        }
                    }
                }
            } task{this};

            for(int iter = 0; iter < iterations; ++iter) {
                pool_.run(n_ - 2, task);
                current_.swap(next_);
            }
        }

        double result() const noexcept override {
            return current_[(n_ / 2) * n_ + (n_ / 2)];
        }

    private:
        size_t n_;
        ThreadPool pool_;
        std::vector<double> current_;
        std::vector<double> next_;
    };

    /** @brief Naive matrix multiplication kernel used as compute-heavy baseline. */
    class GemmNaiveKernel final : public Kernel {
    public:
        GemmNaiveKernel(size_t n, int threads, const std::vector<int>& worker_cpus)
            : n_(n),
              pool_(threads, worker_cpus),
              a_(n * n, 1.0),
              b_(n * n, 2.0),
              c_(n * n, 0.0) {}

        void prepare_for_measurement() override {
            std::fill(c_.begin(), c_.end(), 0.0);
        }

        void run(int iterations) override {
            struct GemmTask final : ThreadPool::Task {
                GemmNaiveKernel* kernel;
                explicit GemmTask(GemmNaiveKernel* k) : kernel(k) {}
                void run(size_t, size_t begin, size_t end) noexcept override {
                    const size_t n = kernel->n_;
                    for(size_t i = begin; i < end; ++i) {
                        for(size_t k = 0; k < n; ++k) {
                            const double aik = kernel->a_[i * n + k];
                            for(size_t j = 0; j < n; ++j) {
                                kernel->c_[i * n + j] += aik * kernel->b_[k * n + j];
                            }
                        }
                    }
                }
            } task{this};

            for(int iter = 0; iter < iterations; ++iter) {
                pool_.run(n_, task);
            }
        }

        double result() const noexcept override {
            return c_[(n_ / 2) * n_ + (n_ / 2)];
        }

    private:
        size_t n_;
        ThreadPool pool_;
        std::vector<double> a_;
        std::vector<double> b_;
        std::vector<double> c_;
    };

    std::unique_ptr<Kernel> make_kernel(const Options& opt) {
        if(opt.kernel == "stream_triad") {
            return std::make_unique<StreamTriadKernel>(opt.size, opt.threads, opt.worker_cpus);
        }
        if(opt.kernel == "reduction") {
            return std::make_unique<ReductionKernel>(opt.size, opt.threads, opt.worker_cpus);
        }
        if(opt.kernel == "stencil_2d") {
            return std::make_unique<Stencil2DKernel>(opt.size, opt.threads, opt.worker_cpus);
        }
        if(opt.kernel == "gemm_naive") {
            return std::make_unique<GemmNaiveKernel>(opt.size, opt.threads, opt.worker_cpus);
        }
        throw std::invalid_argument("unsupported kernel: " + opt.kernel);
    }

    void signal_ready_and_wait(const Options& opt) {
        if(opt.ready_fd >= 0) {
            // Signal after allocation, worker creation, and warmup. The parent
            // starts telemetry only after receiving this byte.
            const char ready = 'R';
            if(::write(opt.ready_fd, &ready, 1) != 1) {
                throw std::runtime_error("failed to signal readiness");
            }
        }
        if(opt.go_fd >= 0) {
            // The launcher sends go after collector/consumer setup. This keeps
            // setup overhead outside elapsed_ns.
            char go = 0;
            if(::read(opt.go_fd, &go, 1) != 1) {
                throw std::runtime_error("failed to wait for launch signal");
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        auto kernel = make_kernel(opt);
        if(opt.warmup > 0) kernel->run(opt.warmup);
        kernel->prepare_for_measurement();
        signal_ready_and_wait(opt);

        // Only the kernel loop is timed. Setup, warmup, and synchronization
        // with the launcher are intentionally outside this interval.
        const uint64_t start = telemetry::experiment::now_ns();
        kernel->run(opt.iterations);
        const uint64_t elapsed = telemetry::experiment::now_ns() - start;

        std::printf("elapsed_ns=%llu\n", static_cast<unsigned long long>(elapsed));
        sink = kernel->result();
        std::printf("sink=%.6f\n", static_cast<double>(sink));
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "telemetry_kernel_workload: %s\n", e.what());
        return 1;
    }
}
