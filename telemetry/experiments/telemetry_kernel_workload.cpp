#include "telemetry/experiment_utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
    volatile double sink = 0.0;

    struct Options {
        std::string kernel = "stream_triad";
        size_t size = 1'000'000;
        int iterations = 10;
        int warmup = 1;
        int threads = 1;
        int ready_fd = -1;
        int go_fd = -1;
    };

    [[noreturn]] void usage(const char* argv0) {
        std::fprintf(stderr,
                     "usage: %s --kernel <name> --size <N> --iterations <N> "
                     "--warmup <N> --threads <N> [--ready-fd fd --go-fd fd]\n",
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
        return opt;
    }

    template <typename Fn>
    void parallel_for(size_t count, int threads, Fn fn) {
        const size_t workers = std::min(static_cast<size_t>(threads), count);
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for(size_t worker = 0; worker < workers; ++worker) {
            const size_t begin = worker * count / workers;
            const size_t end = (worker + 1) * count / workers;
            pool.emplace_back([begin, end, &fn]() {
                for(size_t i = begin; i < end; ++i) fn(i);
            });
        }
        for(auto& thread : pool) thread.join();
    }

    void run_stream_triad(size_t n, int iterations, int threads) {
        std::vector<double> a(n, 1.0), b(n, 2.0), c(n, 0.0);
        constexpr double scalar = 3.0;
        for(int iter = 0; iter < iterations; ++iter) {
            parallel_for(n, threads, [&](size_t i) {
                c[i] = a[i] + scalar * b[i];
            });
        }
        sink = c[n / 2];
    }

    void run_reduction(size_t n, int iterations, int threads) {
        std::vector<double> values(n, 1.0);
        std::vector<double> partial(static_cast<size_t>(threads), 0.0);
        for(int iter = 0; iter < iterations; ++iter) {
            std::fill(partial.begin(), partial.end(), 0.0);
            const size_t workers = std::min(static_cast<size_t>(threads), n);
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for(size_t worker = 0; worker < workers; ++worker) {
                const size_t begin = worker * n / workers;
                const size_t end = (worker + 1) * n / workers;
                pool.emplace_back([begin, end, worker, &values, &partial]() {
                    double sum = 0.0;
                    for(size_t i = begin; i < end; ++i) sum += values[i];
                    partial[worker] = sum;
                });
            }
            for(auto& thread : pool) thread.join();
        }
        double total = 0.0;
        for(double value : partial) total += value;
        sink = total;
    }

    void run_stencil_2d(size_t n, int iterations, int threads) {
        if(n < 3) throw std::invalid_argument("stencil_2d requires --size >= 3");
        std::vector<double> current(n * n, 1.0);
        std::vector<double> next(n * n, 0.0);
        const size_t interior_rows = n - 2;
        for(int iter = 0; iter < iterations; ++iter) {
            parallel_for(interior_rows, threads, [&](size_t row_index) {
                const size_t r = row_index + 1;
                for(size_t c = 1; c + 1 < n; ++c) {
                    const size_t idx = r * n + c;
                    next[idx] = 0.25 * (current[idx - 1] + current[idx + 1] +
                                        current[idx - n] + current[idx + n]);
                }
            });
            current.swap(next);
        }
        sink = current[(n / 2) * n + (n / 2)];
    }

    void run_gemm_naive(size_t n, int iterations, int threads) {
        std::vector<double> a(n * n, 1.0);
        std::vector<double> b(n * n, 2.0);
        std::vector<double> c(n * n, 0.0);
        for(int iter = 0; iter < iterations; ++iter) {
            parallel_for(n, threads, [&](size_t i) {
                for(size_t k = 0; k < n; ++k) {
                    const double aik = a[i * n + k];
                    for(size_t j = 0; j < n; ++j) {
                        c[i * n + j] += aik * b[k * n + j];
                    }
                }
            });
        }
        sink = c[(n / 2) * n + (n / 2)];
    }

    void run_kernel(const Options& opt, int iterations) {
        if(opt.kernel == "stream_triad") {
            run_stream_triad(opt.size, iterations, opt.threads);
        } else if(opt.kernel == "reduction") {
            run_reduction(opt.size, iterations, opt.threads);
        } else if(opt.kernel == "stencil_2d") {
            run_stencil_2d(opt.size, iterations, opt.threads);
        } else if(opt.kernel == "gemm_naive") {
            run_gemm_naive(opt.size, iterations, opt.threads);
        } else {
            throw std::invalid_argument("unsupported kernel: " + opt.kernel);
        }
    }

    void signal_ready_and_wait(const Options& opt) {
        if(opt.ready_fd >= 0) {
            const char ready = 'R';
            if(::write(opt.ready_fd, &ready, 1) != 1) {
                throw std::runtime_error("failed to signal readiness");
            }
        }
        if(opt.go_fd >= 0) {
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
        if(opt.warmup > 0) run_kernel(opt, opt.warmup);
        signal_ready_and_wait(opt);

        const uint64_t start = telemetry::experiment::now_ns();
        run_kernel(opt, opt.iterations);
        const uint64_t elapsed = telemetry::experiment::now_ns() - start;

        std::printf("elapsed_ns=%llu\n", static_cast<unsigned long long>(elapsed));
        std::printf("sink=%.6f\n", static_cast<double>(sink));
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr, "telemetry_kernel_workload: %s\n", e.what());
        return 1;
    }
}
