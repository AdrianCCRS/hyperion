#include "telemetry/perf_reader.hpp"

#include <csignal>
#include <cstdio>
#include <ctime>
#include <sys/wait.h>
#include <unistd.h>

/**
 * @brief CPP-08: perf_event_open(pid, inherit=1) must produce live, growing
 * counts while the target process runs, not a single jump at exit.
 *
 * This is the exact failure mode documented in Guia_Maestra_Fase1_DVFS.md
 * section 3.2: opening on pid=0 before fork and relying on inherited folding
 * only updates the parent's fd when the child exits, leaving intermediate
 * reads flat. Opening directly on the child's pid (after it stops itself
 * with SIGSTOP) must not have that problem.
 */
int main() {
    pid_t child = ::fork();
    if(child < 0) return 1;

    if(child == 0) {
        ::raise(SIGSTOP);
        // Busy loop: enough work to guarantee several perf reads see nonzero
        // deltas without relying on a fixed sleep duration.
        volatile unsigned long long sink = 0;
        for(unsigned long long i = 0; i < 2'000'000'000ULL; ++i) {
            sink += i * i;
        }
        (void) sink;
        _exit(0);
    }

    int status = 0;
    if(::waitpid(child, &status, WUNTRACED) < 0 || !WIFSTOPPED(status)) {
        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
        return 1;
    }

    telemetry::PerfReader reader(child, -1);
    try {
        reader.open();
    } catch(const std::exception& e) {
        std::fprintf(stderr, "perf open failed (likely no PMU permission in this env): %s\n", e.what());
        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
        return 77; // treated as skip by callers that check for this code
    }

    if(::kill(child, SIGCONT) != 0) {
        reader.close();
        ::kill(child, SIGKILL);
        ::waitpid(child, &status, 0);
        return 1;
    }

    uint64_t previous_instructions = 0;
    int growing_deltas = 0;
    int flat_reads = 0;
    int worst_ratio_reads = 0;
    for(int sample = 0; sample < 20; ++sample) {
        struct timespec t{0, 20'000'000}; // 20ms
        nanosleep(&t, nullptr);

        int wstatus = 0;
        pid_t r = ::waitpid(child, &wstatus, WNOHANG);
        const bool child_alive = (r == 0);

        telemetry::CpuSample cpu{};
        if(!reader.read(cpu)) continue;

        if(cpu.instructions > previous_instructions) {
            if(previous_instructions > 0) ++growing_deltas;
            previous_instructions = cpu.instructions;
        } else if(previous_instructions > 0) {
            ++flat_reads;
        }

        // Regression check for the worst-ratio-across-events fix: the
        // exported time_enabled_ns/time_running_ns must describe SOME
        // opened counter's real state (running <= enabled, both nonzero
        // once the reader has produced any sample at all), never the
        // zeroed-out placeholder from before an event was ever read.
        if(cpu.time_enabled_ns > 0) {
            ++worst_ratio_reads;
            if(cpu.time_running_ns > cpu.time_enabled_ns) {
                std::fprintf(stderr, "time_running_ns (%llu) > time_enabled_ns (%llu)\n",
                             static_cast<unsigned long long>(cpu.time_running_ns),
                             static_cast<unsigned long long>(cpu.time_enabled_ns));
                reader.close();
                ::kill(child, SIGKILL);
                ::waitpid(child, &wstatus, 0);
                return 1;
            }
        }

        if(!child_alive) break;
    }

    if(worst_ratio_reads == 0) {
        std::fprintf(stderr, "expected at least one read with a populated time_enabled_ns\n");
        int cleanup_status = 0;
        reader.close();
        ::kill(child, SIGKILL);
        ::waitpid(child, &cleanup_status, 0);
        return 1;
    }

    reader.close();
    int wstatus = 0;
    ::waitpid(child, &wstatus, 0);

    // At least a couple of intermediate reads must show real, incremental
    // progress. A silent failure would show growing_deltas == 0 with all the
    // work landing in a single jump only visible after the child exits.
    if(growing_deltas < 2) {
        std::fprintf(stderr, "expected live incremental perf reads, got growing_deltas=%d flat_reads=%d\n",
                     growing_deltas, flat_reads);
        return 1;
    }

    return 0;
}
