#pragma once
#include "metrics.hpp"
#include <cstdint>
#include <deque>
#include <string>
#include <sys/types.h>
#include <vector>

namespace telemetry {
    namespace detail {
        /**
         * @brief Parse one sysfs uncore event-format string into a raw config.
         *
         * Uncore PMU event definitions are exposed by the kernel itself under
         * /sys/bus/event_source/devices/<pmu>/events/<name> as strings like
         * "event=0x04,umask=0x03" (optionally with cmask=/edge=/inv=). Still
         * used here purely to validate the string is well-formed before
         * trusting it inside a shell-adjacent argv (see
         * build_perf_stat_event_list) -- the numeric config it produces is
         * no longer opened via perf_event_open directly (see class comment
         * below for why).
         */
        bool parse_uncore_event_format(const std::string& text, uint64_t& config) noexcept;

        /** @brief One uncore PMU box and its two raw CAS_COUNT event-format strings. */
        struct UncoreBoxEvents {
            std::string pmu_name;   // e.g. "uncore_imc_3"
            std::string read_format;   // e.g. "event=0x04,umask=0x0f"
            std::string write_format;  // e.g. "event=0x04,umask=0x30"
        };

        /**
         * @brief Build the comma-joined `-e` argument for `perf stat`.
         *
         * Produces "<pmu>/<read_format>/,<pmu>/<write_format>/,..." in the
         * exact order every box was given, read term before write term --
         * UncoreReader::read() relies on this fixed order (not on parsing
         * perf's own echoed event name back) to know which CSV line is a
         * read count and which is a write count.
         */
        std::string build_perf_stat_event_list(const std::vector<UncoreBoxEvents>& boxes);

        /** @brief One parsed field-set from one `perf stat -x<sep> -I` output line. */
        struct PerfStatCsvLine {
            double interval_time_s = 0.0;
            uint64_t value = 0;
            bool valid = false;  // false for "<not counted>"/"<not supported>"/malformed lines
        };

        /**
         * @brief Parse one CSV line from `perf stat -I <ms> -x<sep> --field-separator`.
         *
         * Field layout: interval-time,value,unit,event,time-running,percent-
         * running[,...]. A semicolon separator (not comma) is used on
         * purpose: perf's own raw event syntax ("event=0x04,umask=0x0f")
         * contains commas, which would collide with a comma field
         * separator -- see Registro_Cambios ARC-118 for why this bit us
         * before it was caught.
         */
        PerfStatCsvLine parse_perf_stat_csv_line(const std::string& line, char sep) noexcept;
    }

    /**
     * @brief Reader for Intel uncore memory-controller (iMC) CAS_COUNT_READ/
     * CAS_COUNT_WRITE counters, sourced from the `perf` CLI binary.
     *
     * ARC-118: uncore events are socket/system-scope (pid=-1), which
     * requires CAP_PERFMON (or perf_event_paranoid<1). The cluster admin
     * granted CAP_PERFMON specifically as a file capability on the `perf`
     * binary (`setcap cap_perfmon+ep /usr/bin/perf`), NOT as an ambient/
     * effective capability inheritable by an arbitrary process calling
     * perf_event_open() directly -- confirmed empirically on pacca (ARC-117):
     * a process opening the syscall itself always sees CapEff=0 and EACCES,
     * while the file capability only takes effect for whatever actually
     * execve()s that specific binary. So this reader shells out to `perf
     * stat -a -I <ms>` as a child process and parses its periodic CSV
     * output, instead of opening uncore_imc PMUs via perf_event_open like
     * the original ARC-116 design did (superseded, never worked on this
     * cluster's permission model).
     *
     * This trades some measurement precision for being the only channel
     * that can actually reach uncore_imc here: timestamps come from perf's
     * own interval clock (not the producer thread's CLOCK_MONOTONIC tick),
     * `-I`'s practical minimum interval is coarser than the 1ms per-PID
     * sampling cadence, and there is a short startup skew while the child
     * process spins up (same category of gap PerfReader's SIGSTOP/SIGCONT
     * dance exists specifically to avoid, but not reproducible for a
     * separate OS process). Documented explicitly, not silently accepted.
     *
     * ARC-119: each UncoreSnapshot this class produces is ALREADY a
     * per-interval delta (`perf stat -I` semantics), never a cumulative
     * counter -- see the UncoreSnapshot doc comment in metrics.hpp. The
     * consumer (postprocess.py) must not difference two readings against
     * each other, and must account for `-I`'s interval being coarser than
     * the 1ms per-PID CPU sampling cadence before attributing bytes to a
     * single CPU window.
     */
    class UncoreReader {
    public:
        /**
         * @param interval_ms Requested `perf stat -I` interval, clamped to a
         * conservative floor (see .cpp) since very small intervals are not
         * reliably supported by every perf build.
         * @param pin_cpu ARC-131: logical CPU to pin the `perf stat` child
         * to via sched_setaffinity(), or -1 (default) for no pinning. The
         * child otherwise inherits the launcher process's affinity
         * unrestricted -- found empirically on paccaA100 (smoke test,
         * ARC-130) to let the kernel schedule it onto the SAME delegated
         * CPUs the workload's own perf_event_open counters run on, which
         * measurably starved FP_ARITH_INST_RETIRED (a 4-way raw counter
         * group already at pmc_count's budget) via scheduling contention,
         * not PMC hardware contention (uncore_imc is physically separate
         * hardware). Pinning the child to an idle CPU outside
         * delegated_cpus/collector_cpu/consumer_cpu removes that overlap.
         */
        explicit UncoreReader(long interval_ms = 100, int pin_cpu = -1);
        ~UncoreReader();

        /**
         * @brief Discover uncore_imc boxes, launch `perf stat` as a child
         * process, and confirm it is still alive a moment later. Never
         * throws: any failure (perf missing, still EACCES, malformed sysfs
         * event definitions) degrades to is_open()==false, same contract as
         * the ARC-116 design and every other optional backend in this
         * codebase.
         */
        void open() noexcept;

        /** @brief Terminate the perf child (if running) and close the pipe. */
        void close() noexcept;

        /** @return true while the perf child process is confirmed alive. */
        bool is_open() const noexcept { return child_pid_ > 0; }

        /** @return number of uncore_imc boxes summed into each interval. */
        size_t box_count() const noexcept { return box_count_; }

        /**
         * @brief Drain whatever perf output is available and, if a full
         * interval's worth of CSV lines has arrived, emit it as one
         * snapshot. Non-blocking: returns false (not an error) when no
         * complete interval is ready yet on this call, exactly like a
         * cadence-gated reader (see Collector's GPU handling).
         */
        bool read(UncoreSnapshot& out) noexcept;

    private:
        long interval_ms_;
        int pin_cpu_;
        pid_t child_pid_ = -1;
        int pipe_fd_ = -1;
        size_t box_count_ = 0;

        // ARC-124: our CLOCK_MONOTONIC reading taken once, right when perf
        // is confirmed alive in open() -- every snapshot's timestamp is
        // this anchor plus perf's OWN relative interval_time_s (already
        // parsed from its output), never a fresh clock_gettime() at fold
        // time. Fixes a real artifact found under stress testing: if the
        // pipe accumulates a backlog (e.g. perf ran for a while during
        // open()'s liveness wait before the producer thread's first
        // read()), draining it folds many buckets in one tight loop, and
        // clock_gettime() called once per bucket in that loop clusters
        // their timestamps microseconds apart -- even though the
        // underlying data represents genuinely separate ~10ms intervals.
        // Anchoring to perf's own even cadence instead removes that
        // clustering entirely.
        long long launch_time_ns_ = 0;

        // Fixed request order (read term before write term, per box, same
        // order passed to build_perf_stat_event_list) -- read() indexes into
        // this by position within the current interval bucket instead of
        // parsing perf's own echoed event-name field back, which avoids any
        // risk of a formatting mismatch between what we asked for and what
        // perf prints.
        std::vector<bool> term_is_write_;

        std::string line_buffer_;      // partial trailing line across read() calls
        double bucket_time_s_ = -1.0;  // interval_time_s of the bucket being accumulated
        size_t bucket_seen_ = 0;       // lines folded into the current bucket so far
        uint64_t bucket_read_interval_ = 0;
        uint64_t bucket_write_interval_ = 0;
        bool bucket_has_data_ = false;
        bool bucket_any_valid_ = false;  // ARC-120: true once at least one term in this bucket parsed as valid

        // Completed-but-not-yet-returned snapshots. Normally at most one
        // deep (perf's -I interval is coarser than the producer thread's
        // poll cadence), but bounds correctness instead of silently
        // dropping a bucket if the producer thread is briefly descheduled
        // and perf emits more than one interval between two read() calls.
        std::deque<UncoreSnapshot> pending_;

        /** @brief Fold one already-parsed CSV line into the current bucket, pushing a snapshot into pending_ when a new interval starts. */
        void fold_line(const std::string& line);
    };
}
