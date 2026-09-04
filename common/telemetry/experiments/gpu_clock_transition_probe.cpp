#include "telemetry/gpu_transition_analysis.hpp"
#include "telemetry/experiment_utils.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifdef TELEMETRY_WITH_GPU
#include <nvml.h>
#endif

/**
 * @file gpu_clock_transition_probe.cpp
 * @brief F1-GPU-002 -- reproducible probe for NVML effective cadence and
 *        T_transicion_gpu under real sustained GPU load on the A100.
 *
 * Design (Seguimiento_Cambios_Plan_Director.md, F1-GPU-002; plan §2.4.1):
 *  - Independent executable. Does NOT touch collector.hpp cadence, the GPU
 *    trainer, campaign manifests, or derive_policy_table.py logic.
 *  - Clock sampling is ALWAYS nvmlDeviceGetClockInfo, never repeated
 *    `nvidia-smi` subprocesses.
 *  - `nvidia-smi` is used ONLY for actuation (`-lgc` / `-rgc`), reusing the
 *    exact command form and sudo/restore contract of
 *    common/hpc/gpu_freqctl.py (_default_run_nvidia_smi / restore_gpu_state).
 *    If that contract changes there, mirror it here.
 *  - Stability / cadence / metrics math lives in the NVML-free header
 *    telemetry/gpu_transition_analysis.hpp and is unit tested without a GPU.
 *  - Artifacts are written even on failure; the raw trace is never discarded.
 *
 * This file compiles in CPU-only builds so CI catches breakage; without
 * TELEMETRY_WITH_GPU `main` explains the situation and exits non-zero.
 */

#ifndef TELEMETRY_WITH_GPU
int main() {
    std::cerr <<
        "gpu_clock_transition_probe was built without TELEMETRY_WITH_GPU.\n"
        "Rebuild common/telemetry with -DWITH_GPU=ON on a node with the NVIDIA\n"
        "driver + NVML to run a real transition measurement (F1-GPU-002).\n";
    return 2;
}
#else

namespace {

namespace gt = telemetry::gpu_transition;
namespace fs = std::filesystem;
using telemetry::experiment::now_ns;
using telemetry::experiment::json_escape;

// --------------------------------------------------------------------------
// CLI
// --------------------------------------------------------------------------

struct Options {
    std::string workload_cmd;                 // required: sustained CUDA load, run via `sh -c`
    std::string gpu = "";                     // index or UUID; "" -> resolve below
    std::string from_clock = "REF";           // "REF" or integer MHz
    std::string to_clock;                     // required: integer MHz
    int64_t probe_interval_ns = 5'000'000;    // 5 ms fine probe (NOT the campaign cadence)
    int      tolerance_mhz = -1;              // required
    int      stable_consecutive = 3;
    int64_t  max_wait_ns = 3'000'000'000;     // deadline for stability after t_solicitud
    int64_t  warmup_ns = 2'000'000'000;
    int64_t  workload_min_active_ns = 4'000'000'000;  // must stay active this long after warmup
    int64_t  request_at_ns = -1;             // relative to workload start; -1 -> warmup + min_active/2
    int64_t  source_settle_ns = 500'000'000; // used only when from_clock == REF
    unsigned int active_util_threshold_pct = 5;
    std::string out_dir;                      // required
    std::string nvidia_smi = "nvidia-smi";
    bool     use_sudo = true;
    int      replicate_id = 0;
    std::string label = "";                   // e.g. "F0->F3"; default derived
    bool     dry_run_actuation = false;       // skip -lgc/-rgc (local smoke only, NOT a real measurement)
};

[[noreturn]] void usage_and_exit(int code) {
    std::cerr <<
        "gpu_clock_transition_probe  (F1-GPU-002)\n"
        "Measure NVML cadence and T_transicion_gpu = t_estable - t_solicitud under load.\n\n"
        "Required:\n"
        "  --workload-cmd \"<shell cmd>\"   Sustained CUDA load; run via `sh -c`.\n"
        "  --to-clock <MHz>               Destination graphics clock (snapped to a supported clock).\n"
        "  --tolerance-mhz <MHz>          Stability band; must be < half the neighbouring supported-clock gap.\n"
        "  --out-dir <path>              Directory for the three artifacts.\n\n"
        "Optional:\n"
        "  --gpu <index|UUID>            Default: $CUDA_VISIBLE_DEVICES first entry, else 0.\n"
        "  --from-clock <MHz|REF>        Source clock (REF = nvidia-smi -rgc). Default REF.\n"
        "  --probe-interval-ns <ns>      Fine probe cadence for THIS probe. Default 5e6 (5 ms).\n"
        "  --stable-consecutive <N>      Consecutive in-tolerance readings for 'stable'. Default 3.\n"
        "  --max-wait-ns <ns>           Deadline for stability after t_solicitud. Default 3e9.\n"
        "  --warmup-ns <ns>            Wait after workload start before checking activity. Default 2e9.\n"
        "  --workload-min-active-ns <ns> Load must stay active this long after warmup. Default 4e9.\n"
        "  --request-at-ns <ns>         When (after workload start) to issue -lgc. Default warmup + min_active/2.\n"
        "  --source-settle-ns <ns>      Settle wait after -rgc when --from-clock REF. Default 5e8.\n"
        "  --active-util-threshold-pct <p> GPU counts as active at/above this util. Default 5.\n"
        "  --nvidia-smi <path>          Default 'nvidia-smi'.\n"
        "  --no-sudo                    Do not prefix actuation with sudo (default: sudo, as gpu_freqctl.py).\n"
        "  --replicate-id <int>         Metadata tag for aggregation. Default 0.\n"
        "  --label <str>               Transition label. Default '<from>-><to>'.\n"
        "  --dry-run-actuation          Skip -lgc/-rgc. Local smoke ONLY -- not a valid measurement.\n";
    std::exit(code);
}

int64_t parse_i64(const char* s, const char* flag) {
    try { return std::stoll(s); }
    catch (...) { std::cerr << "invalid integer for " << flag << ": " << s << "\n"; std::exit(2); }
}

Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { std::cerr << "missing value for " << argv[i] << "\n"; std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") usage_and_exit(0);
        else if (a == "--workload-cmd") o.workload_cmd = need(i);
        else if (a == "--gpu") o.gpu = need(i);
        else if (a == "--from-clock") o.from_clock = need(i);
        else if (a == "--to-clock") o.to_clock = need(i);
        else if (a == "--probe-interval-ns") o.probe_interval_ns = parse_i64(need(i), "--probe-interval-ns");
        else if (a == "--tolerance-mhz") o.tolerance_mhz = static_cast<int>(parse_i64(need(i), "--tolerance-mhz"));
        else if (a == "--stable-consecutive") o.stable_consecutive = static_cast<int>(parse_i64(need(i), "--stable-consecutive"));
        else if (a == "--max-wait-ns") o.max_wait_ns = parse_i64(need(i), "--max-wait-ns");
        else if (a == "--warmup-ns") o.warmup_ns = parse_i64(need(i), "--warmup-ns");
        else if (a == "--workload-min-active-ns") o.workload_min_active_ns = parse_i64(need(i), "--workload-min-active-ns");
        else if (a == "--request-at-ns") o.request_at_ns = parse_i64(need(i), "--request-at-ns");
        else if (a == "--source-settle-ns") o.source_settle_ns = parse_i64(need(i), "--source-settle-ns");
        else if (a == "--active-util-threshold-pct") o.active_util_threshold_pct = static_cast<unsigned int>(parse_i64(need(i), "--active-util-threshold-pct"));
        else if (a == "--out-dir") o.out_dir = need(i);
        else if (a == "--nvidia-smi") o.nvidia_smi = need(i);
        else if (a == "--no-sudo") o.use_sudo = false;
        else if (a == "--replicate-id") o.replicate_id = static_cast<int>(parse_i64(need(i), "--replicate-id"));
        else if (a == "--label") o.label = need(i);
        else if (a == "--dry-run-actuation") o.dry_run_actuation = true;
        else { std::cerr << "unknown argument: " << a << "\n"; usage_and_exit(2); }
    }
    if (o.workload_cmd.empty()) { std::cerr << "--workload-cmd is required\n"; usage_and_exit(2); }
    if (o.to_clock.empty())     { std::cerr << "--to-clock is required\n"; usage_and_exit(2); }
    if (o.tolerance_mhz < 0)     { std::cerr << "--tolerance-mhz is required\n"; usage_and_exit(2); }
    if (o.out_dir.empty())      { std::cerr << "--out-dir is required\n"; usage_and_exit(2); }
    if (o.stable_consecutive < 1) o.stable_consecutive = 1;
    if (o.request_at_ns < 0) o.request_at_ns = o.warmup_ns + o.workload_min_active_ns / 2;
    if (o.label.empty()) o.label = o.from_clock + "->" + o.to_clock;
    if (o.gpu.empty()) {
        const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
        std::string v = cvd ? cvd : "";
        auto comma = v.find(',');
        o.gpu = v.empty() ? "0" : v.substr(0, comma);
    }
    return o;
}

// --------------------------------------------------------------------------
// Signal-safe stop flag + restore
// --------------------------------------------------------------------------

volatile std::sig_atomic_t g_stop = 0;
volatile std::sig_atomic_t g_stop_count = 0;
pid_t g_workload_pid = -1;

extern "C" void on_signal(int) {
    g_stop = 1;
    ++g_stop_count;
    // Negative pid -> the workload's process group (child calls setpgid(0,0)),
    // so `sh -c` wrappers forward the signal to the real load. async-signal-safe.
    if (g_workload_pid > 1) ::kill(-g_workload_pid, SIGTERM);
    // Do not use _exit here: it skips atexit and could leave -lgc applied.
    // The command runner is interruptible and the normal cleanup path is
    // bounded, so a second signal only reinforces the stop request.
}

// argv for `nvidia-smi -rgc`, kept global so an atexit handler can run it even
// on an unexpected exit path.
std::vector<std::string> g_rgc_argv;
bool g_actuation_touched_clock = false;
bool g_restore_done = false;
bool g_restore_ok = false;

int run_command_blocking(const std::vector<std::string>& args, int64_t& t_return_ns,
                         int64_t timeout_ns = 30'000'000'000LL,
                         bool interrupt_on_stop = true);

void run_restore() {
    if (g_restore_done) return;
    g_restore_done = true;
    if (!g_actuation_touched_clock || g_rgc_argv.empty()) { g_restore_ok = true; return; }
    int64_t ignored = 0;
    // A signal must stop the workload, but it must not cancel the restoration
    // command itself.  Give -rgc its bounded timeout even during teardown.
    const int rc = run_command_blocking(g_rgc_argv, ignored, 30'000'000'000LL,
                                        /*interrupt_on_stop=*/false);
    g_restore_ok = (rc == 0);
    if (!g_restore_ok) {
        std::cerr << "WARNING: `nvidia-smi -rgc` restore returned " << rc
                  << " -- verify the GPU clock lock was cleared manually.\n";
    }
}

void atexit_restore() { run_restore(); }

// --------------------------------------------------------------------------
// Small process / hashing helpers
// --------------------------------------------------------------------------

int run_command_blocking(const std::vector<std::string>& args, int64_t& t_return_ns,
                         int64_t timeout_ns, bool interrupt_on_stop) {
    std::vector<char*> c_argv;
    c_argv.reserve(args.size() + 1);
    for (const auto& s : args) c_argv.push_back(const_cast<char*>(s.c_str()));
    c_argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) { t_return_ns = now_ns(); return -1; }
    if (pid == 0) {
        ::execvp(c_argv[0], c_argv.data());
        ::_exit(127);
    }
    int status = 0;
    const int64_t deadline = static_cast<int64_t>(now_ns()) + timeout_ns;
    while (true) {
        const pid_t w = ::waitpid(pid, &status, WNOHANG);
        if (w == pid) break;
        if (w == 0) {
            if ((interrupt_on_stop && g_stop) || static_cast<int64_t>(now_ns()) >= deadline) {
                ::kill(pid, SIGTERM);
                (void)::waitpid(pid, &status, 0);
                t_return_ns = now_ns();
                return g_stop ? -3 : -2;
            }
            struct timespec nap{0, 10'000'000};
            nanosleep(&nap, nullptr);
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0) { t_return_ns = now_ns(); return -1; }
    }
    t_return_ns = now_ns();
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::string csv_escape(const std::string& text) {
    std::string out{"\""};
    for (char c : text) {
        if (c == '\"') out += "\"\"";
        else out += c;
    }
    out += '\"';
    return out;
}

std::string sha256_of_file(const std::string& path) {
    if (path.empty() || !fs::exists(path)) return "";
    // Portable: shell out to sha256sum. This is metadata, not a hot path.
    std::string cmd = "sha256sum '" + path + "' 2>/dev/null";
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return "";
    std::array<char, 128> buf{};
    std::string out;
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), p)) out += buf.data();
    ::pclose(p);
    auto sp = out.find(' ');
    return sp == std::string::npos ? "" : out.substr(0, sp);
}

// First whitespace-separated token of the workload command, for checksumming.
std::string workload_argv0(const std::string& cmd) {
    std::istringstream is(cmd);
    std::string tok;
    is >> tok;
    return tok;
}

// --------------------------------------------------------------------------
// Artifact writers
// --------------------------------------------------------------------------

struct RawRow {
    int64_t seq = 0;
    gt::ClockReading r;
    const char* phase = "pre_request";   // "pre_request" | "post_request"
};

struct ProbeReport {
    Options opt;
    // static metadata
    std::string gpu_name, gpu_uuid, driver_version, cuda_version;
    std::vector<unsigned int> supported_sm_clocks;
    bool supported_clocks_available = false;
    int from_mhz = -1;               // -1 == REF
    int to_mhz = -1;
    std::string workload_checksum;
    // timeline (monotonic ns; 0 == not reached)
    int64_t t_workload_start_ns = 0;
    int64_t t_warmup_done_ns = 0;
    int64_t t_source_locked_ns = 0;
    int64_t t_source_stable_ns = 0;
    int64_t t_request_ns = 0;
    int64_t t_command_return_ns = 0;
    int64_t t_stable_ns = 0;
    int command_returncode = -1;
    // outcome
    std::string result = "not_started";  // stable|timeout|aborted|workload_inactive|source_not_stable|command_error|no_gpu
    std::string failure_reason;
    bool restoration_attempted = false;
    bool restoration_confirmed = false;
    // derived
    gt::CadenceStats cadence;
    gt::TransitionMetrics metrics;
    gt::SignalStepStats step_power, step_util, step_mem_util, step_temperature,
                        step_energy, step_graphics_clock, step_sm_clock;
    std::vector<RawRow> raw;
};

void write_raw_csv(const ProbeReport& rep, const fs::path& path) {
    std::ofstream out(path);
    out << "replicate_id,label,from_clock,to_clock,seq,phase,t_mono_ns,dt_ns_from_prev,"
           "graphics_clock_mhz,graphics_clock_valid,sm_clock_mhz,sm_clock_valid,util_pct,util_valid,mem_util_pct,mem_util_valid,"
           "power_mw,power_valid,temperature_c,temperature_valid,energy_mj,energy_valid,"
           "throttle_reasons_hex,throttle_valid\n";
    int64_t prev = 0;
    bool have_prev = false;
    for (const auto& row : rep.raw) {
        const auto& r = row.r;
        const int64_t dt = have_prev ? (r.t_mono_ns - prev) : 0;
        prev = r.t_mono_ns; have_prev = true;
        out << rep.opt.replicate_id << ',' << csv_escape(rep.opt.label) << ','
            << rep.opt.from_clock << ',' << rep.opt.to_clock << ','
            << row.seq << ',' << row.phase << ',' << r.t_mono_ns << ',' << dt << ','
            << r.graphics_clock_mhz << ',' << (r.graphics_clock_valid ? 1 : 0) << ','
            << r.sm_clock_mhz << ',' << (r.sm_clock_valid ? 1 : 0) << ','
            << r.util_pct << ',' << (r.util_valid ? 1 : 0) << ','
            << r.mem_util_pct << ',' << (r.mem_util_valid ? 1 : 0) << ','
            << r.power_mw << ',' << (r.power_valid ? 1 : 0) << ','
            << r.temperature_c << ',' << (r.temperature_valid ? 1 : 0) << ','
            << r.energy_mj << ',' << (r.energy_valid ? 1 : 0) << ','
            << "0x" << std::hex << r.throttle_reasons << std::dec << ','
            << (r.throttle_valid ? 1 : 0) << '\n';
    }
}

void write_matrix_csv(const ProbeReport& rep, const fs::path& path) {
    std::ofstream out(path);
    out << "replicate_id,label,from_clock,to_clock_mhz,result,failure_reason,"
           "t_actuacion_ns,command_latency_ns,settle_after_command_ns,"
           "conservative_upper_bound_ns,optimistic_ns,metrics_valid,"
           "probe_interval_p50_ns,probe_interval_p95_ns,n_readings,"
           "gpu_uuid,driver_version,cuda_version\n";
    out << rep.opt.replicate_id << ',' << csv_escape(rep.opt.label) << ','
        << rep.opt.from_clock << ',' << rep.to_mhz << ','
        << rep.result << ',' << csv_escape(rep.failure_reason) << ','
        << rep.metrics.t_actuacion_ns << ',' << rep.metrics.command_latency_ns << ','
        << rep.metrics.settle_after_command_ns << ',' << rep.metrics.conservative_upper_bound_ns << ','
        << rep.metrics.optimistic_ns << ',' << (rep.metrics.valid ? 1 : 0) << ','
        << rep.cadence.p50_delta_ns << ',' << rep.cadence.p95_delta_ns << ','
        << rep.raw.size() << ',' << csv_escape(rep.gpu_uuid) << ','
        << csv_escape(rep.driver_version) << ',' << csv_escape(rep.cuda_version) << '\n';
}

void write_summary_json(const ProbeReport& rep, const fs::path& path) {
    std::ofstream o(path);
    auto q = [](const std::string& s) { return "\"" + json_escape(s) + "\""; };
    o << "{\n";
    o << "  \"schema\": \"f1-gpu-002/gpu_clock_transition_summary/1\",\n";
    o << "  \"replicate_id\": " << rep.opt.replicate_id << ",\n";
    o << "  \"label\": " << q(rep.opt.label) << ",\n";
    o << "  \"gpu\": {\n";
    o << "    \"selector\": " << q(rep.opt.gpu) << ",\n";
    o << "    \"name\": " << q(rep.gpu_name) << ",\n";
    o << "    \"uuid\": " << q(rep.gpu_uuid) << ",\n";
    o << "    \"driver_version\": " << q(rep.driver_version) << ",\n";
    o << "    \"cuda_version\": " << q(rep.cuda_version) << "\n";
    o << "  },\n";
    o << "  \"supported_sm_clocks_available\": " << (rep.supported_clocks_available ? "true" : "false") << ",\n";
    o << "  \"supported_sm_clocks_mhz\": [";
    for (size_t i = 0; i < rep.supported_sm_clocks.size(); ++i)
        o << (i ? "," : "") << rep.supported_sm_clocks[i];
    o << "],\n";
    o << "  \"workload_cmd\": " << q(rep.opt.workload_cmd) << ",\n";
    o << "  \"workload_checksum_sha256\": " << q(rep.workload_checksum) << ",\n";
    o << "  \"from_clock\": " << q(rep.opt.from_clock) << ",\n";
    o << "  \"from_clock_mhz\": " << rep.from_mhz << ",\n";
    o << "  \"to_clock_mhz\": " << rep.to_mhz << ",\n";
    o << "  \"stability_criterion\": {\n";
    o << "    \"tolerance_mhz\": " << rep.opt.tolerance_mhz << ",\n";
    o << "    \"required_consecutive\": " << rep.opt.stable_consecutive << ",\n";
    o << "    \"active_util_threshold_pct\": " << rep.opt.active_util_threshold_pct << ",\n";
    o << "    \"invalidating_throttle_mask_hex\": \"0x" << std::hex
      << gt::kDefaultInvalidatingThrottleMask << std::dec << "\"\n";
    o << "  },\n";
    o << "  \"probe_interval_ns_requested\": " << rep.opt.probe_interval_ns << ",\n";
    o << "  \"warmup_ns\": " << rep.opt.warmup_ns << ",\n";
    o << "  \"request_at_ns\": " << rep.opt.request_at_ns << ",\n";
    o << "  \"max_wait_ns\": " << rep.opt.max_wait_ns << ",\n";
    o << "  \"dry_run_actuation\": " << (rep.opt.dry_run_actuation ? "true" : "false") << ",\n";
    o << "  \"timeline_mono_ns\": {\n";
    o << "    \"t_workload_start\": " << rep.t_workload_start_ns << ",\n";
    o << "    \"t_warmup_done\": " << rep.t_warmup_done_ns << ",\n";
    o << "    \"t_source_locked\": " << rep.t_source_locked_ns << ",\n";
    o << "    \"t_source_stable\": " << rep.t_source_stable_ns << ",\n";
    o << "    \"t_solicitud\": " << rep.t_request_ns << ",\n";
    o << "    \"t_command_return\": " << rep.t_command_return_ns << ",\n";
    o << "    \"t_estable\": " << rep.t_stable_ns << "\n";
    o << "  },\n";
    o << "  \"command_returncode\": " << rep.command_returncode << ",\n";
    o << "  \"observed_cadence\": {\n";
    o << "    \"n_intervals\": " << rep.cadence.n_intervals << ",\n";
    o << "    \"p50_delta_ns\": " << rep.cadence.p50_delta_ns << ",\n";
    o << "    \"p95_delta_ns\": " << rep.cadence.p95_delta_ns << ",\n";
    o << "    \"min_delta_ns\": " << rep.cadence.min_delta_ns << ",\n";
    o << "    \"max_delta_ns\": " << rep.cadence.max_delta_ns << "\n";
    o << "  },\n";
    auto dump_steps = [&](const char* name, const gt::SignalStepStats& s, bool last) {
        o << "    \"" << name << "\": {\n";
        o << "      \"n_valid\": " << s.n_valid << ",\n";
        o << "      \"n_consecutive_changes_lower_bound\": " << s.n_consecutive_changes << ",\n";
        o << "      \"redundancy_ratio\": " << s.redundancy_ratio << ",\n";
        o << "      \"median_step_duration_ns\": " << s.median_step_duration_ns << ",\n";
        o << "      \"max_step_duration_ns\": " << s.max_step_duration_ns << ",\n";
        o << "      \"observed_update_rate_hz_lower_bound\": " << s.observed_update_rate_hz_lower_bound << "\n";
        o << "    }" << (last ? "\n" : ",\n");
    };
    o << "  \"signal_step_analysis\": {\n";
    o << "    \"_note\": \"n_consecutive_changes_lower_bound and observed_update_rate_hz_lower_bound are LOWER BOUNDS on physical sensor updates, never a confirmed physical rate. A fine probe interval does not turn redundant readings into independent 5 ms observations.\",\n";
    dump_steps("power_mw", rep.step_power, false);
    dump_steps("util_pct", rep.step_util, false);
    dump_steps("mem_util_pct", rep.step_mem_util, false);
    dump_steps("temperature_c", rep.step_temperature, false);
    dump_steps("energy_mj", rep.step_energy, false);
    dump_steps("graphics_clock_mhz", rep.step_graphics_clock, false);
    dump_steps("sm_clock_mhz", rep.step_sm_clock, true);
    o << "  },\n";
    o << "  \"transition_metrics\": {\n";
    o << "    \"valid\": " << (rep.metrics.valid ? "true" : "false") << ",\n";
    o << "    \"command_latency_ns\": " << rep.metrics.command_latency_ns << ",\n";
    o << "    \"t_actuacion_ns\": " << rep.metrics.t_actuacion_ns << ",\n";
    o << "    \"settle_after_command_ns\": " << rep.metrics.settle_after_command_ns << ",\n";
    o << "    \"conservative_upper_bound_ns\": " << rep.metrics.conservative_upper_bound_ns << ",\n";
    o << "    \"optimistic_ns\": " << rep.metrics.optimistic_ns << ",\n";
    o << "    \"_note\": \"t_actuacion_ns / conservative_upper_bound_ns is an OBSERVABLE UPPER BOUND (MHz-granular clock, finite poll cadence), safe for min_dwell_ns / --t-transicion-gpu-ns. optimistic_ns is context only.\"\n";
    o << "  },\n";
    o << "  \"result\": " << q(rep.result) << ",\n";
    o << "  \"failure_reason\": " << q(rep.failure_reason) << ",\n";
    o << "  \"restoration\": { \"attempted\": " << (rep.restoration_attempted ? "true" : "false")
      << ", \"confirmed\": " << (rep.restoration_confirmed ? "true" : "false") << " },\n";
    o << "  \"n_raw_readings\": " << rep.raw.size() << "\n";
    o << "}\n";
}

void write_all_artifacts(const ProbeReport& rep) {
    std::error_code ec;
    fs::create_directories(rep.opt.out_dir, ec);
    const fs::path dir(rep.opt.out_dir);
    write_raw_csv(rep, dir / "gpu_clock_transition_raw.csv");
    write_matrix_csv(rep, dir / "gpu_clock_transition_matrix.csv");
    write_summary_json(rep, dir / "gpu_clock_transition_summary.json");
    std::cerr << "artifacts written under " << rep.opt.out_dir << "\n";
}

}  // namespace

namespace {

// ---- NVML helpers ----

std::string nvml_str(nvmlReturn_t rc) { return nvmlErrorString(rc); }

bool resolve_device(const std::string& sel, nvmlDevice_t& dev, std::string& err) {
    nvmlReturn_t rc;
    if (sel.find('-') != std::string::npos || sel.rfind("GPU-", 0) == 0) {
        rc = nvmlDeviceGetHandleByUUID(sel.c_str(), &dev);
    } else {
        unsigned int idx = 0;
        try { idx = static_cast<unsigned int>(std::stoul(sel)); }
        catch (...) { err = "cannot parse --gpu as index or UUID: " + sel; return false; }
        rc = nvmlDeviceGetHandleByIndex_v2(idx, &dev);
    }
    if (rc != NVML_SUCCESS) { err = "nvmlDeviceGetHandle*: " + nvml_str(rc); return false; }
    return true;
}

std::vector<unsigned int> supported_sm_clocks(nvmlDevice_t dev) {
    std::vector<unsigned int> out;
    unsigned int n_mem = 0;
    if (nvmlDeviceGetSupportedMemoryClocks(dev, &n_mem, nullptr) != NVML_ERROR_INSUFFICIENT_SIZE &&
        n_mem == 0) {
        return out;
    }
    std::vector<unsigned int> mem(n_mem ? n_mem : 8);
    n_mem = static_cast<unsigned int>(mem.size());
    if (nvmlDeviceGetSupportedMemoryClocks(dev, &n_mem, mem.data()) != NVML_SUCCESS) return out;
    mem.resize(n_mem);
    for (unsigned int mclk : mem) {
        unsigned int n_g = 0;
        nvmlDeviceGetSupportedGraphicsClocks(dev, mclk, &n_g, nullptr);
        if (n_g == 0) continue;
        std::vector<unsigned int> g(n_g);
        if (nvmlDeviceGetSupportedGraphicsClocks(dev, mclk, &n_g, g.data()) != NVML_SUCCESS) continue;
        g.resize(n_g);
        out.insert(out.end(), g.begin(), g.end());
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

int snap_to_supported(int requested, const std::vector<unsigned int>& supported, bool& exact) {
    exact = false;
    if (supported.empty()) return requested;
    int best = static_cast<int>(supported.front());
    long best_d = std::labs(static_cast<long>(best) - requested);
    for (unsigned int c : supported) {
        const long d = std::labs(static_cast<long>(c) - requested);
        if (d < best_d) { best_d = d; best = static_cast<int>(c); }
    }
    exact = (best_d == 0);
    return best;
}

// Smallest gap between `mhz` and its neighbours in the supported list.
int neighbour_gap(int mhz, const std::vector<unsigned int>& supported) {
    int gap = INT32_MAX;
    for (unsigned int c : supported) {
        const int d = std::abs(static_cast<int>(c) - mhz);
        if (d > 0 && d < gap) gap = d;
    }
    return gap;
}

gt::ClockReading sample_once(nvmlDevice_t dev) {
    gt::ClockReading r;
    r.t_poll_start_ns = static_cast<int64_t>(now_ns());
    unsigned int v = 0;
    r.graphics_clock_valid = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &v) == NVML_SUCCESS);
    r.graphics_clock_mhz = r.graphics_clock_valid ? v : 0;
    // Timestamp AFTER the actuation-domain query: this makes a stable reading
    // an observable upper bound rather than a timestamp from before its value.
    r.t_mono_ns = static_cast<int64_t>(now_ns());
    v = 0;
    r.sm_clock_valid = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &v) == NVML_SUCCESS);
    r.sm_clock_mhz = r.sm_clock_valid ? v : 0;
    nvmlUtilization_t u{};
    if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) {
        r.util_pct = u.gpu; r.util_valid = true;
        r.mem_util_pct = u.memory; r.mem_util_valid = true;
    }
    unsigned int p = 0;
    r.power_valid = (nvmlDeviceGetPowerUsage(dev, &p) == NVML_SUCCESS);
    r.power_mw = r.power_valid ? p : 0;
    unsigned int t = 0;
    r.temperature_valid = (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS);
    r.temperature_c = r.temperature_valid ? t : 0;
    unsigned long long e = 0;
    r.energy_valid = (nvmlDeviceGetTotalEnergyConsumption(dev, &e) == NVML_SUCCESS);
    r.energy_mj = r.energy_valid ? e : 0;
    unsigned long long thr = 0;
    r.throttle_valid = (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &thr) == NVML_SUCCESS);
    r.throttle_reasons = r.throttle_valid ? thr : 0;
    return r;
}

void sleep_until_ns(int64_t target_ns) {
    struct timespec ts;
    ts.tv_sec = static_cast<time_t>(target_ns / 1'000'000'000);
    ts.tv_nsec = static_cast<long>(target_ns % 1'000'000'000);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
        if (g_stop) return;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    ProbeReport rep;
    rep.opt = opt;
    rep.workload_checksum = sha256_of_file(workload_argv0(opt.workload_cmd));

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::atexit(atexit_restore);

    // ---- NVML init + static metadata ----
    nvmlReturn_t rc = nvmlInit_v2();
    if (rc != NVML_SUCCESS) {
        rep.result = "no_gpu";
        rep.failure_reason = "nvmlInit_v2: " + nvml_str(rc);
        write_all_artifacts(rep);
        return 3;
    }
    nvmlDevice_t dev{};
    std::string err;
    if (!resolve_device(opt.gpu, dev, err)) {
        rep.result = "no_gpu"; rep.failure_reason = err;
        nvmlShutdown(); write_all_artifacts(rep); return 3;
    }
    { char buf[NVML_DEVICE_NAME_BUFFER_SIZE] = {0};
      if (nvmlDeviceGetName(dev, buf, sizeof buf) == NVML_SUCCESS) rep.gpu_name = buf; }
    { char buf[NVML_DEVICE_UUID_BUFFER_SIZE] = {0};
      if (nvmlDeviceGetUUID(dev, buf, sizeof buf) == NVML_SUCCESS) rep.gpu_uuid = buf; }
    { char buf[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
      if (nvmlSystemGetDriverVersion(buf, sizeof buf) == NVML_SUCCESS) rep.driver_version = buf; }
    { int cv = 0;
      if (nvmlSystemGetCudaDriverVersion_v2(&cv) == NVML_SUCCESS) {
          rep.cuda_version = std::to_string(cv / 1000) + "." + std::to_string((cv % 1000) / 10);
      } }
    rep.supported_sm_clocks = supported_sm_clocks(dev);
    rep.supported_clocks_available = !rep.supported_sm_clocks.empty();

    // ---- resolve + validate target clock and tolerance ----
    const bool from_ref = (opt.from_clock == "REF" || opt.from_clock == "ref");
    if (!from_ref) {
        bool exact = false;
        rep.from_mhz = snap_to_supported(static_cast<int>(parse_i64(opt.from_clock.c_str(), "--from-clock")),
                                         rep.supported_sm_clocks, exact);
        opt.from_clock = std::to_string(rep.from_mhz);  // normalise for artifacts
    }
    {
        bool exact = false;
        rep.to_mhz = snap_to_supported(static_cast<int>(parse_i64(opt.to_clock.c_str(), "--to-clock")),
                                       rep.supported_sm_clocks, exact);
        opt.to_clock = std::to_string(rep.to_mhz);
        rep.opt = opt;
        if (rep.supported_clocks_available && !exact) {
            std::cerr << "NOTE: --to-clock snapped to nearest supported " << rep.to_mhz << " MHz\n";
        }
    }
    if (rep.supported_clocks_available) {
        const int gap = neighbour_gap(rep.to_mhz, rep.supported_sm_clocks);
        if (gap != INT32_MAX && opt.tolerance_mhz * 2 >= gap) {
            rep.result = "aborted";
            rep.failure_reason = "tolerance_mhz (" + std::to_string(opt.tolerance_mhz) +
                ") must be < half the neighbouring supported-clock gap (" + std::to_string(gap) + ")";
            nvmlShutdown(); write_all_artifacts(rep); return 2;
        }
    }

    // actuation argv (mirrors gpu_freqctl.py._default_run_nvidia_smi / restore_gpu_state)
    auto smi_argv = [&](std::vector<std::string> tail) {
        std::vector<std::string> a;
        if (opt.use_sudo) { a.push_back("sudo"); a.push_back("-n"); }
        a.push_back(opt.nvidia_smi);
        a.push_back("-i"); a.push_back(opt.gpu);
        for (auto& t : tail) a.push_back(std::move(t));
        return a;
    };
    g_rgc_argv = smi_argv({"-rgc"});

    // ---- launch sustained workload ----
    rep.t_workload_start_ns = static_cast<int64_t>(now_ns());
    const pid_t wpid = ::fork();
    if (wpid < 0) {
        rep.result = "aborted"; rep.failure_reason = "fork(workload) failed";
        nvmlShutdown(); write_all_artifacts(rep); return 4;
    }
    if (wpid == 0) {
        ::setpgid(0, 0);  // own process group -> we can signal the whole load
        execl("/bin/sh", "sh", "-c", opt.workload_cmd.c_str(), (char*)nullptr);
        ::_exit(127);
    }
    ::setpgid(wpid, wpid);  // race-free: also set from the parent side
    g_workload_pid = wpid;

    auto workload_alive = [&]() {
        if (g_workload_pid <= 1) return false;
        int st = 0;
        const pid_t w = ::waitpid(wpid, &st, WNOHANG);
        if (w == wpid) { g_workload_pid = -1; return false; }  // exited & reaped here
        return w == 0;  // 0 -> still running; <0 -> already gone
    };
    auto abort_with = [&](const char* result, const std::string& why) {
        rep.result = result; rep.failure_reason = why;
    };

    std::vector<RawRow> raw;
    int64_t seq = 0;
    auto poll_push = [&](const char* phase) {
        gt::ClockReading r = sample_once(dev);
        raw.push_back(RawRow{seq++, r, phase});
        return r;
    };

    // ---- warmup ----
    const int64_t start = rep.t_workload_start_ns;
    int64_t next_tick = start;
    bool failed = false;
    while (!g_stop) {
        const int64_t t = static_cast<int64_t>(now_ns());
        if (t - start >= opt.warmup_ns) break;
        if (!workload_alive()) { abort_with("workload_inactive", "workload exited during warmup"); failed = true; break; }
        poll_push("pre_request");
        next_tick += opt.probe_interval_ns;
        sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
    }
    if (g_stop && !failed) { abort_with("aborted", "SIGINT/SIGTERM during warmup"); failed = true; }
    if (!failed) rep.t_warmup_done_ns = static_cast<int64_t>(now_ns());

    // ---- confirm activity (need `stable_consecutive` active reads) ----
    if (!failed) {
        int active_run = 0;
        int checks = 0;
        while (!g_stop && checks < 50) {
            gt::ClockReading r = poll_push("pre_request");
            ++checks;
            const bool active = r.util_valid && r.util_pct >= opt.active_util_threshold_pct;
            active_run = active ? active_run + 1 : 0;
            if (active_run >= opt.stable_consecutive) break;
            if (!workload_alive()) { abort_with("workload_inactive", "workload exited before it reached sustained activity"); failed = true; break; }
            next_tick += opt.probe_interval_ns;
            sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
        }
        if (!failed && active_run < opt.stable_consecutive) {
            abort_with("workload_inactive",
                       "GPU never reached >=" + std::to_string(opt.active_util_threshold_pct) +
                       "% utilisation for " + std::to_string(opt.stable_consecutive) + " consecutive reads");
            failed = true;
        }
    }

    // ---- lock + confirm SOURCE clock under load ----
    if (!failed) {
        std::vector<std::string> src_argv = from_ref
            ? smi_argv({"-rgc"})
            : smi_argv({"-lgc", opt.from_clock + "," + opt.from_clock});
        int64_t t_ret = 0;
        int src_rc = 0;
        if (!opt.dry_run_actuation) {
            src_rc = run_command_blocking(src_argv, t_ret);
            g_actuation_touched_clock = true;
        }
        rep.t_source_locked_ns = static_cast<int64_t>(now_ns());
        if (src_rc != 0) {
            abort_with("command_error", "source actuation returned " + std::to_string(src_rc) +
                       " (cmd: " + (src_argv.empty() ? "" : src_argv.back()) + ")");
            failed = true;
        } else if (from_ref) {
            // No single target to converge to; take a fixed settle wait.
            const int64_t until = static_cast<int64_t>(now_ns()) + opt.source_settle_ns;
            while (!g_stop && static_cast<int64_t>(now_ns()) < until) {
                poll_push("pre_request");
                next_tick += opt.probe_interval_ns;
                sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
            }
            rep.t_source_stable_ns = static_cast<int64_t>(now_ns());
        } else {
            gt::StabilityConfig scfg;
            scfg.target_mhz = static_cast<unsigned int>(rep.from_mhz);
            scfg.tolerance_mhz = static_cast<unsigned int>(opt.tolerance_mhz);
            scfg.required_consecutive = opt.stable_consecutive;
            scfg.active_util_threshold_pct = opt.active_util_threshold_pct;
            const int64_t src_request_ns = static_cast<int64_t>(now_ns());
            const int64_t src_deadline = src_request_ns + opt.max_wait_ns;
            std::vector<gt::ClockReading> src_readings;
            while (!g_stop) {
                gt::ClockReading r = poll_push("pre_request");
                src_readings.push_back(r);
                auto st = gt::detect_stability(src_readings, src_request_ns, scfg, src_deadline);
                if (st.outcome == gt::StabilityOutcome::Stable) { rep.t_source_stable_ns = st.t_stable_ns; break; }
                if (st.outcome == gt::StabilityOutcome::Timeout) {
                    abort_with("source_not_stable",
                               "source clock " + std::to_string(rep.from_mhz) +
                               " MHz never held for " + std::to_string(opt.stable_consecutive) +
                               " reads within " + std::to_string(opt.max_wait_ns) + " ns");
                    failed = true; break;
                }
                if (!workload_alive()) { abort_with("workload_inactive", "workload exited while locking source clock"); failed = true; break; }
                next_tick += opt.probe_interval_ns;
                sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
            }
        }
    }
    if (g_stop && !failed) { abort_with("aborted", "SIGINT/SIGTERM before the transition request"); failed = true; }

    // ---- wait until request_at, then issue the transition and poll to stable ----
    gt::StabilityResult stability;
    if (!failed) {
        while (!g_stop) {
            const int64_t t = static_cast<int64_t>(now_ns());
            if (t - start >= opt.request_at_ns) break;
            if (!workload_alive()) { abort_with("workload_inactive", "workload exited before the transition request time"); failed = true; break; }
            poll_push("pre_request");
            next_tick += opt.probe_interval_ns;
            sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
        }
    }

    std::vector<gt::ClockReading> post_readings;
    if (!failed && !g_stop) {
        std::vector<std::string> to_argv = smi_argv({"-lgc", opt.to_clock + "," + opt.to_clock});
        rep.t_request_ns = static_cast<int64_t>(now_ns());
        int64_t t_ret = 0;
        int to_rc = 0;
        if (!opt.dry_run_actuation) {
            to_rc = run_command_blocking(to_argv, t_ret);
            g_actuation_touched_clock = true;
        } else {
            t_ret = static_cast<int64_t>(now_ns());
        }
        rep.t_command_return_ns = t_ret;
        rep.command_returncode = to_rc;
        if (to_rc != 0) {
            abort_with("command_error", "`nvidia-smi -lgc " + opt.to_clock + "` returned " + std::to_string(to_rc));
            failed = true;
        } else {
            gt::StabilityConfig tcfg;
            tcfg.target_mhz = static_cast<unsigned int>(rep.to_mhz);
            tcfg.tolerance_mhz = static_cast<unsigned int>(opt.tolerance_mhz);
            tcfg.required_consecutive = opt.stable_consecutive;
            tcfg.active_util_threshold_pct = opt.active_util_threshold_pct;
            const int64_t deadline = rep.t_request_ns + opt.max_wait_ns;
            while (!g_stop) {
                gt::ClockReading r = poll_push("post_request");
                post_readings.push_back(r);
                stability = gt::detect_stability(post_readings, rep.t_request_ns, tcfg, deadline);
                if (stability.outcome == gt::StabilityOutcome::Stable) { rep.t_stable_ns = stability.t_stable_ns; break; }
                if (stability.outcome == gt::StabilityOutcome::Timeout) {
                    abort_with("timeout", "destination clock " + std::to_string(rep.to_mhz) +
                               " MHz not confirmed stable within " + std::to_string(opt.max_wait_ns) + " ns");
                    failed = true; break;
                }
                if (!workload_alive()) { abort_with("workload_inactive", "workload exited before the destination clock settled"); failed = true; break; }
                next_tick += opt.probe_interval_ns;
                sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
            }
        }
    }
    // `workload_min_active_ns` is a contract, not merely a default used to
    // place the request.  Keep observing until that post-warmup duration has
    // actually elapsed, otherwise reject the replicate as too short.
    if (!failed && stability.outcome == gt::StabilityOutcome::Stable) {
        const int64_t min_active_end = start + opt.warmup_ns + opt.workload_min_active_ns;
        while (!g_stop && static_cast<int64_t>(now_ns()) < min_active_end) {
            if (!workload_alive()) {
                abort_with("workload_inactive", "workload ended before workload_min_active_ns elapsed");
                failed = true;
                break;
            }
            post_readings.push_back(poll_push("post_request"));
            next_tick += opt.probe_interval_ns;
            sleep_until_ns(std::max(next_tick, static_cast<int64_t>(now_ns())));
        }
    }
    if (g_stop && rep.result.empty()) abort_with("aborted", "SIGINT/SIGTERM while waiting for the destination clock");
    if (g_stop && rep.result == "not_started") abort_with("aborted", "SIGINT/SIGTERM while waiting for the destination clock");

    // ---- restore + reap ----
    rep.restoration_attempted = true;
    run_restore();
    rep.restoration_confirmed = g_restore_ok;
    if (g_workload_pid > 1) {
        ::kill(-g_workload_pid, SIGTERM);
        for (int i = 0; i < 50; ++i) {
            int st = 0;
            if (::waitpid(g_workload_pid, &st, WNOHANG) == g_workload_pid) { g_workload_pid = -1; break; }
            struct timespec nap{0, 100'000'000};
            nanosleep(&nap, nullptr);
        }
        if (g_workload_pid > 1) { ::kill(-g_workload_pid, SIGKILL); int st = 0; ::waitpid(g_workload_pid, &st, 0); g_workload_pid = -1; }
    }
    nvmlShutdown();

    // ---- assemble derived analysis ----
    rep.raw = raw;
    if (!failed && stability.outcome == gt::StabilityOutcome::Stable) rep.result = "stable";
    else if (rep.result.empty() || rep.result == "not_started") rep.result = "aborted";

    std::vector<int64_t> ts;
    std::vector<long long> pw, ut, mu, temp, energy, graphics, sm;
    std::vector<bool> pw_ok, ut_ok, mu_ok, temp_ok, energy_ok, graphics_ok, sm_ok;
    ts.reserve(raw.size());
    for (const auto& row : raw) {
        ts.push_back(row.r.t_mono_ns);
        pw.push_back(row.r.power_mw);   pw_ok.push_back(row.r.power_valid);
        ut.push_back(row.r.util_pct);   ut_ok.push_back(row.r.util_valid);
        mu.push_back(row.r.mem_util_pct); mu_ok.push_back(row.r.mem_util_valid);
        temp.push_back(row.r.temperature_c); temp_ok.push_back(row.r.temperature_valid);
        energy.push_back(static_cast<long long>(row.r.energy_mj)); energy_ok.push_back(row.r.energy_valid);
        graphics.push_back(row.r.graphics_clock_mhz); graphics_ok.push_back(row.r.graphics_clock_valid);
        sm.push_back(row.r.sm_clock_mhz); sm_ok.push_back(row.r.sm_clock_valid);
    }
    // The command invocation deliberately blocks polling.  It must not be
    // mistaken for collector jitter when selecting q_produccion, so cadence
    // is computed from the uninterrupted post-request polling segment.
    std::vector<int64_t> steady_ts;
    steady_ts.reserve(post_readings.size());
    for (const auto& reading : post_readings) steady_ts.push_back(reading.t_mono_ns);
    rep.cadence = gt::compute_cadence_stats(steady_ts);
    rep.step_power = gt::analyze_signal_steps(ts, pw, pw_ok);
    rep.step_util = gt::analyze_signal_steps(ts, ut, ut_ok);
    rep.step_mem_util = gt::analyze_signal_steps(ts, mu, mu_ok);
    rep.step_temperature = gt::analyze_signal_steps(ts, temp, temp_ok);
    rep.step_energy = gt::analyze_signal_steps(ts, energy, energy_ok);
    rep.step_graphics_clock = gt::analyze_signal_steps(ts, graphics, graphics_ok);
    rep.step_sm_clock = gt::analyze_signal_steps(ts, sm, sm_ok);
    rep.metrics = gt::compute_transition_metrics(rep.t_request_ns, rep.t_command_return_ns,
                                                 post_readings, stability);

    write_all_artifacts(rep);

    std::cerr << "result=" << rep.result;
    if (rep.metrics.valid) {
        std::cerr << "  T_actuacion=" << rep.metrics.t_actuacion_ns << "ns"
                  << "  command_latency=" << rep.metrics.command_latency_ns << "ns"
                  << "  conservative_upper_bound=" << rep.metrics.conservative_upper_bound_ns << "ns";
    } else if (!rep.failure_reason.empty()) {
        std::cerr << "  (" << rep.failure_reason << ")";
    }
    std::cerr << "\n";
    return rep.result == "stable" ? 0 : 1;
}

#endif  // TELEMETRY_WITH_GPU
