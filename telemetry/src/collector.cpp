#include "telemetry/collector.hpp"
#include <sched.h>
#include <time.h>
#include <stdexcept>
#include <cstring>
#include <string>
#include <utility>

/**
 * @file collector.cpp
 * @brief Producer thread implementation for the telemetry subsystem.
 *
 * This file is the main hot-path implementation. Keep it conservative: backend
 * setup may allocate or throw in start(), but run() should stay limited to
 * timestamping, reader read() calls, ring pushes, and absolute sleeping.
 */
namespace telemetry
{
    namespace {
        std::runtime_error pthread_error(const char* context, int error_code) {
            return std::runtime_error(std::string(context) + ": " + std::strerror(error_code));
        }
    }

    Collector::Collector(CollectorConfig cfg, Ring& ring)
        : cfg_(std::move(cfg)),
          ring_(ring),
          perf_reader_(cfg_.target_pid, -1),
          perf_cgroup_reader_(cfg_.perf_cgroup_path, cfg_.perf_cpus),
          rapl_reader_(cfg_.rapl_pkg_path, cfg_.rapl_dram_path),
          nvml_reader_(0) {}

    Collector::~Collector(){
        stop();
    }

    void Collector::start(){
        if(thread_started_) return;
        if(cfg_.interval_ns <= 0){
            throw std::invalid_argument("Collector interval_ns must be positive");
        }
        if(cfg_.enable_gpu && !NvmlReader::compiled_with_gpu()){
            throw std::runtime_error("GPU telemetry requested but telemetry was built without NVML support");
        }

        try {
            // Open only the enabled backends. This keeps local tests usable
            // without PMU permissions and avoids unnecessary work per sample.
            if(cfg_.enable_perf) {
                if(!cfg_.perf_cgroup_path.empty()) {
                    perf_cgroup_reader_.open();
                } else {
                    perf_reader_.open();
                }
            }
            if(!cfg_.rapl_pkg_path.empty()) rapl_reader_.open();
            if(cfg_.enable_gpu) nvml_reader_.open();
        } catch (...) {
            close_readers();
            running_.store(false);
            stop_flag_.store(true, std::memory_order_relaxed);
            throw;
        }

        // Reset experiment-local state only after all enabled readers are open.
        push_retries_.store(0, std::memory_order_relaxed);
        stop_flag_.store(false, std::memory_order_relaxed);

        pthread_attr_t attr;
        int rc = pthread_attr_init(&attr);
        if(rc != 0){
            close_readers();
            throw pthread_error("pthread_attr_init failed", rc);
        }

        // Optional producer pinning is applied before thread creation so the
        // sampling loop starts on the requested CPU instead of migrating later.
        if (cfg_.producer_cpu >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cfg_.producer_cpu, &cpuset);
            rc = pthread_attr_setaffinity_np(&attr, sizeof(cpuset), &cpuset);
            if(rc != 0){
                pthread_attr_destroy(&attr);
                close_readers();
                throw pthread_error("pthread_attr_setaffinity_np failed", rc);
            }
        }

        rc = pthread_create(&thread_, &attr, thread_entry, this);
        pthread_attr_destroy(&attr);
        if(rc != 0){
            running_.store(false);
            close_readers();
            throw pthread_error("pthread_create failed", rc);
        }
        thread_started_ = true;
        running_.store(true);
    }

    void* Collector::thread_entry(void* arg){
        static_cast<Collector*>(arg)->run();
        return nullptr;
    }
    
    void Collector::run(){
        running_.store(true);

        // Absolute scheduling reduces accumulated drift compared with relative
        // nanosleep. Jitter is still measured and exported by the launcher.
        struct timespec next_wake;
        clock_gettime(CLOCK_MONOTONIC, &next_wake);

        while(!stop_flag_.load(std::memory_order_relaxed)){
            Sample s;
            auto push_sample = [this](const Sample& sample) {
                // Do not log or allocate here. A retry means the consumer is
                // behind or the ring is full; the counter is exported later.
                while(!stop_flag_.load(std::memory_order_relaxed) && !ring_.try_push(sample)) {
                    push_retries_.fetch_add(1, std::memory_order_relaxed);
                }
            };

            // CPU counters are mutually exclusive between cgroup and simple
            // PID modes. Cgroup mode is used by the multithreaded launcher.
            if(perf_cgroup_reader_.is_open()){
                s.tag = SampleTag::CPU;
                if(perf_cgroup_reader_.read(s.cpu)){
                    push_sample(s);
                }
            } else if(perf_reader_.is_open()){
                s.tag = SampleTag::CPU;
                if(perf_reader_.read(s.cpu)){
                    push_sample(s);
                }
            }

            // RAPL rows are raw snapshots. Deltas and overflow handling are
            // computed in the export/consumer path, not here.
            if(rapl_reader_.is_open()){
                s.tag = SampleTag::ENERGY;
                if(rapl_reader_.read(s.energy)){
                    push_sample(s);
                }
            }

            // NVML is optional and device-level. It is compiled only when the
            // library is built with TELEMETRY_WITH_GPU.
            #ifdef TELEMETRY_WITH_GPU
                if (nvml_reader_.is_open())
                {
                    s.tag = SampleTag::GPU;
                    if(nvml_reader_.read(s.gpu)){
                        push_sample(s);
                    }
                }
            #endif
            
            ring_.flush_producer();

            // Sleep until the next absolute sampling instant.
            next_wake.tv_sec += cfg_.interval_ns / 1'000'000'000L;
            next_wake.tv_nsec += cfg_.interval_ns % 1'000'000'000L;
            if (next_wake.tv_nsec >= 1'000'000'000L) {
                next_wake.tv_sec++;
                next_wake.tv_nsec -= 1'000'000'000L;
            }
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, nullptr);
            }
            ring_.flush_producer();
            running_.store(false);  

    }

    void Collector::stop(){
        stop_flag_.store(true, std::memory_order_relaxed);
        if(thread_started_){
            pthread_join(thread_, nullptr);
            thread_started_ = false;
        }
        close_readers();
        running_.store(false);
    }

    void Collector::close_readers() noexcept {
        perf_reader_.close();
        perf_cgroup_reader_.close();
        rapl_reader_.close();
        nvml_reader_.close();
    }

    void Collector::sleep_ns(long ns) const noexcept {
        if(ns <= 0) return;

        struct timespec t;
        t.tv_sec = ns / 1'000'000'000L;
        t.tv_nsec = ns % 1'000'000'000L;
        nanosleep(&t, nullptr);
    }
} // namespace telemetry
