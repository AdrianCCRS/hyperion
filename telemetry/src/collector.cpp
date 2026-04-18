#include "telemetry/collector.hpp"
#include <sched.h>
#include <time.h>
#include <stdexcept>
#include <cstring>
#include <utility>

namespace telemetry
{
    Collector::Collector(CollectorConfig cfg, Ring& ring)
        : cfg_(std::move(cfg)),
          ring_(ring),
          perf_reader_(cfg_.target_pid, -1),
          rapl_reader_(cfg_.rapl_pkg_path, cfg_.rapl_dram_path),
          nvml_reader_(0) {}

    Collector::~Collector(){
        stop();
    }

    void Collector::start(){
        if(running_.load()) return;

        perf_reader_.open();
        if(!cfg_.rapl_pkg_path.empty()) rapl_reader_.open();
        if(cfg_.enable_gpu) nvml_reader_.open();

        stop_flag_.store(false);

        pthread_attr_t attr;
        pthread_attr_init(&attr);

        // Pin producer to a specific core if requested.
        // This limits TLB disruption and keeps cache state stable —
        // see LKML Chapter 35 (Linux Programming Interface, ch.29/35)
        if (cfg_.producer_cpu >= 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cfg_.producer_cpu, &cpuset);
            pthread_attr_setaffinity_np(&attr, sizeof(cpuset), &cpuset);
        }

        if(pthread_create(&thread_, &attr, thread_entry, this) != 0){
            pthread_attr_destroy(&attr);
            running_.store(false);
            throw std::runtime_error("Failed to create producer thread");
        }
        pthread_attr_destroy(&attr);
        running_.store(true);
    }

    void* Collector::thread_entry(void* arg){
        static_cast<Collector*>(arg)->run();
        return nullptr;
    }
    
    void Collector::run(){
        struct timespec next_wake;
        clock_gettime(CLOCK_MONOTONIC, &next_wake);

        while(!stop_flag_.load(std::memory_order_relaxed)){
            Sample s;
            // --- CPU Sample ---
            s.tag = SampleTag::CPU;
            if(perf_reader_.read(s.cpu)){
                while(!ring_.try_push(s)){}
            }

            // --- Energy Sample ---
            if(rapl_reader_.is_open()){
                s.tag = SampleTag::ENERGY;
                if(rapl_reader_.read(s.energy)){
                    while(!ring_.try_push(s)){}
                }
            }

            // --- GPU Sample ---
            #ifdef TELEMETRY_WITH_GPU
                if (nvml_reader_.is_open())
                {
                    s.tag = SampleTag::GPU;
                    if(nvml_reader_.read(s.gpu)){
                        while(!ring_.try_push(s)){}
                    }
                }
            #endif
            
            ring_.flush_producer();

            // --- Sleep until next interval (absolute timer) ---
            next_wake.tv_nsec += cfg_.interval_ns;
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
        if(!running_.load()) return;

        stop_flag_.store(true, std::memory_order_relaxed);
        pthread_join(thread_, nullptr);
        perf_reader_.disable();
        rapl_reader_.close();
        nvml_reader_.close();
        running_.store(false);
    }

    void Collector::sleep_ns(long ns) const noexcept {
        if(ns <= 0) return;

        struct timespec t;
        t.tv_sec = ns / 1'000'000'000L;
        t.tv_nsec = ns % 1'000'000'000L;
        nanosleep(&t, nullptr);
    }
} // namespace telemetry
