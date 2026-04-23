#include "telemetry/spsc_ring.hpp"

#include <cstdint>
#include <thread>
#include <cstdio>

int main() {
    telemetry::SPSCRing<int, 64> ring;
    constexpr int N = 100'000;
    uint64_t sum_produced = 0;
    uint64_t sum_consumed = 0;

    std::thread producer([&]{
        for (int i = 0; i < N; ++i) {
            while (!ring.try_push(i)) {}
            sum_produced += static_cast<uint64_t>(i);
        }
        ring.flush_producer();
    });

    int received = 0;
    while (received < N) {
        auto v = ring.try_pop();
        if (v) {
            if (*v != received) return 1;
            sum_consumed += static_cast<uint64_t>(*v);
            ++received;
        }
    }

    producer.join();
    if (sum_produced != sum_consumed) return 1;

    std::printf("SPSC ring: PASS (N=%d, sum=%llu)\n",
                N,
                static_cast<unsigned long long>(sum_consumed));
    return 0;
}
