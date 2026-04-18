#pragma once
#include <atomic>
#include <array>
#include <optional>
#include <cstddef>
#include <cassert>

/** 
 * @file spsc_ring.hpp
 * @brief Lock-free Single-Producer Single-Consumer (SPSC) ring buffer for telemetry.
 *
 * This structure connects the acquisition thread (producer) with the processing 
 * thread (consumer) without using mutexes, keeping the critical path wait-free.
 */

namespace telemetry {
    
    // Hardware cache line size (bytes). Validated at runtime in tests.
    static constexpr size_t CACHE_LINE = 64;

    /**
     * @brief Single-producer single-consumer lock-free ring buffer.
     *
     * @tparam T Element type. Must be trivially copyable (POD) to ensure safe memory operations.
     * @tparam N Capacity of the ring buffer. Must be a power of two to allow rapid bitwise modulo.
     *
     * @details
     * This implementation relies on principles from high-performance networking and HPC:
     * 1. **FastForward** (Giacomoni et al., PPoPP'08): Temporal slipping and cache-line separation.
     * 2. **MCRingBuffer** (Lee et al., ANCS'09): Batch updates to reduce coherence traffic.
     *
     * ### Memory Layout & False Sharing Avoidance
     * To prevent false sharing (where cores invalidate each other's cache lines), control 
     * variables are grouped by access pattern and aligned to distinct hardware cache lines:
     * `[ pad | shared state | pad | consumer-local | pad | producer-local | pad | data[] ]`
     *
     * ### Thread Safety Constraints
     * - ONLY the Producer thread may call `try_push()` and `flush_producer()`.
     * - ONLY the Consumer thread may call `try_pop()` and `flush_consumer()`.
     */
    template <typename T, size_t N>
    class SPSCRing {
        static_assert((N & (N - 1)) == 0, "Capacity N must be a power of two");
        static_assert(N >= 16, "N too small for temporal slipping");

        // ---- Shared state (written by producer, read by both) ----
        //Use of alignas to ensure each variable is on a separate cache line, preventing false sharing.
        //alignas set the variable on a multiple of the cache line size, ensuring that no two variables share the same cache line.
        //padding ensures that the next variable starts on a new cache line, even if the previous variable is smaller than the cache line size.
        alignas(CACHE_LINE) std::atomic<size_t> write_{0};
        char _pad0[CACHE_LINE - sizeof(std::atomic<size_t>)];

        // ---- Consumer-local state (written ONLY by consumer) ----
        alignas(CACHE_LINE) size_t next_read_{0};
        size_t local_write_{0};
        size_t r_batch_{0};
        char _pad1[CACHE_LINE - 3*sizeof(size_t)]; // *3 because we have 3 size_t variables in this section

        // ---- Producer-local state (written ONLY by producer) ----
        alignas(CACHE_LINE) size_t next_write_{0};
        size_t local_read_{0};
        size_t w_batch_{0};
        char _pad2[CACHE_LINE - 3*sizeof(size_t)];

        // ---- Shared read pointer ----
        alignas(CACHE_LINE) std::atomic<size_t> read_{0};
        char _pad3[CACHE_LINE - sizeof(std::atomic<size_t>)];

        // --- Ring buffer ----
        alignas(CACHE_LINE) std::array<T, N> buf_;

        //Batch-update thresholds (MCRingBuffer §2 - batch updates)
        static constexpr size_t BATCH = (CACHE_LINE / sizeof(T)) > 1 ? (CACHE_LINE / sizeof(T)) : 1;

        public:

        /**
         * @brief Attempt to append an item to the ring buffer (non-blocking).
         * @note MUST ONLY be called by the PRODUCER thread.
         * @param item The telemetry sample to push.
         * @return true if the sample was successfully appended, false if the buffer is full.
         */
        bool try_push(const T& item) noexcept {
            const size_t after = (next_write_ + 1) & (N - 1); // Wrap around using bitwise AND (N is power of two), like the modulo operation but faster.
            if(after == local_read_){
                local_read_ = read_.load(std::memory_order_acquire);
                if(after == local_read_) return false; // Ring is full
            }
            buf_[next_write_] = item;
            next_write_ = after;
            if(++w_batch_ >= BATCH){
                write_.store(next_write_, std::memory_order_release);
                w_batch_ = 0;
            }
            return true;
        }
    
        /**
         * @brief Synchronize batched write pointers with the shared state.
         * @details This avoids the overhead of atomic writes per element. Call this 
         * method at the end of each sample burst to make elements visible to the Consumer.
         * @note MUST ONLY be called by the PRODUCER thread.
         */
        void flush_producer() noexcept {
            write_.store(next_write_, std::memory_order_release);
            w_batch_ = 0;
        }

        /**
         * @brief Attempt to extract the next item from the ring buffer (non-blocking).
         * @note MUST ONLY be called by the CONSUMER thread.
         * @return std::optional containing the element if available, std::nullopt if empty.
         */
        std::optional<T> try_pop() noexcept {
            if(next_read_ == local_write_){
                local_write_ = write_.load(std::memory_order_acquire);
                if(next_read_ == local_write_) return std::nullopt; // Ring is empty
            }
            T item = buf_[next_read_];
            next_read_ = (next_read_ + 1) & (N - 1);
            if(++r_batch_ >= BATCH){
                read_.store(next_read_, std::memory_order_release);
                r_batch_ = 0;
            }
            return item;
        }

        /**
         * @brief Synchronize batched read pointers with the shared state.
         * @details This informs the Producer that new buffer slots are free to use. Call 
         * this method at the end of each sample burst logic loop.
         * @note MUST ONLY be called by the CONSUMER thread.
         */
        void flush_consumer() noexcept {
            read_.store(next_read_, std::memory_order_release);
            r_batch_ = 0;
        }

        /**
         * @return The maximum capacity of the ring buffer.
         */
        size_t capacity() const noexcept { return N; }

    };

}