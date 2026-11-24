#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <span>
#include <type_traits>

#include "qsl/common/platform.hpp"

namespace qsl {

/// Bounded wait-free single-producer / single-consumer ring buffer.
///
/// The usual hand-off between a feed-handler thread and a strategy or
/// matching thread. Each index is written by exactly one thread, so the
/// queue needs only acquire/release ordering (no CAS, no locks):
///   producer: write slot, then tail.store(release)
///   consumer: tail.load(acquire), then read slot
/// Each side also keeps a cached copy of the other side's index and reloads
/// it only when the queue looks full or empty, so in steady state the two
/// cores do not ping-pong cache lines on every operation. Indices grow
/// monotonically and are masked, so full/empty need no extra flag.
template <typename T, std::size_t Capacity>
    requires(Capacity >= 2 && std::has_single_bit(Capacity) && std::is_trivially_copyable_v<T>)
class SpscQueue {
public:
    SpscQueue() = default;
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    /// Producer thread only. Returns false if the queue is full.
    bool try_push(const T& value) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_cache_ == Capacity) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ == Capacity) return false;
        }
        buffer_[tail & kMask] = value;
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    /// Consumer thread only. Returns false if the queue is empty.
    bool try_pop(T& out) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_cache_) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head == tail_cache_) return false;
        }
        out = buffer_[head & kMask];
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    /// Producer thread only. Copies up to items.size() elements and publishes
    /// them with one release store; returns how many were pushed. When the
    /// consumer keeps up, every single push costs a cross-core cache-line
    /// transfer, so batching is how a feed handler reaches peak throughput.
    std::size_t push_bulk(std::span<const T> items) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        std::size_t space = Capacity - (tail - head_cache_);
        if (space < items.size()) {
            head_cache_ = head_.load(std::memory_order_acquire);
            space = Capacity - (tail - head_cache_);
        }
        const std::size_t n = std::min(space, items.size());
        for (std::size_t i = 0; i < n; ++i) buffer_[(tail + i) & kMask] = items[i];
        if (n > 0) tail_.store(tail + n, std::memory_order_release);
        return n;
    }

    /// Consumer thread only. Pops up to out.size() elements with one release
    /// store; returns how many were popped.
    std::size_t pop_bulk(std::span<T> out) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        std::size_t ready = tail_cache_ - head;
        if (ready < out.size()) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            ready = tail_cache_ - head;
        }
        const std::size_t n = std::min(ready, out.size());
        for (std::size_t i = 0; i < n; ++i) out[i] = buffer_[(head + i) & kMask];
        if (n > 0) head_.store(head + n, std::memory_order_release);
        return n;
    }

    /// Approximate when called concurrently; exact when the queue is quiescent.
    [[nodiscard]] std::size_t size() const noexcept {
        return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire);
    }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    // Consumer-owned line: its index plus its cached view of the producer.
    alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};
    std::size_t tail_cache_ = 0;
    // Producer-owned line.
    alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};
    std::size_t head_cache_ = 0;
    alignas(kCacheLineSize) std::array<T, Capacity> buffer_{};
};

}  // namespace qsl
