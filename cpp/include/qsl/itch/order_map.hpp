#pragma once
// Live-order store for ITCH replay: open addressing with the order data
// stored inline in the slot, indexed by the order reference itself.
//
// Why identity indexing instead of a mixing hash: NASDAQ assigns order
// reference numbers in (mostly) increasing order, and most book updates hit
// young orders (on 30 Dec 2019 the median updated order was added only ~430
// orders earlier). With `slot = ref & mask`, recent orders sit in a small,
// contiguous, cache-resident region, so a lookup is usually a cache hit. A
// mixing hash would scatter them and turn every lookup into a DRAM miss.
// The price of identity indexing: references arrive in dense runs, so a run
// of new orders fills a contiguous stretch of slots, and at a load factor
// near 1/2 those stretches merge with survivors into long probe clusters.
// Measured on the first 1 GB of the 30 Dec 2019 file: 143 ns/msg with the
// table ~45% full versus 50 ns/msg at ~20%. The table therefore doubles
// whenever it would pass 1/4 full (about 200 MB at that day's peak of
// 1.9 million live orders).
// Correctness does not depend on the key pattern: collisions are resolved by
// linear probing and deletion uses backward shift (no tombstones).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qsl/itch/messages.hpp"

namespace qsl::itch {

struct OrderEntry {
    OrderRef ref;
    Price4 price;
    std::uint32_t shares;
    Locate locate;
    bool is_bid;
    bool used;
};

class OrderMap {
public:
    explicit OrderMap(std::size_t expected_live = std::size_t{1} << 20) {
        std::size_t capacity = 16;
        while (capacity < kLoadDivisor * expected_live) capacity <<= 1;
        slots_.assign(capacity, OrderEntry{});
        mask_ = capacity - 1;
    }

    [[nodiscard]] OrderEntry* find(OrderRef ref) noexcept {
        for (std::size_t i = home(ref);; i = (i + 1) & mask_) {
            OrderEntry& e = slots_[i];
            if (!e.used) return nullptr;
            if (e.ref == ref) return &e;
        }
    }

    /// Inserts a new entry for `ref` and returns it, or nullptr if `ref` is
    /// already present. The returned pointer is valid until the next insert.
    OrderEntry* insert(OrderRef ref) {
        if (kLoadDivisor * (size_ + 1) > slots_.size()) grow();
        for (std::size_t i = home(ref);; i = (i + 1) & mask_) {
            OrderEntry& e = slots_[i];
            if (!e.used) {
                e = OrderEntry{ref, 0, 0, 0, false, true};
                ++size_;
                return &e;
            }
            if (e.ref == ref) return nullptr;
        }
    }

    /// Removes an entry previously returned by find() or insert().
    void erase(OrderEntry* entry) noexcept {
        std::size_t hole = static_cast<std::size_t>(entry - slots_.data());
        // Backward shift: pull back every later entry in the cluster whose
        // probe path passes over the hole.
        for (std::size_t j = (hole + 1) & mask_; slots_[j].used; j = (j + 1) & mask_) {
            const std::size_t probe_len = (j - home(slots_[j].ref)) & mask_;
            const std::size_t dist_to_hole = (j - hole) & mask_;
            if (probe_len >= dist_to_hole) {
                slots_[hole] = slots_[j];
                hole = j;
            }
        }
        slots_[hole].used = false;
        --size_;
    }

    /// Starts loading the slot `ref` would occupy (a hint; no effect on state).
    void prefetch(OrderRef ref) const noexcept { __builtin_prefetch(&slots_[home(ref)]); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t grows() const noexcept { return grows_; }

private:
    static constexpr std::size_t kLoadDivisor = 4;  ///< keep the table at most 1/4 full

    [[nodiscard]] std::size_t home(OrderRef ref) const noexcept { return static_cast<std::size_t>(ref) & mask_; }

    void grow() {
        std::vector<OrderEntry> old(slots_.size() * 2, OrderEntry{});
        old.swap(slots_);
        mask_ = slots_.size() - 1;
        ++grows_;
        for (const OrderEntry& e : old) {
            if (!e.used) continue;
            std::size_t i = home(e.ref);
            while (slots_[i].used) i = (i + 1) & mask_;
            slots_[i] = e;
        }
    }

    std::vector<OrderEntry> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t grows_ = 0;
};

}  // namespace qsl::itch
