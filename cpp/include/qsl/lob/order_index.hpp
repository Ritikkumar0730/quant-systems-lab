#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qsl/lob/types.hpp"

namespace qsl::lob {

/// Open-addressing hash map OrderId -> pool slot.
///
/// Linear probing for cache-friendly lookups, backward-shift deletion so no
/// tombstones accumulate and probe lengths never degrade under heavy
/// cancel/replace traffic, and a fixed power-of-two capacity (load factor
/// <= 0.5) so the matching path never allocates or rehashes.
class OrderIndex {
public:
    static constexpr std::uint32_t kNotFound = ~std::uint32_t{0};

    explicit OrderIndex(std::size_t max_entries) : max_entries_(max_entries) {
        std::size_t capacity = 16;
        while (capacity < 2 * max_entries) capacity <<= 1;
        slots_.assign(capacity, Slot{0, kNotFound});
        mask_ = capacity - 1;
    }

    [[nodiscard]] std::uint32_t find(OrderId key) const noexcept {
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            const Slot& slot = slots_[i];
            if (slot.value == kNotFound) return kNotFound;
            if (slot.key == key) return slot.value;
        }
    }

    /// Inserts key -> value. Returns false if the key is present or the map
    /// is at its configured capacity.
    bool insert(OrderId key, std::uint32_t value) noexcept {
        if (size_ >= max_entries_) return false;
        for (std::size_t i = home(key);; i = (i + 1) & mask_) {
            Slot& slot = slots_[i];
            if (slot.value == kNotFound) {
                slot = Slot{key, value};
                ++size_;
                return true;
            }
            if (slot.key == key) return false;
        }
    }

    bool erase(OrderId key) noexcept {
        std::size_t hole = home(key);
        for (;; hole = (hole + 1) & mask_) {
            if (slots_[hole].value == kNotFound) return false;
            if (slots_[hole].key == key) break;
        }
        // Backward shift: walk the rest of the cluster and pull back every
        // entry whose probe path [home, j] passes over the hole.
        for (std::size_t j = (hole + 1) & mask_; slots_[j].value != kNotFound; j = (j + 1) & mask_) {
            const std::size_t probe_len = (j - home(slots_[j].key)) & mask_;
            const std::size_t dist_to_hole = (j - hole) & mask_;
            if (probe_len >= dist_to_hole) {
                slots_[hole] = slots_[j];
                hole = j;
            }
        }
        slots_[hole].value = kNotFound;
        --size_;
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }

private:
    struct Slot {
        OrderId key;
        std::uint32_t value;
    };

    /// SplitMix64 finalizer: full avalanche, so sequential exchange order
    /// ids do not form long clusters.
    static constexpr std::uint64_t mix(std::uint64_t x) noexcept {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    [[nodiscard]] std::size_t home(OrderId key) const noexcept {
        return static_cast<std::size_t>(mix(key)) & mask_;
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t max_entries_;
};

}  // namespace qsl::lob
