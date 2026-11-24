#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace qsl::lob {

/// Fixed-size bitset over [0, n) with O(log64 n) find_next / find_prev.
///
/// Level 0 holds the bits; every higher level keeps one summary bit per
/// 64-bit word of the level below (set iff that word is non-zero). For a
/// 2^18-tick price ladder that is three levels, so finding the next non-empty
/// price level when the best level empties costs a handful of word operations
/// instead of a linear scan over empty ticks.
class HierarchicalBitset {
public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    explicit HierarchicalBitset(std::size_t n) : size_(n) {
        std::size_t bits = n == 0 ? 1 : n;
        do {
            const std::size_t words = (bits + 63) / 64;
            levels_.emplace_back(words, std::uint64_t{0});
            bits = words;
        } while (bits > 1);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool none() const noexcept { return levels_.back()[0] == 0; }

    [[nodiscard]] bool test(std::size_t i) const noexcept {
        return ((levels_[0][i >> 6] >> (i & 63)) & 1u) != 0;
    }

    void set(std::size_t i) noexcept {
        for (auto& level : levels_) {
            std::uint64_t& word = level[i >> 6];
            const bool was_zero = word == 0;
            word |= bit(i);
            if (!was_zero) return;  // parent summary bit is already set
            i >>= 6;
        }
    }

    void reset(std::size_t i) noexcept {
        for (auto& level : levels_) {
            std::uint64_t& word = level[i >> 6];
            word &= ~bit(i);
            if (word != 0) return;  // word still populated: parents unchanged
            i >>= 6;
        }
    }

    /// Smallest set index >= i, or npos.
    [[nodiscard]] std::size_t find_next(std::size_t i) const noexcept {
        if (i >= size_) return npos;
        std::size_t level = 0;
        std::size_t idx = i;
        for (;;) {  // climb until some word has a set bit at or after idx
            const auto& words = levels_[level];
            const std::size_t w = idx >> 6;
            if (w >= words.size()) return npos;
            const std::uint64_t masked = words[w] & (~std::uint64_t{0} << (idx & 63));
            if (masked != 0) {
                idx = (w << 6) | static_cast<std::size_t>(std::countr_zero(masked));
                break;
            }
            if (++level == levels_.size()) return npos;
            idx = w + 1;  // resume at the next word, addressed from the parent level
        }
        while (level-- > 0) {  // descend along the lowest set bits
            idx = (idx << 6) | static_cast<std::size_t>(std::countr_zero(levels_[level][idx]));
        }
        return idx;
    }

    /// Largest set index <= i (i is clamped to size() - 1), or npos.
    [[nodiscard]] std::size_t find_prev(std::size_t i) const noexcept {
        if (size_ == 0) return npos;
        if (i >= size_) i = size_ - 1;
        std::size_t level = 0;
        std::size_t idx = i;
        for (;;) {  // climb until some word has a set bit at or before idx
            const auto& words = levels_[level];
            const std::size_t w = idx >> 6;
            const std::size_t b = idx & 63;
            const std::uint64_t keep = b == 63 ? ~std::uint64_t{0} : (std::uint64_t{1} << (b + 1)) - 1;
            const std::uint64_t masked = words[w] & keep;
            if (masked != 0) {
                idx = (w << 6) | highest_bit(masked);
                break;
            }
            if (w == 0 || ++level == levels_.size()) return npos;
            idx = w - 1;
        }
        while (level-- > 0) {  // descend along the highest set bits
            idx = (idx << 6) | highest_bit(levels_[level][idx]);
        }
        return idx;
    }

private:
    static constexpr std::uint64_t bit(std::size_t i) noexcept { return std::uint64_t{1} << (i & 63); }
    static constexpr std::size_t highest_bit(std::uint64_t word) noexcept {
        return 63 - static_cast<std::size_t>(std::countl_zero(word));
    }

    std::size_t size_;
    std::vector<std::vector<std::uint64_t>> levels_;
};

}  // namespace qsl::lob
