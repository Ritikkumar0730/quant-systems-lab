#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qsl/lob/hierarchical_bitset.hpp"
#include "qsl/lob/order_index.hpp"
#include "qsl/lob/types.hpp"

namespace qsl::lob {

/// Single-instrument limit order book and matching engine with price-time
/// priority.
///
/// Design (all memory is reserved in the constructor; the hot path never
/// allocates):
///   * Price ladder: a flat array with one Level per tick. A tick holds bids
///     or asks, never both, because the book is never crossed.
///   * Each level is an intrusive doubly-linked FIFO of orders stored in a
///     preallocated pool and linked by 32-bit indices, so an order is 32 bytes
///     and cancel is O(1).
///   * Two hierarchical bitsets (bids, asks) mark non-empty levels; when the
///     best level empties, the next one is found in O(log64 ticks).
///   * OrderIndex maps exchange order ids to pool slots with open addressing.
///
/// Complexity: add O(1) + O(fills); cancel/reduce O(1); best bid/ask O(1).
template <TradeListener Listener = NullListener>
class OrderBook {
public:
    struct OrderInfo {
        Side side;
        Price price;
        Qty qty;
    };

    explicit OrderBook(const BookConfig& config, Listener listener = Listener{})
        : config_(validate(config)),
          ticks_(static_cast<std::size_t>(config.max_price - config.min_price) + 1),
          levels_(ticks_),
          bids_(ticks_),
          asks_(ticks_),
          pool_(config.max_orders),
          index_(config.max_orders),
          listener_(std::move(listener)) {
        for (std::uint32_t i = 0; i < config.max_orders; ++i) {
            pool_[i].next = i + 1 < config.max_orders ? i + 1 : kNil;
        }
        free_head_ = 0;
    }

    // ------------------------------------------------------------ order entry

    /// Limit order. Matches against the opposite side up to `price`, then
    /// handles any remainder according to `tif`. Rejections have no effect.
    ExecResult add(OrderId id, Side side, Price price, Qty qty,
                   TimeInForce tif = TimeInForce::GoodTillCancel) {
        if (qty == 0) [[unlikely]] return {Reject::InvalidQuantity};
        if (price < config_.min_price || price > config_.max_price) [[unlikely]]
            return {Reject::InvalidPrice};
        if (index_.find(id) != OrderIndex::kNotFound) [[unlikely]] return {Reject::DuplicateOrderId};

        const std::size_t tick = to_tick(price);
        switch (tif) {
            case TimeInForce::GoodTillCancel:
                // Conservative but atomic: a GTC order needs a free slot in case
                // it rests, so a full book rejects before any execution.
                if (free_head_ == kNil) [[unlikely]] return {Reject::BookFull};
                break;
            case TimeInForce::PostOnly:
                if (crosses(side, tick)) return {Reject::PostOnlyWouldCross};
                if (free_head_ == kNil) [[unlikely]] return {Reject::BookFull};
                break;
            case TimeInForce::FillOrKill:
                if (available(side, tick, qty) < qty) return {Reject::FillOrKillUnfilled};
                break;
            case TimeInForce::ImmediateOrCancel:
                break;
        }

        const Qty filled = match(id, side, tick, qty);
        const Qty left = qty - filled;
        if (left > 0 && (tif == TimeInForce::GoodTillCancel || tif == TimeInForce::PostOnly)) {
            rest(id, side, tick, left);
            return {Reject::None, filled, left};
        }
        return {Reject::None, filled, 0};
    }

    /// Market order: sweeps the opposite side, never rests.
    ExecResult market(OrderId id, Side side, Qty qty) {
        if (qty == 0) [[unlikely]] return {Reject::InvalidQuantity};
        if (index_.find(id) != OrderIndex::kNotFound) [[unlikely]] return {Reject::DuplicateOrderId};
        const std::size_t limit = side == Side::Buy ? ticks_ - 1 : 0;
        return {Reject::None, match(id, side, limit, qty), 0};
    }

    Reject cancel(OrderId id) {
        const std::uint32_t slot = index_.find(id);
        if (slot == OrderIndex::kNotFound) return Reject::UnknownOrder;
        unlink(slot);
        index_.erase(id);
        release(slot);
        return Reject::None;
    }

    /// Shrinks a resting order in place, keeping its queue position (the
    /// exchange-standard rule: only size decreases preserve priority).
    Reject reduce(OrderId id, Qty new_qty) {
        const std::uint32_t slot = index_.find(id);
        if (slot == OrderIndex::kNotFound) return Reject::UnknownOrder;
        if (new_qty == 0) return cancel(id);
        Order& order = pool_[slot];
        if (new_qty >= order.qty) return Reject::InvalidQuantity;
        levels_[order.tick].qty -= order.qty - new_qty;
        order.qty = new_qty;
        return Reject::None;
    }

    /// Cancel/replace. Same price with a smaller or equal size keeps priority;
    /// anything else re-enters at the back of the queue and may trade.
    ExecResult replace(OrderId id, Price new_price, Qty new_qty) {
        const std::uint32_t slot = index_.find(id);
        if (slot == OrderIndex::kNotFound) return {Reject::UnknownOrder};
        if (new_qty == 0) return {Reject::InvalidQuantity};
        if (new_price < config_.min_price || new_price > config_.max_price)
            return {Reject::InvalidPrice};
        Order& order = pool_[slot];
        if (to_tick(new_price) == order.tick && new_qty <= order.qty) {
            levels_[order.tick].qty -= order.qty - new_qty;
            order.qty = new_qty;
            return {Reject::None, 0, new_qty};
        }
        const Side side = order.side;
        cancel(id);
        return add(id, side, new_price, new_qty, TimeInForce::GoodTillCancel);
    }

    // ---------------------------------------------------------------- queries

    [[nodiscard]] std::optional<Price> best_bid() const noexcept {
        if (best_bid_ == kNone) return std::nullopt;
        return to_price(best_bid_);
    }
    [[nodiscard]] std::optional<Price> best_ask() const noexcept {
        if (best_ask_ == kNone) return std::nullopt;
        return to_price(best_ask_);
    }

    /// Resting quantity at a price on one side (0 if none).
    [[nodiscard]] Qty volume_at(Side side, Price price) const noexcept {
        if (price < config_.min_price || price > config_.max_price) return 0;
        const std::size_t tick = to_tick(price);
        return side_bits(side).test(tick) ? levels_[tick].qty : 0;
    }

    /// Market-by-price snapshot, best level first. Returns levels written.
    std::size_t depth(Side side, std::span<LevelInfo> out) const noexcept {
        std::size_t n = 0;
        for (std::size_t tick = best(side); tick != kNone && n < out.size(); tick = next_worse(side, tick)) {
            const Level& level = levels_[tick];
            out[n++] = LevelInfo{to_price(tick), level.qty, level.count};
        }
        return n;
    }

    [[nodiscard]] std::optional<OrderInfo> find(OrderId id) const noexcept {
        const std::uint32_t slot = index_.find(id);
        if (slot == OrderIndex::kNotFound) return std::nullopt;
        const Order& order = pool_[slot];
        return OrderInfo{order.side, to_price(order.tick), order.qty};
    }

    [[nodiscard]] std::size_t order_count() const noexcept { return index_.size(); }
    [[nodiscard]] const BookConfig& config() const noexcept { return config_; }
    [[nodiscard]] Listener& listener() noexcept { return listener_; }

    /// Full O(ticks + orders) consistency check of every internal structure.
    /// Used by the tests after randomized operation sequences.
    [[nodiscard]] bool check_invariants() const {
        std::size_t live = 0;
        for (std::size_t tick = 0; tick < ticks_; ++tick) {
            const Level& level = levels_[tick];
            const bool is_bid = bids_.test(tick);
            const bool is_ask = asks_.test(tick);
            if (level.count == 0) {
                if (is_bid || is_ask || level.head != kNil || level.tail != kNil || level.qty != 0)
                    return false;
                continue;
            }
            if (is_bid == is_ask) return false;  // exactly one side owns a populated tick
            const Side side = is_bid ? Side::Buy : Side::Sell;
            Qty qty = 0;
            std::uint32_t count = 0;
            std::uint32_t prev = kNil;
            for (std::uint32_t slot = level.head; slot != kNil; slot = pool_[slot].next) {
                const Order& order = pool_[slot];
                if (order.prev != prev || order.tick != tick || order.side != side || order.qty == 0)
                    return false;
                if (index_.find(order.id) != slot) return false;
                qty += order.qty;
                ++count;
                prev = slot;
            }
            if (prev != level.tail || qty != level.qty || count != level.count) return false;
            live += count;
        }
        std::size_t free_slots = 0;
        for (std::uint32_t slot = free_head_; slot != kNil; slot = pool_[slot].next) ++free_slots;
        if (live != index_.size() || live + free_slots != pool_.size()) return false;
        if (best_bid_ != bids_.find_prev(ticks_ - 1) || best_ask_ != asks_.find_next(0)) return false;
        return best_bid_ == kNone || best_ask_ == kNone || best_bid_ < best_ask_;  // never crossed
    }

private:
    static constexpr std::uint32_t kNil = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::size_t kNone = HierarchicalBitset::npos;

    struct Order {
        OrderId id = 0;
        Qty qty = 0;
        std::uint32_t tick = 0;
        std::uint32_t prev = kNil;
        std::uint32_t next = kNil;
        Side side = Side::Buy;
    };
    static_assert(sizeof(Order) == 32, "two orders per 64-byte cache line");

    struct Level {
        Qty qty = 0;
        std::uint32_t head = kNil;
        std::uint32_t tail = kNil;
        std::uint32_t count = 0;
    };

    static BookConfig validate(const BookConfig& c) {
        if (c.max_price < c.min_price) throw std::invalid_argument("max_price < min_price");
        if (c.max_price - c.min_price >= (Price{1} << 26))
            throw std::invalid_argument("tick ladder wider than 2^26 levels");
        if (c.max_orders == 0 || c.max_orders == kNil)
            throw std::invalid_argument("max_orders must be in [1, 2^32 - 2]");
        return c;
    }

    [[nodiscard]] std::size_t to_tick(Price price) const noexcept {
        return static_cast<std::size_t>(price - config_.min_price);
    }
    [[nodiscard]] Price to_price(std::size_t tick) const noexcept {
        return config_.min_price + static_cast<Price>(tick);
    }

    [[nodiscard]] const HierarchicalBitset& side_bits(Side side) const noexcept {
        return side == Side::Buy ? bids_ : asks_;
    }
    [[nodiscard]] std::size_t best(Side side) const noexcept {
        return side == Side::Buy ? best_bid_ : best_ask_;
    }
    /// Next level away from the touch on `side` (lower bids, higher asks).
    [[nodiscard]] std::size_t next_worse(Side side, std::size_t tick) const noexcept {
        if (side == Side::Buy) return tick == 0 ? kNone : bids_.find_prev(tick - 1);
        return asks_.find_next(tick + 1);
    }

    /// Would an order on `side` at `tick` trade on arrival?
    [[nodiscard]] bool crosses(Side side, std::size_t tick) const noexcept {
        if (side == Side::Buy) return best_ask_ != kNone && tick >= best_ask_;
        return best_bid_ != kNone && tick <= best_bid_;
    }

    /// Opposite-side quantity executable at `limit` or better, capped once
    /// `needed` is reached (fill-or-kill pre-check).
    [[nodiscard]] Qty available(Side side, std::size_t limit, Qty needed) const noexcept {
        Qty sum = 0;
        const Side passive = opposite(side);
        for (std::size_t tick = best(passive); tick != kNone; tick = next_worse(passive, tick)) {
            if (side == Side::Buy ? tick > limit : tick < limit) break;
            sum += levels_[tick].qty;
            if (sum >= needed) break;
        }
        return sum;
    }

    Qty match(OrderId taker, Side side, std::size_t limit, Qty qty) {
        Qty remaining = qty;
        if (side == Side::Buy) {
            while (remaining > 0 && best_ask_ != kNone && best_ask_ <= limit) {
                remaining = consume(taker, side, best_ask_, remaining);
                if (levels_[best_ask_].count == 0) {
                    asks_.reset(best_ask_);
                    best_ask_ = asks_.find_next(best_ask_ + 1);
                }
            }
        } else {
            while (remaining > 0 && best_bid_ != kNone && best_bid_ >= limit) {
                remaining = consume(taker, side, best_bid_, remaining);
                if (levels_[best_bid_].count == 0) {
                    bids_.reset(best_bid_);
                    best_bid_ = best_bid_ == 0 ? kNone : bids_.find_prev(best_bid_ - 1);
                }
            }
        }
        return qty - remaining;
    }

    /// Fills against one level in FIFO order; returns the taker's remainder.
    Qty consume(OrderId taker, Side taker_side, std::size_t tick, Qty remaining) {
        Level& level = levels_[tick];
        const Price price = to_price(tick);
        while (remaining > 0 && level.head != kNil) {
            const std::uint32_t slot = level.head;
            Order& maker = pool_[slot];
            const Qty fill = std::min(remaining, maker.qty);
            maker.qty -= fill;
            level.qty -= fill;
            remaining -= fill;
            listener_.on_trade(Trade{taker, maker.id, price, fill, taker_side});
            if (maker.qty == 0) {
                level.head = maker.next;
                if (level.head != kNil) pool_[level.head].prev = kNil;
                else level.tail = kNil;
                --level.count;
                index_.erase(maker.id);
                release(slot);
            }
        }
        return remaining;
    }

    void rest(OrderId id, Side side, std::size_t tick, Qty qty) {
        const std::uint32_t slot = acquire();  // availability checked before matching
        Level& level = levels_[tick];
        pool_[slot] = Order{id, qty, static_cast<std::uint32_t>(tick), level.tail, kNil, side};
        if (level.tail != kNil) pool_[level.tail].next = slot;
        else level.head = slot;
        level.tail = slot;
        ++level.count;
        level.qty += qty;
        index_.insert(id, slot);
        if (side == Side::Buy) {
            if (level.count == 1) bids_.set(tick);
            if (best_bid_ == kNone || tick > best_bid_) best_bid_ = tick;
        } else {
            if (level.count == 1) asks_.set(tick);
            if (best_ask_ == kNone || tick < best_ask_) best_ask_ = tick;
        }
    }

    void unlink(std::uint32_t slot) {
        const Order& order = pool_[slot];
        Level& level = levels_[order.tick];
        if (order.prev != kNil) pool_[order.prev].next = order.next;
        else level.head = order.next;
        if (order.next != kNil) pool_[order.next].prev = order.prev;
        else level.tail = order.prev;
        level.qty -= order.qty;
        if (--level.count > 0) return;
        const std::size_t tick = order.tick;
        if (order.side == Side::Buy) {
            bids_.reset(tick);
            if (tick == best_bid_) best_bid_ = tick == 0 ? kNone : bids_.find_prev(tick - 1);
        } else {
            asks_.reset(tick);
            if (tick == best_ask_) best_ask_ = asks_.find_next(tick + 1);
        }
    }

    std::uint32_t acquire() noexcept {
        const std::uint32_t slot = free_head_;
        free_head_ = pool_[slot].next;
        return slot;
    }
    void release(std::uint32_t slot) noexcept {
        pool_[slot].next = free_head_;
        free_head_ = slot;
    }

    BookConfig config_;
    std::size_t ticks_;
    std::vector<Level> levels_;
    HierarchicalBitset bids_;
    HierarchicalBitset asks_;
    std::vector<Order> pool_;
    OrderIndex index_;
    std::uint32_t free_head_ = kNil;
    std::size_t best_bid_ = kNone;
    std::size_t best_ask_ = kNone;
    Listener listener_;
};

}  // namespace qsl::lob
