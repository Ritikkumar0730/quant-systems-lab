#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "qsl/lob/types.hpp"

namespace qsl::lob {

/// Deliberately naive order book: std::map price levels, std::list queues,
/// linear-time cancels. It exists only as a test oracle and benchmark
/// baseline. It implements exactly the same contract as OrderBook (including
/// rejection precedence) in the most obvious way possible, so that a
/// randomized differential test can compare the two operation by operation.
class ReferenceBook {
public:
    explicit ReferenceBook(const BookConfig& config) : config_(config) {}

    ExecResult add(OrderId id, Side side, Price price, Qty qty,
                   TimeInForce tif = TimeInForce::GoodTillCancel) {
        if (qty == 0) return {Reject::InvalidQuantity};
        if (price < config_.min_price || price > config_.max_price) return {Reject::InvalidPrice};
        if (live_.contains(id)) return {Reject::DuplicateOrderId};
        const bool full = live_.size() >= config_.max_orders;
        if (tif == TimeInForce::GoodTillCancel && full) return {Reject::BookFull};
        if (tif == TimeInForce::PostOnly) {
            if (would_cross(side, price)) return {Reject::PostOnlyWouldCross};
            if (full) return {Reject::BookFull};
        }
        if (tif == TimeInForce::FillOrKill && executable(side, price) < qty)
            return {Reject::FillOrKillUnfilled};

        const Qty filled = match(id, side, price, qty);
        const Qty left = qty - filled;
        if (left > 0 && (tif == TimeInForce::GoodTillCancel || tif == TimeInForce::PostOnly)) {
            book(side)[price].push_back({id, left});
            live_[id] = {side, price};
            return {Reject::None, filled, left};
        }
        return {Reject::None, filled, 0};
    }

    ExecResult market(OrderId id, Side side, Qty qty) {
        if (qty == 0) return {Reject::InvalidQuantity};
        if (live_.contains(id)) return {Reject::DuplicateOrderId};
        const Price limit = side == Side::Buy ? config_.max_price : config_.min_price;
        return {Reject::None, match(id, side, limit, qty), 0};
    }

    Reject cancel(OrderId id) {
        const auto it = live_.find(id);
        if (it == live_.end()) return Reject::UnknownOrder;
        auto& levels = book(it->second.side);
        auto& queue = levels[it->second.price];
        queue.erase(std::find_if(queue.begin(), queue.end(), [id](const Resting& r) { return r.id == id; }));
        if (queue.empty()) levels.erase(it->second.price);
        live_.erase(it);
        return Reject::None;
    }

    Reject reduce(OrderId id, Qty new_qty) {
        Resting* order = find_resting(id);
        if (order == nullptr) return Reject::UnknownOrder;
        if (new_qty == 0) return cancel(id);
        if (new_qty >= order->qty) return Reject::InvalidQuantity;
        order->qty = new_qty;
        return Reject::None;
    }

    ExecResult replace(OrderId id, Price new_price, Qty new_qty) {
        Resting* order = find_resting(id);
        if (order == nullptr) return {Reject::UnknownOrder};
        if (new_qty == 0) return {Reject::InvalidQuantity};
        if (new_price < config_.min_price || new_price > config_.max_price) return {Reject::InvalidPrice};
        const auto [side, price] = live_.at(id);
        if (new_price == price && new_qty <= order->qty) {
            order->qty = new_qty;
            return {Reject::None, 0, new_qty};
        }
        cancel(id);
        return add(id, side, new_price, new_qty, TimeInForce::GoodTillCancel);
    }

    [[nodiscard]] std::optional<Price> best_bid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.rbegin()->first;
    }
    [[nodiscard]] std::optional<Price> best_ask() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    [[nodiscard]] Qty volume_at(Side side, Price price) const {
        const auto& levels = side == Side::Buy ? bids_ : asks_;
        const auto it = levels.find(price);
        return it == levels.end() ? 0 : total(it->second);
    }

    /// Full depth, best level first.
    [[nodiscard]] std::vector<LevelInfo> levels(Side side) const {
        std::vector<LevelInfo> out;
        auto emit = [&out](const auto& kv) {
            out.push_back({kv.first, total(kv.second), static_cast<std::uint32_t>(kv.second.size())});
        };
        if (side == Side::Buy) std::for_each(bids_.rbegin(), bids_.rend(), emit);
        else std::for_each(asks_.begin(), asks_.end(), emit);
        return out;
    }

    [[nodiscard]] std::size_t order_count() const { return live_.size(); }
    [[nodiscard]] const std::vector<Trade>& trades() const { return trades_; }
    void clear_trades() { trades_.clear(); }

private:
    struct Resting {
        OrderId id;
        Qty qty;
    };
    struct Location {
        Side side;
        Price price;
    };
    using Queue = std::list<Resting>;

    std::map<Price, Queue>& book(Side side) { return side == Side::Buy ? bids_ : asks_; }

    static Qty total(const Queue& queue) {
        Qty sum = 0;
        for (const Resting& r : queue) sum += r.qty;
        return sum;
    }

    Resting* find_resting(OrderId id) {
        const auto it = live_.find(id);
        if (it == live_.end()) return nullptr;
        for (Resting& r : book(it->second.side)[it->second.price])
            if (r.id == id) return &r;
        return nullptr;
    }

    [[nodiscard]] bool would_cross(Side side, Price price) const {
        if (side == Side::Buy) return !asks_.empty() && price >= asks_.begin()->first;
        return !bids_.empty() && price <= bids_.rbegin()->first;
    }

    [[nodiscard]] Qty executable(Side side, Price limit) const {
        Qty sum = 0;
        if (side == Side::Buy) {
            for (const auto& [price, queue] : asks_)
                if (price <= limit) sum += total(queue);
        } else {
            for (const auto& [price, queue] : bids_)
                if (price >= limit) sum += total(queue);
        }
        return sum;
    }

    Qty match(OrderId taker, Side side, Price limit, Qty qty) {
        Qty remaining = qty;
        auto& passive = book(opposite(side));
        while (remaining > 0 && !passive.empty()) {
            auto level = side == Side::Buy ? passive.begin() : std::prev(passive.end());
            if (side == Side::Buy ? level->first > limit : level->first < limit) break;
            Queue& queue = level->second;
            while (remaining > 0 && !queue.empty()) {
                Resting& maker = queue.front();
                const Qty fill = std::min(remaining, maker.qty);
                trades_.push_back({taker, maker.id, level->first, fill, side});
                maker.qty -= fill;
                remaining -= fill;
                if (maker.qty == 0) {
                    live_.erase(maker.id);
                    queue.pop_front();
                }
            }
            if (queue.empty()) passive.erase(level);
        }
        return qty - remaining;
    }

    BookConfig config_;
    std::map<Price, Queue> bids_;
    std::map<Price, Queue> asks_;
    std::unordered_map<OrderId, Location> live_;
    std::vector<Trade> trades_;
};

}  // namespace qsl::lob
