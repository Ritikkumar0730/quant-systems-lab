#pragma once
// Rebuilds NASDAQ's displayed order book from an ITCH 5.0 feed.
//
// ITCH is a market-by-order feed: the exchange reports every add, execution,
// cancel, delete and replace for each order reference, and it has already
// done the matching. Rebuilding the book is therefore bookkeeping, not
// matching: an L3 order store (reference -> locate, side, price, shares) plus
// aggregated price levels per stock and side.
//
// The builder trusts nothing. Every event that cannot be applied cleanly
// (unknown reference, overfill, an execution reported under a different stock
// locate than the order's) is counted in Integrity, so replaying a real day
// doubles as a check of both the feed handling and the parser offsets.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "qsl/itch/messages.hpp"
#include "qsl/itch/order_map.hpp"

namespace qsl::itch {

/// One aggregated price level.
struct Level {
    Price4 price;
    std::uint64_t shares;
    std::uint32_t orders;
};

/// Price levels of one side, kept sorted so the best price is at the back:
/// most updates happen near the top of the book, so inserts and erases there
/// move few elements, and the best level is a single load.
class SideBook {
public:
    explicit SideBook(bool is_bid = true) noexcept : is_bid_(is_bid) {}

    void add(Price4 price, std::uint64_t shares) {
        const std::uint32_t k = key(price);
        auto it = lower(k);
        if (it != levels_.end() && it->key == k) {
            it->shares += shares;
            ++it->orders;
        } else {
            levels_.insert(it, Entry{shares, k, 1});
        }
    }

    /// Removes `shares` from the level at `price`; `order_gone` also removes
    /// one order from the level's count. Returns false if the level is
    /// missing or too small (the book is then inconsistent with the feed).
    bool reduce(Price4 price, std::uint64_t shares, bool order_gone) {
        const std::uint32_t k = key(price);
        auto it = lower(k);
        if (it == levels_.end() || it->key != k || it->shares < shares || (order_gone && it->orders == 0)) return false;
        it->shares -= shares;
        if (order_gone) --it->orders;
        if (it->orders == 0) {
            const bool consistent = it->shares == 0;
            levels_.erase(it);
            return consistent;
        }
        return true;
    }

    /// Starts loading the best levels (a hint; no effect on state).
    void prefetch_top() const noexcept {
        if (!levels_.empty()) __builtin_prefetch(&levels_.back());
    }

    [[nodiscard]] bool empty() const noexcept { return levels_.empty(); }
    [[nodiscard]] std::size_t levels() const noexcept { return levels_.size(); }
    [[nodiscard]] Level best() const noexcept { return level(levels_.back()); }

    /// Level `i` counted from the top of the book (0 = best).
    [[nodiscard]] Level at(std::size_t i) const noexcept { return level(levels_[levels_.size() - 1 - i]); }

private:
    struct Entry {  // 16 bytes: four levels per cache line
        std::uint64_t shares;
        std::uint32_t key;
        std::uint32_t orders;
    };

    // Bids ascend by price and asks by inverted price, so for both sides
    // "better" means "larger key" and the best level is the last one.
    [[nodiscard]] std::uint32_t key(Price4 price) const noexcept { return is_bid_ ? price : ~price; }
    [[nodiscard]] Level level(const Entry& e) const noexcept {
        return Level{is_bid_ ? e.key : ~e.key, e.shares, e.orders};
    }
    /// First level with key >= k. On real NASDAQ data about two thirds of
    /// updates land in the top four levels, so scan down from the top first
    /// (touching one or two cache lines) and binary-search only deep levels.
    std::vector<Entry>::iterator lower(std::uint32_t k) {
        std::size_t j = levels_.size();
        for (int steps = 0; j > 0 && levels_[j - 1].key >= k; --j)
            if (++steps == kTopScan) {
                return std::lower_bound(levels_.begin(), levels_.begin() + static_cast<std::ptrdiff_t>(j), k,
                                        [](const Entry& e, std::uint32_t v) { return e.key < v; });
            }
        return levels_.begin() + static_cast<std::ptrdiff_t>(j);
    }

    static constexpr int kTopScan = 8;

    std::vector<Entry> levels_;
    bool is_bid_;
};

struct StockBook {
    Symbol symbol{};
    bool listed = false;       ///< a Stock Directory message has been seen
    char trading_state = 'T';  ///< from Stock Trading Action: T trading, H halted, P paused, Q quotation only
    SideBook bids{true};
    SideBook asks{false};

    [[nodiscard]] bool crossed() const noexcept {
        return !bids.empty() && !asks.empty() && bids.best().price >= asks.best().price;
    }
};

/// What kind of execution produced a print.
enum class Fill : std::uint8_t {
    Displayed,     ///< 'E': resting displayed order traded at its price
    WithPrice,     ///< 'C' printable: traded at a different price
    NonPrintable,  ///< 'C' non-printable: not tape volume (already counted in a cross)
    Hidden,        ///< 'P': non-displayed order
    Cross,         ///< 'Q': opening / closing / halt cross print
};

/// Counts of feed events the builder could not apply. On a correct feed
/// with a correct parser every one of these is zero.
struct Integrity {
    std::uint64_t unknown_order = 0;    ///< E/C/X/D/U for a reference not on the book
    std::uint64_t duplicate_order = 0;  ///< add for a reference already on the book
    std::uint64_t overfill = 0;         ///< executed or cancelled more than the order's remaining size
    std::uint64_t locate_mismatch = 0;  ///< message locate differs from the order's locate
    std::uint64_t level_mismatch = 0;   ///< aggregated level disagrees with the order store

    [[nodiscard]] std::uint64_t total() const noexcept {
        return unknown_order + duplicate_order + overfill + locate_mismatch + level_mismatch;
    }
};

/// How often an update left that stock's book crossed (best bid >= best ask).
struct CrossedCounts {
    std::uint64_t trading_regular = 0;  ///< state T, 09:30-16:00
    std::uint64_t not_trading = 0;      ///< halted, paused or quotation-only (auction pending)
    std::uint64_t outside_regular = 0;  ///< state T, before 09:30 or after 16:00
    [[nodiscard]] std::uint64_t total() const noexcept { return trading_regular + not_trading + outside_regular; }
};

/// Listener that ignores every print.
struct NullPrintListener {
    void on_print(Locate, Timestamp, Price4, std::uint64_t, Fill, char /*cross_type*/) noexcept {}
};

/// ITCH handler that maintains every stock's book.
/// `Listener::on_print(locate, timestamp, price, shares, fill, cross_type)` sees every execution.
template <class Listener = NullPrintListener>
class BookBuilder {
public:
    static constexpr Timestamp kRegularOpen = (9 * 60 + 30) * kNsPerMinute;
    static constexpr Timestamp kRegularClose = 16 * 60 * kNsPerMinute;

    explicit BookBuilder(std::size_t expected_live_orders = std::size_t{1} << 22, Listener listener = Listener{})
        : orders_(expected_live_orders), listener_(std::move(listener)), books_(kLocates) {}

    // ----------------------------------------------------- handler callbacks
    /// Called by the parser before a message is dispatched. Stage 0 (far
    /// ahead) starts loading the stock's book record and the referenced
    /// order's slot; stage 1 (nearer) reads the now-cached book record and
    /// starts loading the top of both price ladders. Purely a performance hint.
    void prefetch(const unsigned char* p, std::size_t length, int stage) const noexcept {
        if (length < kHeaderLength) return;
        const StockBook& b = books_[be16(p + 1)];
        if (stage == 1) {
            b.bids.prefetch_top();
            b.asks.prefetch_top();
            return;
        }
        __builtin_prefetch(&b);
        switch (p[0]) {
            case 'E': case 'C': case 'X': case 'D': case 'U':
                if (length >= kHeaderLength + 8) orders_.prefetch(be64(p + 11));
                break;
            default:
                break;
        }
    }

    void on_stock_directory(const StockDirectory& m) {
        StockBook& b = books_[m.header.locate];
        b.symbol = m.stock;
        b.listed = true;
    }

    void on_trading_action(const TradingAction& m) { books_[m.header.locate].trading_state = m.trading_state; }

    void on_add(const AddOrder& m) {
        add_order(m.ref, m.header.locate, m.side == 'B', m.price, m.shares);
        after_update(m.header);
    }

    void on_executed(const OrderExecuted& m) {
        OrderEntry* o = lookup(m.ref, m.header.locate);
        if (o == nullptr) return;
        const Price4 price = o->price;
        if (reduce(o, m.executed_shares))
            listener_.on_print(m.header.locate, m.header.timestamp, price, m.executed_shares, Fill::Displayed, ' ');
        after_update(m.header);
    }

    void on_executed_with_price(const OrderExecutedWithPrice& m) {
        OrderEntry* o = lookup(m.ref, m.header.locate);
        if (o == nullptr) return;
        if (reduce(o, m.executed_shares))
            listener_.on_print(m.header.locate, m.header.timestamp, m.execution_price, m.executed_shares,
                               m.printable == 'Y' ? Fill::WithPrice : Fill::NonPrintable, ' ');
        after_update(m.header);
    }

    void on_cancel(const OrderCancel& m) {
        OrderEntry* o = lookup(m.ref, m.header.locate);
        if (o == nullptr) return;
        reduce(o, m.cancelled_shares);
        after_update(m.header);
    }

    void on_delete(const OrderDelete& m) {
        OrderEntry* o = lookup(m.ref, m.header.locate);
        if (o == nullptr) return;
        reduce(o, o->shares);
        after_update(m.header);
    }

    void on_replace(const OrderReplace& m) {
        OrderEntry* o = lookup(m.original_ref, m.header.locate);
        if (o == nullptr) return;
        const bool is_bid = o->is_bid;
        reduce(o, o->shares);
        add_order(m.new_ref, m.header.locate, is_bid, m.price, m.shares);
        after_update(m.header);
    }

    void on_trade(const Trade& m) {
        listener_.on_print(m.header.locate, m.header.timestamp, m.price, m.shares, Fill::Hidden, ' ');
    }

    void on_cross_trade(const CrossTrade& m) {
        if (m.shares > 0)
            listener_.on_print(m.header.locate, m.header.timestamp, m.cross_price, m.shares, Fill::Cross, m.cross_type);
    }

    void on_broken_trade(const BrokenTrade&) { ++broken_trades_; }

    // ---------------------------------------------------------------- queries
    [[nodiscard]] const StockBook& book(Locate locate) const noexcept { return books_[locate]; }
    [[nodiscard]] const Integrity& integrity() const noexcept { return integrity_; }
    [[nodiscard]] const CrossedCounts& crossed() const noexcept { return crossed_; }
    [[nodiscard]] std::size_t live_orders() const noexcept { return orders_.size(); }
    [[nodiscard]] std::size_t peak_live_orders() const noexcept { return peak_live_; }
    [[nodiscard]] std::uint64_t broken_trades() const noexcept { return broken_trades_; }
    [[nodiscard]] const OrderMap& orders() const noexcept { return orders_; }
    [[nodiscard]] Listener& listener() noexcept { return listener_; }
    [[nodiscard]] const Listener& listener() const noexcept { return listener_; }

    /// Locate code of a symbol, or 0 if it was never listed (0 is not a valid locate).
    [[nodiscard]] Locate find_symbol(std::string_view symbol) const noexcept {
        for (std::size_t i = 1; i < books_.size(); ++i)
            if (books_[i].listed && books_[i].symbol.view() == symbol) return static_cast<Locate>(i);
        return 0;
    }

private:
    static constexpr std::size_t kLocates = 65536;

    SideBook& side(const OrderEntry& o) noexcept {
        StockBook& b = books_[o.locate];
        return o.is_bid ? b.bids : b.asks;
    }

    void add_order(OrderRef ref, Locate locate, bool is_bid, Price4 price, std::uint32_t shares) {
        OrderEntry* o = orders_.insert(ref);
        if (o == nullptr) {
            ++integrity_.duplicate_order;
            return;
        }
        o->price = price;
        o->shares = shares;
        o->locate = locate;
        o->is_bid = is_bid;
        side(*o).add(price, shares);
        peak_live_ = std::max(peak_live_, orders_.size());
    }

    /// Live order for `ref`, or nullptr (counted as an unknown order).
    OrderEntry* lookup(OrderRef ref, Locate locate) noexcept {
        OrderEntry* o = orders_.find(ref);
        if (o == nullptr) {
            ++integrity_.unknown_order;
            return nullptr;
        }
        if (o->locate != locate) ++integrity_.locate_mismatch;
        return o;
    }

    /// Takes `shares` off an order (clamped to what remains) and removes it
    /// when it reaches zero. Returns false on an overfill.
    bool reduce(OrderEntry* o, std::uint32_t shares) {
        bool ok = true;
        if (shares > o->shares) {
            ++integrity_.overfill;
            shares = o->shares;
            ok = false;
        }
        o->shares -= shares;
        const bool gone = o->shares == 0;
        if (!side(*o).reduce(o->price, shares, gone)) ++integrity_.level_mismatch;
        if (gone) orders_.erase(o);
        return ok;
    }

    void after_update(const Header& h) noexcept {
        const StockBook& b = books_[h.locate];
        if (!b.crossed()) return;
        if (b.trading_state != 'T')
            ++crossed_.not_trading;
        else if (h.timestamp >= kRegularOpen && h.timestamp < kRegularClose)
            ++crossed_.trading_regular;
        else
            ++crossed_.outside_regular;
    }

    OrderMap orders_;
    Listener listener_;
    std::vector<StockBook> books_;
    Integrity integrity_;
    CrossedCounts crossed_;
    std::size_t peak_live_ = 0;
    std::uint64_t broken_trades_ = 0;
};

}  // namespace qsl::itch
