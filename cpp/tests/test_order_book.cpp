// Order book tests: data-structure differentials, hand-checked matching
// scenarios, and a randomized differential test against ReferenceBook.
#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "qsl/lob/hierarchical_bitset.hpp"
#include "qsl/lob/order_book.hpp"
#include "qsl/lob/order_index.hpp"
#include "qsl/lob/reference_book.hpp"
#include "testing.hpp"

using namespace qsl::lob;
using qsl::testing::Rng;

namespace {

struct RecordingListener {
    std::vector<Trade> trades;
    void on_trade(const Trade& t) { trades.push_back(t); }
};

using Book = OrderBook<RecordingListener>;

BookConfig small_config(std::uint32_t max_orders = 1024) { return BookConfig{1, 1000, max_orders}; }

std::vector<LevelInfo> full_depth(const Book& book, Side side) {
    std::vector<LevelInfo> levels(book.config().max_orders);
    levels.resize(book.depth(side, levels));
    return levels;
}

}  // namespace

// --------------------------------------------------------------- structures

TEST_CASE(lob_bitset_matches_std_set) {
    for (std::size_t n : {std::size_t{1}, std::size_t{63}, std::size_t{64}, std::size_t{65},
                          std::size_t{4096}, std::size_t{262149}}) {
        HierarchicalBitset bits(n);
        std::set<std::size_t> model;
        Rng rng{n};
        CHECK(bits.none());
        CHECK(bits.find_next(0) == HierarchicalBitset::npos);
        CHECK(bits.find_prev(n) == HierarchicalBitset::npos);
        for (int op = 0; op < 20000; ++op) {
            const std::size_t i = rng.below(n);
            if (rng.chance(0.55)) {
                bits.set(i);
                model.insert(i);
            } else {
                bits.reset(i);
                model.erase(i);
            }
            const std::size_t q = rng.below(n + 2);  // queries may fall past the end
            const auto next = model.lower_bound(q);
            CHECK(bits.find_next(q) == (next == model.end() ? HierarchicalBitset::npos : *next));
            auto prev = model.upper_bound(std::min(q, n - 1));
            CHECK(bits.find_prev(q) == (prev == model.begin() ? HierarchicalBitset::npos : *--prev));
            CHECK(bits.test(i) == model.contains(i));
            CHECK(bits.none() == model.empty());
        }
    }
}

TEST_CASE(lob_order_index_matches_unordered_map) {
    OrderIndex index(5000);
    std::unordered_map<OrderId, std::uint32_t> model;
    Rng rng{99};
    for (int op = 0; op < 400000; ++op) {
        // Small key universe -> long probe clusters and many backward shifts.
        const OrderId key = rng.below(12000) * 0x10001ULL;
        const double r = rng.uniform();
        if (r < 0.45) {
            const auto value = static_cast<std::uint32_t>(rng.below(1u << 30));
            const bool expect = !model.contains(key) && model.size() < 5000;
            CHECK(index.insert(key, value) == expect);
            if (expect) model[key] = value;
        } else if (r < 0.9) {
            CHECK(index.erase(key) == (model.erase(key) == 1));
        }
        const auto it = model.find(key);
        CHECK(index.find(key) == (it == model.end() ? OrderIndex::kNotFound : it->second));
        CHECK(index.size() == model.size());
    }
    for (const auto& [key, value] : model) CHECK(index.find(key) == value);
}

// ---------------------------------------------------------------- scenarios

TEST_CASE(lob_resting_orders_and_top_of_book) {
    Book book(small_config());
    CHECK(!book.best_bid() && !book.best_ask());
    CHECK(book.add(1, Side::Buy, 100, 10).resting == 10);
    CHECK(book.add(2, Side::Buy, 101, 5).resting == 5);
    CHECK(book.add(3, Side::Sell, 105, 7).resting == 7);
    CHECK(book.add(4, Side::Sell, 103, 3).resting == 3);
    CHECK(book.best_bid() == 101 && book.best_ask() == 103);
    CHECK(book.volume_at(Side::Buy, 100) == 10 && book.volume_at(Side::Sell, 105) == 7);
    CHECK(book.volume_at(Side::Sell, 100) == 0);  // wrong side
    CHECK(book.order_count() == 4);
    const auto bids = full_depth(book, Side::Buy);
    REQUIRE(bids.size() == 2);
    CHECK(bids[0] == (LevelInfo{101, 5, 1}) && bids[1] == (LevelInfo{100, 10, 1}));
    CHECK(book.listener().trades.empty());
    CHECK(book.check_invariants());
}

TEST_CASE(lob_price_time_priority_and_maker_price) {
    Book book(small_config());
    book.add(1, Side::Sell, 101, 5);
    book.add(2, Side::Sell, 101, 5);  // same price, later: second in queue
    book.add(3, Side::Sell, 102, 5);
    // Aggressive buy at 102 for 12: fills 1 then 2 at 101 (FIFO), then 3 at 102.
    const ExecResult r = book.add(10, Side::Buy, 102, 12);
    CHECK(r == (ExecResult{Reject::None, 12, 0}));
    const std::vector<Trade> expected{{10, 1, 101, 5, Side::Buy},
                                      {10, 2, 101, 5, Side::Buy},
                                      {10, 3, 102, 2, Side::Buy}};
    CHECK(book.listener().trades == expected);
    CHECK(book.best_ask() == 102 && book.volume_at(Side::Sell, 102) == 3);
    CHECK(!book.find(1) && !book.find(2) && book.find(3)->qty == 3);
    CHECK(book.check_invariants());
}

TEST_CASE(lob_partial_fill_rests_remainder) {
    Book book(small_config());
    book.add(1, Side::Sell, 100, 4);
    const ExecResult r = book.add(2, Side::Buy, 101, 10);
    CHECK(r == (ExecResult{Reject::None, 4, 6}));
    CHECK(book.best_bid() == 101 && !book.best_ask());
    CHECK(book.check_invariants());
}

TEST_CASE(lob_market_order_sweeps_and_never_rests) {
    Book book(small_config());
    book.add(1, Side::Buy, 99, 3);
    book.add(2, Side::Buy, 98, 3);
    const ExecResult r = book.market(3, Side::Sell, 10);
    CHECK(r == (ExecResult{Reject::None, 6, 0}));
    CHECK(!book.best_bid() && book.order_count() == 0);
    CHECK(book.listener().trades.size() == 2 && book.listener().trades[1].price == 98);
    CHECK(book.check_invariants());
}

TEST_CASE(lob_time_in_force_semantics) {
    Book book(small_config());
    book.add(1, Side::Sell, 100, 5);
    book.add(2, Side::Sell, 101, 5);
    // IOC: takes what crosses, discards the rest.
    CHECK(book.add(3, Side::Buy, 100, 8, TimeInForce::ImmediateOrCancel) == (ExecResult{Reject::None, 5, 0}));
    CHECK(!book.find(3) && book.best_ask() == 101);
    // FOK: all-or-nothing, and an unfillable FOK leaves the book untouched.
    CHECK(book.add(4, Side::Buy, 101, 6, TimeInForce::FillOrKill).reject == Reject::FillOrKillUnfilled);
    CHECK(book.volume_at(Side::Sell, 101) == 5);
    CHECK(book.add(5, Side::Buy, 101, 5, TimeInForce::FillOrKill) == (ExecResult{Reject::None, 5, 0}));
    // Post-only: rejected if it would take liquidity, rests otherwise.
    book.add(6, Side::Sell, 105, 5);
    CHECK(book.add(7, Side::Buy, 105, 1, TimeInForce::PostOnly).reject == Reject::PostOnlyWouldCross);
    CHECK(book.add(8, Side::Buy, 104, 1, TimeInForce::PostOnly) == (ExecResult{Reject::None, 0, 1}));
    CHECK(book.check_invariants());
}

TEST_CASE(lob_cancel_reduce_replace_priority) {
    Book book(small_config());
    book.add(1, Side::Buy, 100, 10);
    book.add(2, Side::Buy, 100, 10);
    // Reduce keeps priority: order 1 is still first in the queue.
    CHECK(book.reduce(1, 4) == Reject::None);
    CHECK(book.reduce(1, 4) == Reject::InvalidQuantity);  // not a decrease
    CHECK(book.volume_at(Side::Buy, 100) == 14);
    book.market(50, Side::Sell, 4);
    CHECK(book.listener().trades.back().maker == 1 && !book.find(1));
    // Replace to a new price loses priority and moves the level.
    book.add(3, Side::Buy, 100, 10);
    CHECK(book.replace(2, 99, 10) == (ExecResult{Reject::None, 0, 10}));
    CHECK(book.best_bid() == 100 && book.volume_at(Side::Buy, 99) == 10);
    // Same-price shrink via replace keeps priority (order 3 still at 100).
    CHECK(book.replace(3, 100, 7) == (ExecResult{Reject::None, 0, 7}));
    // A replace that crosses trades immediately.
    book.add(4, Side::Sell, 102, 2);
    CHECK(book.replace(3, 102, 7) == (ExecResult{Reject::None, 2, 5}));
    CHECK(book.cancel(3) == Reject::None);
    CHECK(book.cancel(3) == Reject::UnknownOrder);
    CHECK(book.best_bid() == 99);  // best moves to the next populated level
    CHECK(book.check_invariants());
}

TEST_CASE(lob_rejections_have_no_side_effects) {
    Book book(BookConfig{10, 20, 2});
    CHECK(book.add(1, Side::Buy, 9, 1).reject == Reject::InvalidPrice);
    CHECK(book.add(1, Side::Buy, 21, 1).reject == Reject::InvalidPrice);
    CHECK(book.add(1, Side::Buy, 15, 0).reject == Reject::InvalidQuantity);
    CHECK(book.add(1, Side::Buy, 15, 1).ok());
    CHECK(book.add(1, Side::Sell, 18, 1).reject == Reject::DuplicateOrderId);
    CHECK(book.add(2, Side::Buy, 14, 1).ok());
    CHECK(book.add(3, Side::Buy, 13, 1).reject == Reject::BookFull);  // capacity 2
    CHECK(book.add(3, Side::Sell, 15, 1, TimeInForce::ImmediateOrCancel).filled == 1);  // IOC needs no slot
    CHECK(book.replace(9, 15, 1).reject == Reject::UnknownOrder);
    CHECK(book.replace(2, 25, 1).reject == Reject::InvalidPrice);
    CHECK(book.order_count() == 1 && book.best_bid() == 14);
    CHECK(book.listener().trades.size() == 1);
    // Ladder edges are valid prices.
    CHECK(book.add(4, Side::Sell, 20, 1).ok());
    CHECK(book.market(5, Side::Buy, 5).filled == 1);
    CHECK(book.check_invariants());
}

TEST_CASE(lob_negative_price_ladder) {
    // Spread instruments (e.g. calendar spreads) can trade at negative prices.
    Book book(BookConfig{-50, 50, 64});
    book.add(1, Side::Sell, -3, 2);
    book.add(2, Side::Buy, -10, 2);
    CHECK(book.best_ask() == -3 && book.best_bid() == -10);
    CHECK(book.add(3, Side::Buy, -3, 1).filled == 1);
    CHECK(book.listener().trades.back().price == -3);
    CHECK(book.check_invariants());
}

TEST_CASE(lob_invalid_config_throws) {
    bool threw = false;
    try {
        Book book(BookConfig{10, 5, 8});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ------------------------------------------------------ randomized oracle

namespace {

/// What a randomized run actually exercised, so a test can assert that the
/// random flow reached every code path (a differential test that never
/// triggers a FOK rejection proves nothing about FOK).
struct Coverage {
    std::size_t trades = 0;
    std::size_t rested = 0;
    std::size_t rejects[8] = {};
    void record(Reject r) { ++rejects[static_cast<std::size_t>(r)]; }
    [[nodiscard]] std::size_t count(Reject r) const { return rejects[static_cast<std::size_t>(r)]; }
};

/// Drives OrderBook and ReferenceBook with the same random order flow and
/// compares every result, every trade and (periodically) the full depth.
Coverage differential_run(const BookConfig& cfg, std::uint64_t seed, int ops) {
    Coverage cov;
    Book fast(cfg);
    ReferenceBook ref(cfg);
    Rng rng{seed};
    std::vector<OrderId> ids;
    OrderId next_id = 1;
    const Price span = cfg.max_price - cfg.min_price;
    Price mid = cfg.min_price + span / 2;
    auto any_id = [&]() -> OrderId {
        if (ids.empty() || rng.chance(0.05)) return next_id + 1000;  // unknown id
        const std::size_t window = std::min<std::size_t>(ids.size(), 512);
        return ids[ids.size() - 1 - rng.below(window)];
    };
    auto random_tif = [&]() {
        const double r = rng.uniform();
        if (r < 0.7) return TimeInForce::GoodTillCancel;
        if (r < 0.8) return TimeInForce::ImmediateOrCancel;
        if (r < 0.9) return TimeInForce::FillOrKill;
        return TimeInForce::PostOnly;
    };

    for (int op = 0; op < ops; ++op) {
        if (rng.chance(0.1)) mid = std::clamp<Price>(mid + (rng.chance(0.5) ? 1 : -1), cfg.min_price, cfg.max_price);
        const double r = rng.uniform();
        if (r < 0.45) {
            const Side side = rng.chance(0.5) ? Side::Buy : Side::Sell;
            const Price offset = static_cast<Price>(rng.below(12)) - 3;  // some orders cross
            Price price = side == Side::Buy ? mid - offset : mid + offset;
            if (rng.chance(0.01)) price = rng.chance(0.5) ? cfg.min_price - 1 : cfg.max_price + 1;
            const Qty qty = rng.chance(0.01) ? 0 : 1 + rng.below(40);
            const OrderId id = rng.chance(0.03) ? any_id() : next_id++;
            const TimeInForce tif = random_tif();
            const ExecResult a = fast.add(id, side, price, qty, tif);
            const ExecResult b = ref.add(id, side, price, qty, tif);
            CHECK(a == b);
            cov.record(a.reject);
            cov.rested += a.resting > 0 ? 1 : 0;
            ids.push_back(id);
        } else if (r < 0.70) {
            const OrderId id = any_id();
            const Reject a = fast.cancel(id);
            CHECK(a == ref.cancel(id));
            cov.record(a);
        } else if (r < 0.80) {
            const OrderId id = any_id();
            const Qty qty = rng.below(45);
            CHECK(fast.reduce(id, qty) == ref.reduce(id, qty));
        } else if (r < 0.93) {
            const OrderId id = any_id();
            const Price price = mid + static_cast<Price>(rng.below(9)) - 4;
            const Qty qty = rng.below(45);
            const ExecResult a = fast.replace(id, price, qty);
            CHECK(a == ref.replace(id, price, qty));
            cov.record(a.reject);
        } else {
            const Side side = rng.chance(0.5) ? Side::Buy : Side::Sell;
            const Qty qty = 1 + rng.below(150);
            const OrderId id = next_id++;
            CHECK(fast.market(id, side, qty) == ref.market(id, side, qty));
        }
        CHECK(fast.listener().trades == ref.trades());
        cov.trades += fast.listener().trades.size();
        fast.listener().trades.clear();
        ref.clear_trades();

        if (op % 64 == 0) {
            CHECK(fast.best_bid() == ref.best_bid());
            CHECK(fast.best_ask() == ref.best_ask());
            CHECK(fast.order_count() == ref.order_count());
            CHECK(full_depth(fast, Side::Buy) == ref.levels(Side::Buy));
            CHECK(full_depth(fast, Side::Sell) == ref.levels(Side::Sell));
        }
        if (op % 4096 == 0) CHECK(fast.check_invariants());
    }
    CHECK(fast.check_invariants());
    CHECK(full_depth(fast, Side::Buy) == ref.levels(Side::Buy));
    CHECK(full_depth(fast, Side::Sell) == ref.levels(Side::Sell));
    return cov;
}

/// Every run must reach the matching path and every rejection reason
/// (BookFull only occurs when the capacity is small enough to fill).
void check_coverage(const Coverage& cov, bool expect_book_full) {
    CHECK(cov.trades > 1000);
    CHECK(cov.rested > 1000);
    CHECK(cov.count(Reject::None) > 1000);
    CHECK(cov.count(Reject::DuplicateOrderId) > 0);
    CHECK(cov.count(Reject::InvalidPrice) > 0);
    CHECK(cov.count(Reject::InvalidQuantity) > 0);
    CHECK(cov.count(Reject::UnknownOrder) > 0);
    CHECK(cov.count(Reject::PostOnlyWouldCross) > 0);
    CHECK(cov.count(Reject::FillOrKillUnfilled) > 0);
    CHECK((cov.count(Reject::BookFull) > 0) == expect_book_full);
}

}  // namespace

TEST_CASE(lob_differential_narrow_ladder) {
    for (std::uint64_t seed = 1; seed <= 3; ++seed)
        check_coverage(differential_run(BookConfig{100, 140, 1u << 14}, seed, 60000), false);
}

TEST_CASE(lob_differential_wide_ladder) {
    for (std::uint64_t seed = 11; seed <= 13; ++seed)
        check_coverage(differential_run(BookConfig{1, 65536, 1u << 14}, seed, 60000), false);
}

TEST_CASE(lob_differential_tiny_capacity) {
    for (std::uint64_t seed = 21; seed <= 23; ++seed)
        check_coverage(differential_run(BookConfig{1, 200, 8}, seed, 60000), true);
}

TEST_CASE(lob_differential_negative_prices) {
    for (std::uint64_t seed = 31; seed <= 33; ++seed)
        check_coverage(differential_run(BookConfig{-60, 60, 1u << 12}, seed, 60000), false);
}
