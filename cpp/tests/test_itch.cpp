// ITCH 5.0 parser and book builder tests.
//
// 1. Golden messages: every order-related type is written field by field in
//    spec order by a tiny big-endian writer (independent of the parser's
//    offset table) and decoded back.
// 2. Framing: the same stream fed in random chunk sizes gives identical
//    results; malformed frames are counted and skipped.
// 3. Book builder vs a naive std::map reference on random feeds, including
//    deliberately invalid events (unknown references, overfills, duplicates).

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "qsl/itch/book.hpp"
#include "qsl/itch/parser.hpp"
#include "testing.hpp"

using namespace qsl::itch;
using qsl::testing::Rng;

namespace {

/// Writes ITCH frames field by field in spec order.
struct Writer {
    std::vector<unsigned char> buf;
    std::size_t frame_start = 0;

    void u8(std::uint64_t v) { buf.push_back(static_cast<unsigned char>(v & 0xff)); }
    void be(std::uint64_t v, int bytes) {
        for (int i = bytes - 1; i >= 0; --i) u8(v >> (8 * i));
    }
    void ch(char c) { u8(static_cast<unsigned char>(c)); }
    void sym(const std::string& s) {
        for (std::size_t i = 0; i < 8; ++i) ch(i < s.size() ? s[i] : ' ');
    }
    void begin(char type, Locate locate, std::uint16_t tracking, Timestamp ts) {
        frame_start = buf.size();
        be(0, 2);  // length placeholder
        ch(type);
        be(locate, 2);
        be(tracking, 2);
        be(ts, 6);
    }
    void end() {
        const std::size_t len = buf.size() - frame_start - 2;
        buf[frame_start] = static_cast<unsigned char>(len >> 8);
        buf[frame_start + 1] = static_cast<unsigned char>(len & 0xff);
    }

    void directory(Locate loc, const std::string& symbol, Timestamp ts = 1) {
        begin('R', loc, 0, ts);
        sym(symbol);
        ch('Q'); ch('N'); be(100, 4); ch('N'); ch('C'); ch('Z'); ch(' '); ch('P'); ch('N'); ch(' '); ch('1'); ch('N');
        be(0, 4); ch('N');
        end();
    }
    void add(Locate loc, Timestamp ts, OrderRef ref, char side, std::uint32_t shares, const std::string& s,
             Price4 px, const char* mpid = nullptr) {
        begin(mpid ? 'F' : 'A', loc, 7, ts);
        be(ref, 8); ch(side); be(shares, 4); sym(s); be(px, 4);
        if (mpid)
            for (int i = 0; i < 4; ++i) ch(mpid[i]);
        end();
    }
    void executed(Locate loc, Timestamp ts, OrderRef ref, std::uint32_t shares, std::uint64_t match) {
        begin('E', loc, 0, ts); be(ref, 8); be(shares, 4); be(match, 8); end();
    }
    void executed_px(Locate loc, Timestamp ts, OrderRef ref, std::uint32_t shares, std::uint64_t match,
                     char printable, Price4 px) {
        begin('C', loc, 0, ts); be(ref, 8); be(shares, 4); be(match, 8); ch(printable); be(px, 4); end();
    }
    void cancel(Locate loc, Timestamp ts, OrderRef ref, std::uint32_t shares) {
        begin('X', loc, 0, ts); be(ref, 8); be(shares, 4); end();
    }
    void del(Locate loc, Timestamp ts, OrderRef ref) {
        begin('D', loc, 0, ts); be(ref, 8); end();
    }
    void replace(Locate loc, Timestamp ts, OrderRef old_ref, OrderRef new_ref, std::uint32_t shares, Price4 px) {
        begin('U', loc, 0, ts); be(old_ref, 8); be(new_ref, 8); be(shares, 4); be(px, 4); end();
    }
    void trade(Locate loc, Timestamp ts, OrderRef ref, char side, std::uint32_t shares, const std::string& s,
               Price4 px, std::uint64_t match) {
        begin('P', loc, 0, ts); be(ref, 8); ch(side); be(shares, 4); sym(s); be(px, 4); be(match, 8); end();
    }
    void cross(Locate loc, Timestamp ts, std::uint64_t shares, const std::string& s, Price4 px, std::uint64_t match,
               char type) {
        begin('Q', loc, 0, ts); be(shares, 8); sym(s); be(px, 4); be(match, 8); ch(type); end();
    }
};

/// Records every decoded message.
struct Recorder {
    std::vector<SystemEvent> system;
    std::vector<StockDirectory> directory;
    std::vector<TradingAction> actions;
    std::vector<AddOrder> adds;
    std::vector<OrderExecuted> execs;
    std::vector<OrderExecutedWithPrice> execs_px;
    std::vector<OrderCancel> cancels;
    std::vector<OrderDelete> deletes;
    std::vector<OrderReplace> replaces;
    std::vector<Trade> trades;
    std::vector<CrossTrade> crosses;
    std::vector<BrokenTrade> broken;
    std::vector<Noii> noii;
    std::vector<char> other;

    void on_system_event(const SystemEvent& m) { system.push_back(m); }
    void on_stock_directory(const StockDirectory& m) { directory.push_back(m); }
    void on_trading_action(const TradingAction& m) { actions.push_back(m); }
    void on_add(const AddOrder& m) { adds.push_back(m); }
    void on_executed(const OrderExecuted& m) { execs.push_back(m); }
    void on_executed_with_price(const OrderExecutedWithPrice& m) { execs_px.push_back(m); }
    void on_cancel(const OrderCancel& m) { cancels.push_back(m); }
    void on_delete(const OrderDelete& m) { deletes.push_back(m); }
    void on_replace(const OrderReplace& m) { replaces.push_back(m); }
    void on_trade(const Trade& m) { trades.push_back(m); }
    void on_cross_trade(const CrossTrade& m) { crosses.push_back(m); }
    void on_broken_trade(const BrokenTrade& m) { broken.push_back(m); }
    void on_noii(const Noii& m) { noii.push_back(m); }
    void on_other(const OtherMessage& m) { other.push_back(m.header.type); }
};

constexpr Timestamp kTs = 34'200'123'456'789ULL;  // 09:30:00.123456789

}  // namespace

TEST_CASE(itch_message_lengths_match_the_spec) {
    // Fixed sizes from the ITCH 5.0 specification.
    const std::pair<char, int> spec[] = {{'S', 12}, {'R', 39}, {'H', 25}, {'Y', 20}, {'L', 26}, {'V', 35},
                                         {'W', 12}, {'K', 28}, {'J', 35}, {'h', 21}, {'A', 36}, {'F', 40},
                                         {'E', 31}, {'C', 36}, {'X', 23}, {'D', 19}, {'U', 35}, {'P', 44},
                                         {'Q', 40}, {'B', 19}, {'I', 50}, {'N', 20}, {'O', 48}};
    int defined = 0;
    for (int t = 0; t < 256; ++t) defined += kMessageLength[static_cast<std::size_t>(t)] != 0;
    CHECK(defined == 23);
    for (auto [type, len] : spec) CHECK(kMessageLength[static_cast<unsigned char>(type)] == len);
}

TEST_CASE(itch_decodes_golden_messages_field_by_field) {
    Writer w;
    w.begin('S', 0, 3, kTs); w.ch('Q'); w.end();
    w.directory(42, "MSFT");
    w.begin('H', 42, 0, kTs); w.sym("MSFT"); w.ch('T'); w.ch(' '); for (int i = 0; i < 4; ++i) w.ch(' '); w.end();
    w.add(42, kTs, 0x0102030405060708ULL, 'B', 300, "MSFT", 1575'900);
    w.add(42, kTs + 1, 99, 'S', 0xfffffffe, "MSFT", 0xffffffff, "GSCO");
    w.executed(42, kTs + 2, 0x0102030405060708ULL, 100, 0xa1a2a3a4a5a6a7a8ULL);
    w.executed_px(42, kTs + 3, 99, 50, 77, 'N', 1576'000);
    w.cancel(42, kTs + 4, 99, 25);
    w.del(42, kTs + 5, 99);
    w.replace(42, kTs + 6, 0x0102030405060708ULL, 1234, 400, 1575'800);
    w.trade(42, kTs + 7, 0, 'S', 1000, "MSFT", 1575'850, 88);
    w.cross(42, kTs + 8, 0x1'0000'0000ULL, "MSFT", 1575'900, 89, 'C');
    w.begin('B', 42, 0, kTs + 9); w.be(88, 8); w.end();
    w.begin('I', 42, 0, kTs + 10);
    w.be(5000, 8); w.be(1200, 8); w.ch('B'); w.sym("MSFT"); w.be(1575'000, 4); w.be(1576'000, 4); w.be(1575'500, 4);
    w.ch('C'); w.ch('1');
    w.end();
    w.begin('Y', 42, 0, kTs + 11); w.sym("MSFT"); w.ch('0'); w.end();

    Recorder r;
    Parser<Recorder> parser(r);
    const std::size_t used = parser.parse(w.buf.data(), w.buf.size());
    CHECK(used == w.buf.size());
    CHECK(parser.stats().messages == 15);
    CHECK(parser.stats().bad_length == 0);
    CHECK(parser.stats().unknown_type == 0);

    REQUIRE(r.system.size() == 1);
    CHECK(r.system[0].event_code == 'Q' && r.system[0].header.tracking == 3);
    CHECK(r.system[0].header.timestamp == kTs);

    REQUIRE(r.directory.size() == 1);
    const StockDirectory& d = r.directory[0];
    CHECK(d.header.locate == 42 && d.stock.view() == "MSFT" && d.round_lot_size == 100);
    CHECK(d.market_category == 'Q' && d.issue_classification == 'C' && d.inverse_indicator == 'N');

    REQUIRE(r.adds.size() == 2);
    CHECK(r.adds[0].ref == 0x0102030405060708ULL && r.adds[0].side == 'B' && r.adds[0].shares == 300);
    CHECK(r.adds[0].price == 1575'900 && r.adds[0].stock.view() == "MSFT" && r.adds[0].header.tracking == 7);
    CHECK(r.adds[1].shares == 0xfffffffe && r.adds[1].price == 0xffffffff);  // full unsigned range
    CHECK(std::string(r.adds[1].attribution.data(), 4) == "GSCO");

    REQUIRE(r.execs.size() == 1);
    CHECK(r.execs[0].executed_shares == 100 && r.execs[0].match_number == 0xa1a2a3a4a5a6a7a8ULL);
    REQUIRE(r.execs_px.size() == 1);
    CHECK(r.execs_px[0].printable == 'N' && r.execs_px[0].execution_price == 1576'000 && r.execs_px[0].match_number == 77);
    REQUIRE(r.cancels.size() == 1 && r.deletes.size() == 1);
    CHECK(r.cancels[0].cancelled_shares == 25 && r.deletes[0].ref == 99);
    REQUIRE(r.replaces.size() == 1);
    CHECK(r.replaces[0].original_ref == 0x0102030405060708ULL && r.replaces[0].new_ref == 1234);
    CHECK(r.replaces[0].shares == 400 && r.replaces[0].price == 1575'800);
    REQUIRE(r.trades.size() == 1);
    CHECK(r.trades[0].side == 'S' && r.trades[0].shares == 1000 && r.trades[0].match_number == 88);
    REQUIRE(r.crosses.size() == 1);
    CHECK(r.crosses[0].shares == 0x1'0000'0000ULL && r.crosses[0].cross_type == 'C');  // 64-bit cross size
    CHECK(r.crosses[0].cross_price == 1575'900);
    REQUIRE(r.broken.size() == 1);
    CHECK(r.broken[0].match_number == 88);
    REQUIRE(r.noii.size() == 1);
    CHECK(r.noii[0].paired_shares == 5000 && r.noii[0].imbalance_shares == 1200 && r.noii[0].imbalance_direction == 'B');
    CHECK(r.noii[0].far_price == 1575'000 && r.noii[0].near_price == 1576'000);
    CHECK(r.noii[0].current_reference_price == 1575'500 && r.noii[0].price_variation_indicator == '1');
    REQUIRE(r.other.size() == 1);
    CHECK(r.other[0] == 'Y');
    CHECK(r.actions.size() == 1 && r.actions[0].trading_state == 'T');
}

TEST_CASE(itch_timestamp_uses_all_48_bits) {
    Writer w;
    const Timestamp max48 = (Timestamp{1} << 48) - 1;
    w.begin('S', 0, 0, max48); w.ch('C'); w.end();
    Recorder r;
    Parser<Recorder> p(r);
    p.parse(w.buf.data(), w.buf.size());
    REQUIRE(r.system.size() == 1);
    CHECK(r.system[0].header.timestamp == max48);
}

TEST_CASE(itch_streaming_is_independent_of_chunk_boundaries) {
    Writer w;
    Rng rng{7};
    w.directory(1, "AAA");
    for (OrderRef ref = 1; ref <= 3000; ++ref) {
        w.add(1, kTs + ref, ref, rng.chance(0.5) ? 'B' : 'S', static_cast<std::uint32_t>(1 + rng.below(1000)), "AAA",
              static_cast<Price4>(100'000 + rng.below(500) * 100));
        if (rng.chance(0.3)) w.del(1, kTs + ref, ref);
    }

    Recorder whole;
    Parser<Recorder> p1(whole);
    CHECK(p1.parse(w.buf.data(), w.buf.size()) == w.buf.size());

    // Feed in random chunks, carrying any partial frame into the next chunk
    // exactly as a socket or file reader would.
    Recorder chunked;
    Parser<Recorder> p2(chunked);
    std::vector<unsigned char> pending;
    std::size_t pos = 0;
    while (pos < w.buf.size()) {
        const std::size_t n = std::min<std::size_t>(1 + rng.below(97), w.buf.size() - pos);
        pending.insert(pending.end(), w.buf.begin() + static_cast<std::ptrdiff_t>(pos),
                       w.buf.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
        const std::size_t used = p2.parse(pending.data(), pending.size());
        pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(used));
    }
    CHECK(pending.empty());
    CHECK(p2.stats().messages == p1.stats().messages);
    CHECK(p2.stats().bytes == w.buf.size());
    REQUIRE(chunked.adds.size() == whole.adds.size());
    bool same = chunked.deletes.size() == whole.deletes.size();
    for (std::size_t i = 0; i < whole.adds.size(); ++i)
        same = same && chunked.adds[i].ref == whole.adds[i].ref && chunked.adds[i].price == whole.adds[i].price &&
               chunked.adds[i].shares == whole.adds[i].shares;
    CHECK(same);
}

TEST_CASE(itch_malformed_frames_are_counted_and_skipped) {
    Writer w;
    w.add(1, kTs, 1, 'B', 100, "AAA", 10'000);
    // Add Order with one byte too many.
    w.begin('A', 1, 0, kTs); w.be(2, 8); w.ch('B'); w.be(5, 4); w.sym("AAA"); w.be(10'000, 4); w.u8(0); w.end();
    // Undefined type byte.
    w.begin('z', 1, 0, kTs); w.be(0, 4); w.end();
    // Zero-length frame.
    w.be(0, 2);
    w.add(1, kTs, 3, 'S', 100, "AAA", 10'100);

    Recorder r;
    Parser<Recorder> p(r);
    CHECK(p.parse(w.buf.data(), w.buf.size()) == w.buf.size());
    CHECK(p.stats().messages == 2);
    CHECK(p.stats().bad_length == 2);
    CHECK(p.stats().unknown_type == 1);
    REQUIRE(r.adds.size() == 2);
    CHECK(r.adds[0].ref == 1 && r.adds[1].ref == 3);  // stream stays in sync after bad frames
}

// ------------------------------------------------------------- book builder
namespace {

/// The obvious implementation: hash map of orders, std::map of levels.
struct ReferenceItchBook {
    struct Order { Locate locate; bool bid; Price4 price; std::uint32_t shares; };
    struct Agg { std::uint64_t shares = 0; std::uint32_t orders = 0; };
    std::unordered_map<OrderRef, Order> orders;
    std::map<std::pair<Locate, bool>, std::map<Price4, Agg>> levels;
    Integrity integrity;
    std::uint64_t printed_shares = 0;
    double printed_notional = 0;
    std::uint64_t crossed = 0;

    std::map<Price4, Agg>& side(const Order& o) { return levels[{o.locate, o.bid}]; }

    void remove_shares(OrderRef ref, std::uint32_t shares, bool print, Price4 print_px) {
        Order& o = orders.at(ref);
        if (shares > o.shares) { ++integrity.overfill; shares = o.shares; print = false; }
        auto& lv = side(o);
        Agg& a = lv[o.price];
        a.shares -= shares;
        o.shares -= shares;
        if (print) { printed_shares += shares; printed_notional += double(shares) * print_px; }
        if (o.shares == 0) {
            if (--a.orders == 0) lv.erase(o.price);
            orders.erase(ref);
        }
    }
    bool known(OrderRef ref, Locate loc) {
        auto it = orders.find(ref);
        if (it == orders.end()) { ++integrity.unknown_order; return false; }
        if (it->second.locate != loc) ++integrity.locate_mismatch;
        return true;
    }
    void insert(OrderRef ref, Locate loc, bool bid, Price4 px, std::uint32_t shares) {
        if (orders.count(ref)) { ++integrity.duplicate_order; return; }
        orders[ref] = Order{loc, bid, px, shares};
        Agg& a = side(orders[ref])[px];
        a.shares += shares;
        ++a.orders;
    }
    bool is_crossed(Locate loc) {
        auto& b = levels[{loc, true}];
        auto& s = levels[{loc, false}];
        return !b.empty() && !s.empty() && b.rbegin()->first >= s.begin()->first;
    }
};

struct PrintTotals {
    std::uint64_t shares = 0;
    double notional = 0;
    void on_print(Locate, Timestamp, Price4 px, std::uint64_t sh, Fill fill, char) {
        if (fill == Fill::Displayed || fill == Fill::WithPrice) { shares += sh; notional += double(sh) * px; }
    }
};

bool same_depth(const BookBuilder<PrintTotals>& b, ReferenceItchBook& ref, Locate loc) {
    for (bool bid : {true, false}) {
        const SideBook& s = bid ? b.book(loc).bids : b.book(loc).asks;
        auto& lv = ref.levels[{loc, bid}];
        if (s.levels() != lv.size()) return false;
        std::size_t i = 0;
        if (bid) {
            for (auto it = lv.rbegin(); it != lv.rend(); ++it, ++i) {
                const Level l = s.at(i);
                if (l.price != it->first || l.shares != it->second.shares || l.orders != it->second.orders) return false;
            }
        } else {
            for (auto it = lv.begin(); it != lv.end(); ++it, ++i) {
                const Level l = s.at(i);
                if (l.price != it->first || l.shares != it->second.shares || l.orders != it->second.orders) return false;
            }
        }
    }
    return true;
}

/// Random but realistic ITCH order flow across several stocks, with a small
/// rate of deliberately invalid events. Applies each event to the reference
/// as it writes it.
void random_feed(std::uint64_t seed, std::size_t events, Writer& w, ReferenceItchBook& ref, double invalid_rate) {
    Rng rng{seed};
    const Locate locs[] = {3, 17, 250, 9000};
    for (Locate l : locs) w.directory(l, "S" + std::to_string(l));
    std::vector<OrderRef> live;
    OrderRef next_ref = 1000;
    std::uint64_t match = 1;
    Timestamp ts = 30'000 * kNsPerSecond;
    auto live_order = [&](std::size_t i) -> ReferenceItchBook::Order& { return ref.orders.at(live[i]); };
    auto drop_dead = [&] {
        live.erase(std::remove_if(live.begin(), live.end(), [&](OrderRef r) { return !ref.orders.count(r); }), live.end());
    };

    for (std::size_t e = 0; e < events; ++e) {
        ts += 1 + rng.below(1'000'000);
        const double u = rng.uniform();
        if (rng.chance(invalid_rate)) {
            const int kind = static_cast<int>(rng.below(4));
            const Locate loc = locs[rng.below(4)];
            if (kind == 0) {  // unknown reference
                w.del(loc, ts, 1ULL << 60 | e);
                ref.known(1ULL << 60 | e, loc);
            } else if (kind == 1 && !live.empty()) {  // overfill
                const OrderRef r = live[rng.below(live.size())];
                const auto o = ref.orders.at(r);
                w.executed(o.locate, ts, r, o.shares + 1, match++);
                ref.known(r, o.locate);
                ref.remove_shares(r, o.shares + 1, true, o.price);
            } else if (kind == 2 && !live.empty()) {  // duplicate add
                const OrderRef r = live[rng.below(live.size())];
                const auto o = ref.orders.at(r);
                w.add(o.locate, ts, r, 'B', 100, "X", 10'000);
                ref.insert(r, o.locate, true, 10'000, 100);
            } else if (kind == 3 && !live.empty()) {  // execution reported under the wrong locate
                const OrderRef r = live[rng.below(live.size())];
                const auto o = ref.orders.at(r);
                const Locate wrong = o.locate == 3 ? Locate{17} : Locate{3};
                w.executed(wrong, ts, r, 1, match++);
                ref.known(r, wrong);
                ref.remove_shares(r, 1, true, o.price);
            }
            drop_dead();
            continue;
        }
        if (live.empty() || u < 0.45) {
            const Locate loc = locs[rng.below(4)];
            const bool bid = rng.chance(0.5);
            // Bids below and asks above a 100.00 mid most of the time, so the
            // book is usually uncrossed but occasionally crosses.
            const std::int64_t offset = static_cast<std::int64_t>(rng.below(60)) - 5;
            const auto px = static_cast<Price4>(1'000'000 + (bid ? -offset : offset) * 100);
            const auto sh = static_cast<std::uint32_t>(1 + rng.below(rng.chance(0.1) ? 100'000 : 500));
            const OrderRef r = next_ref++;
            if (rng.chance(0.2))
                w.add(loc, ts, r, bid ? 'B' : 'S', sh, "S", px, "MPID");
            else
                w.add(loc, ts, r, bid ? 'B' : 'S', sh, "S", px);
            ref.insert(r, loc, bid, px, sh);
            live.push_back(r);
        } else {
            const std::size_t i = rng.below(live.size());
            const OrderRef r = live[i];
            const auto o = live_order(i);
            if (u < 0.60) {
                const auto sh = static_cast<std::uint32_t>(1 + rng.below(o.shares));
                w.executed(o.locate, ts, r, sh, match++);
                ref.remove_shares(r, sh, true, o.price);
            } else if (u < 0.65) {
                const auto sh = static_cast<std::uint32_t>(1 + rng.below(o.shares));
                const char printable = rng.chance(0.8) ? 'Y' : 'N';
                const auto px = static_cast<Price4>(o.price + rng.below(3) * 100);
                w.executed_px(o.locate, ts, r, sh, match++, printable, px);
                ref.remove_shares(r, sh, printable == 'Y', px);
            } else if (u < 0.75) {
                const auto sh = static_cast<std::uint32_t>(1 + rng.below(o.shares));
                w.cancel(o.locate, ts, r, sh);
                ref.remove_shares(r, sh, false, 0);
            } else if (u < 0.90) {
                w.del(o.locate, ts, r);
                ref.remove_shares(r, o.shares, false, 0);
            } else {
                const OrderRef nr = next_ref++;
                const auto sh = static_cast<std::uint32_t>(1 + rng.below(1000));
                const auto px = static_cast<Price4>(static_cast<std::int64_t>(o.price) +
                                                    (static_cast<std::int64_t>(rng.below(5)) - 2) * 100);
                w.replace(o.locate, ts, r, nr, sh, px);
                ref.remove_shares(r, o.shares, false, 0);
                ref.insert(nr, o.locate, o.bid, px, sh);
                live.push_back(nr);
            }
            drop_dead();
        }
        if (rng.chance(0.05)) w.trade(locs[rng.below(4)], ts, 0, 'B', 100, "S", 1'000'000, match++);
    }
}

}  // namespace

TEST_CASE(itch_book_builder_matches_reference_on_valid_feeds) {
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        Writer w;
        ReferenceItchBook ref;
        random_feed(seed, 60'000, w, ref, 0.0);
        BookBuilder<PrintTotals> b(1u << 16);
        Parser<BookBuilder<PrintTotals>> p(b);
        CHECK(p.parse(w.buf.data(), w.buf.size()) == w.buf.size());
        CHECK(b.integrity().total() == 0);
        CHECK(b.live_orders() == ref.orders.size());
        for (Locate loc : {Locate{3}, Locate{17}, Locate{250}, Locate{9000}}) CHECK(same_depth(b, ref, loc));
        CHECK(b.listener().shares == ref.printed_shares);
        CHECK(b.listener().notional == ref.printed_notional);
        CHECK(b.book(250).symbol.view() == "S250" && b.find_symbol("S9000") == 9000 && b.find_symbol("NOPE") == 0);
    }
}

TEST_CASE(itch_book_depth_matches_reference_every_message) {
    Writer w;
    ReferenceItchBook ref_final;
    random_feed(99, 8'000, w, ref_final, 0.0);

    // Replay message by message, rebuilding the reference in lockstep from
    // the decoded messages themselves.
    struct Lockstep {
        BookBuilder<PrintTotals> book{1u << 14};
        ReferenceItchBook ref;
        std::uint64_t mismatches = 0;
        std::uint64_t crossed_ref = 0;
        void check(Locate loc) {
            if (!same_depth(book, ref, loc)) ++mismatches;
            if (ref.is_crossed(loc)) ++crossed_ref;
        }
        void on_stock_directory(const StockDirectory& m) { book.on_stock_directory(m); }
        void on_add(const AddOrder& m) {
            book.on_add(m); ref.insert(m.ref, m.header.locate, m.side == 'B', m.price, m.shares); check(m.header.locate);
        }
        void on_executed(const OrderExecuted& m) {
            book.on_executed(m);
            if (ref.known(m.ref, m.header.locate)) ref.remove_shares(m.ref, m.executed_shares, true, ref.orders.at(m.ref).price);
            check(m.header.locate);
        }
        void on_executed_with_price(const OrderExecutedWithPrice& m) {
            book.on_executed_with_price(m);
            if (ref.known(m.ref, m.header.locate)) ref.remove_shares(m.ref, m.executed_shares, m.printable == 'Y', m.execution_price);
            check(m.header.locate);
        }
        void on_cancel(const OrderCancel& m) {
            book.on_cancel(m);
            if (ref.known(m.ref, m.header.locate)) ref.remove_shares(m.ref, m.cancelled_shares, false, 0);
            check(m.header.locate);
        }
        void on_delete(const OrderDelete& m) {
            book.on_delete(m);
            if (ref.known(m.ref, m.header.locate)) ref.remove_shares(m.ref, ref.orders.at(m.ref).shares, false, 0);
            check(m.header.locate);
        }
        void on_replace(const OrderReplace& m) {
            book.on_replace(m);
            if (ref.known(m.original_ref, m.header.locate)) {
                const auto o = ref.orders.at(m.original_ref);
                ref.remove_shares(m.original_ref, o.shares, false, 0);
                ref.insert(m.new_ref, m.header.locate, o.bid, m.price, m.shares);
            }
            check(m.header.locate);
        }
    } ls;
    Parser<Lockstep> p(ls);
    p.parse(w.buf.data(), w.buf.size());
    CHECK(ls.mismatches == 0);
    CHECK(ls.crossed_ref > 0);  // the feed does exercise crossed states
    CHECK(ls.book.crossed().total() == ls.crossed_ref);
}

TEST_CASE(itch_book_builder_counts_every_invalid_event) {
    for (std::uint64_t seed = 11; seed <= 13; ++seed) {
        Writer w;
        ReferenceItchBook ref;
        random_feed(seed, 40'000, w, ref, 0.03);
        BookBuilder<PrintTotals> b(1u << 16);
        Parser<BookBuilder<PrintTotals>> p(b);
        p.parse(w.buf.data(), w.buf.size());
        const Integrity& got = b.integrity();
        CHECK(got.unknown_order == ref.integrity.unknown_order && got.unknown_order > 0);
        CHECK(got.overfill == ref.integrity.overfill && got.overfill > 0);
        CHECK(got.duplicate_order == ref.integrity.duplicate_order && got.duplicate_order > 0);
        CHECK(got.locate_mismatch == ref.integrity.locate_mismatch && got.locate_mismatch > 0);
        CHECK(got.level_mismatch == 0);
        CHECK(b.live_orders() == ref.orders.size());
        for (Locate loc : {Locate{3}, Locate{17}, Locate{250}, Locate{9000}}) CHECK(same_depth(b, ref, loc));
        CHECK(b.listener().shares == ref.printed_shares);
    }
}

TEST_CASE(itch_order_map_handles_adversarial_keys) {
    // Identity indexing is fast for exchange-style increasing references; it
    // must stay correct for any key pattern, including keys that all share a
    // home slot, and across growth.
    for (int pattern = 0; pattern < 3; ++pattern) {
        Rng rng{static_cast<std::uint64_t>(100 + pattern)};
        OrderMap map(8);
        std::unordered_map<OrderRef, std::uint32_t> ref;
        std::vector<OrderRef> keys;
        OrderRef next = 1;
        bool ok = true;
        for (int step = 0; step < 200'000; ++step) {
            if (keys.empty() || rng.chance(0.55)) {
                OrderRef k = 0;
                if (pattern == 0) k = (next += 1 + rng.below(8));                // exchange-like
                if (pattern == 1) k = rng.next();                                 // random
                if (pattern == 2) k = (next++) << 32;                             // every key collides
                OrderEntry* e = map.insert(k);
                const bool fresh = ref.find(k) == ref.end();
                ok = ok && ((e != nullptr) == fresh);
                if (e != nullptr) {
                    e->shares = static_cast<std::uint32_t>(step);
                    ref[k] = static_cast<std::uint32_t>(step);
                    keys.push_back(k);
                }
            } else {
                const std::size_t i = rng.below(keys.size());
                OrderEntry* e = map.find(keys[i]);
                ok = ok && e != nullptr && e->shares == ref[keys[i]];
                if (e != nullptr && rng.chance(0.7)) {
                    map.erase(e);
                    ref.erase(keys[i]);
                    keys[i] = keys.back();
                    keys.pop_back();
                }
            }
            if (pattern == 2 && keys.size() > 2000) {  // keep the colliding cluster bounded
                map.erase(map.find(keys.back()));
                ref.erase(keys.back());
                keys.pop_back();
            }
        }
        CHECK(ok);
        CHECK(map.size() == ref.size());
        CHECK(map.grows() > 0);
        for (const auto& [k, v] : ref) {
            const OrderEntry* e = map.find(k);
            CHECK(e != nullptr && e->shares == v);
        }
        CHECK(map.find(~OrderRef{0}) == nullptr);
    }
}

TEST_CASE(itch_side_book_orders_levels_best_first) {
    SideBook bids(true), asks(false);
    for (Price4 px : {100u, 300u, 200u, 300u}) { bids.add(px, 10); asks.add(px, 10); }
    CHECK(bids.best().price == 300 && bids.best().shares == 20 && bids.best().orders == 2);
    CHECK(asks.best().price == 100);
    CHECK(bids.at(2).price == 100 && asks.at(2).price == 300);
    CHECK(bids.reduce(300, 10, true) && bids.best().orders == 1);
    CHECK(!bids.reduce(250, 1, false));  // missing level
    CHECK(!bids.reduce(300, 11, false));  // more shares than the level holds
    CHECK(bids.reduce(300, 10, true) && bids.best().price == 200);
}
