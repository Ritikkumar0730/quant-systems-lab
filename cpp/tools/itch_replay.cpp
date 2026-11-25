// Replays a NASDAQ TotalView-ITCH 5.0 file through the parser and book builder.
//
//   gzip -dc 12302019.NASDAQ_ITCH50.gz | itch_replay - [options]
//
// Options:
//   --reserve N         initial live-order capacity; the store grows as needed (default 1,000,000)
//   --limit N           stop after N frames
//   --checksum          print per-type field checksums (cross-checked by scripts/itch_crosscheck.py)
//   --bars FILE --top K write regular-hours minute bars for the K most traded symbols
//   --crosses FILE      write every symbol's opening / closing cross and volumes
//   --bench-bytes N     load the first N bytes into memory and time parse-only vs parse + book
//
// Prints message counts, book integrity counters and throughput. Processing
// time excludes reading and decompression, which run in a separate process.

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qsl/itch/book.hpp"
#include "qsl/itch/parser.hpp"

using namespace qsl::itch;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kMinutes = 390;  // 09:30 to 16:00
constexpr Timestamp kOpen = (9 * 60 + 30) * kNsPerMinute;
constexpr Timestamp kClose = 16 * 60 * kNsPerMinute;

struct Bar {
    std::uint64_t volume = 0;
    double notional = 0;  // dollars
    Price4 last = 0;
};

struct StockStats {
    std::uint64_t continuous_regular = 0;  // shares traded 09:30-16:00 outside crosses
    std::uint64_t total = 0;               // all printed shares including crosses and extended hours
    Price4 open_cross_px = 0, close_cross_px = 0;
    std::uint64_t open_cross_shares = 0, close_cross_shares = 0;
    int bars = -1;  // index into bar storage, allocated on first regular-hours print
};

/// Collects per-stock volumes, crosses and minute bars from the book builder's prints.
struct BarListener {
    std::vector<StockStats> stocks = std::vector<StockStats>(65536);
    std::vector<std::array<Bar, kMinutes>> bars;
    std::uint64_t prints = 0, printed_shares = 0;

    void on_print(Locate loc, Timestamp ts, Price4 px, std::uint64_t shares, Fill fill, char cross_type) {
        if (fill == Fill::NonPrintable) return;  // already printed by a cross
        ++prints;
        printed_shares += shares;
        StockStats& s = stocks[loc];
        s.total += shares;
        if (fill == Fill::Cross) {
            if (cross_type == 'O') { s.open_cross_px = px; s.open_cross_shares += shares; }
            if (cross_type == 'C') { s.close_cross_px = px; s.close_cross_shares += shares; }
            return;
        }
        if (ts < kOpen || ts >= kClose) return;
        if (s.bars < 0) {
            s.bars = static_cast<int>(bars.size());
            bars.emplace_back();
        }
        Bar& b = bars[static_cast<std::size_t>(s.bars)][static_cast<std::size_t>((ts - kOpen) / kNsPerMinute)];
        b.volume += shares;
        b.notional += static_cast<double>(shares) * px / kPriceScale;
        b.last = px;
        s.continuous_regular += shares;
    }
};

/// Wrapping sums of decoded fields, so an independent decoder can confirm
/// that every field was read from the right offset.
struct Checksums {
    std::uint64_t ts = 0, locate = 0;
    std::uint64_t add_ref = 0, add_shares = 0, add_price = 0, add_buys = 0;
    std::uint64_t exec_shares = 0, exec_match = 0;
    std::uint64_t execpx_shares = 0, execpx_price = 0, execpx_printable = 0;
    std::uint64_t cancel_shares = 0, delete_ref = 0;
    std::uint64_t replace_new_ref = 0, replace_shares = 0, replace_price = 0;
    std::uint64_t trade_shares = 0, trade_price = 0, cross_shares = 0, cross_price = 0;
    std::uint64_t directory_round_lot = 0;

    void header(const Header& h) { ts += h.timestamp; locate += h.locate; }

    void print() const {
        const std::pair<const char*, std::uint64_t> rows[] = {
            {"ts", ts}, {"locate", locate}, {"add_ref", add_ref}, {"add_shares", add_shares},
            {"add_price", add_price}, {"add_buys", add_buys}, {"exec_shares", exec_shares},
            {"exec_match", exec_match}, {"execpx_shares", execpx_shares}, {"execpx_price", execpx_price},
            {"execpx_printable", execpx_printable}, {"cancel_shares", cancel_shares}, {"delete_ref", delete_ref},
            {"replace_new_ref", replace_new_ref}, {"replace_shares", replace_shares},
            {"replace_price", replace_price}, {"trade_shares", trade_shares}, {"trade_price", trade_price},
            {"cross_shares", cross_shares}, {"cross_price", cross_price},
            {"directory_round_lot", directory_round_lot}};
        for (auto [k, v] : rows) std::printf("checksum %s %" PRIu64 "\n", k, v);
    }
};

/// Parser handler: forwards to the book builder and updates checksums.
struct Replay {
    BookBuilder<BarListener> book;
    Checksums sums;
    bool checksum = false;
    std::array<Timestamp, 256> system_events{};

    explicit Replay(std::size_t max_orders) : book(max_orders) {}

    void on_system_event(const SystemEvent& m) {
        if (checksum) sums.header(m.header);
        system_events[static_cast<unsigned char>(m.event_code)] = m.header.timestamp;
    }
    void on_stock_directory(const StockDirectory& m) {
        if (checksum) { sums.header(m.header); sums.directory_round_lot += m.round_lot_size; }
        book.on_stock_directory(m);
    }
    void on_trading_action(const TradingAction& m) { book.on_trading_action(m); }
    void on_add(const AddOrder& m) {
        if (checksum) {
            sums.header(m.header);
            sums.add_ref += m.ref; sums.add_shares += m.shares; sums.add_price += m.price; sums.add_buys += m.side == 'B';
        }
        book.on_add(m);
    }
    void on_executed(const OrderExecuted& m) {
        if (checksum) { sums.header(m.header); sums.exec_shares += m.executed_shares; sums.exec_match += m.match_number; }
        book.on_executed(m);
    }
    void on_executed_with_price(const OrderExecutedWithPrice& m) {
        if (checksum) {
            sums.header(m.header);
            sums.execpx_shares += m.executed_shares; sums.execpx_price += m.execution_price;
            sums.execpx_printable += m.printable == 'Y';
        }
        book.on_executed_with_price(m);
    }
    void on_cancel(const OrderCancel& m) {
        if (checksum) { sums.header(m.header); sums.cancel_shares += m.cancelled_shares; }
        book.on_cancel(m);
    }
    void on_delete(const OrderDelete& m) {
        if (checksum) { sums.header(m.header); sums.delete_ref += m.ref; }
        book.on_delete(m);
    }
    void on_replace(const OrderReplace& m) {
        if (checksum) {
            sums.header(m.header);
            sums.replace_new_ref += m.new_ref; sums.replace_shares += m.shares; sums.replace_price += m.price;
        }
        book.on_replace(m);
    }
    void on_trade(const Trade& m) {
        if (checksum) { sums.header(m.header); sums.trade_shares += m.shares; sums.trade_price += m.price; }
        book.on_trade(m);
    }
    void on_cross_trade(const CrossTrade& m) {
        if (checksum) { sums.header(m.header); sums.cross_shares += m.shares; sums.cross_price += m.cross_price; }
        book.on_cross_trade(m);
    }
    void on_broken_trade(const BrokenTrade& m) {
        if (checksum) sums.header(m.header);
        book.on_broken_trade(m);
    }
};

/// Decode-only benchmark handler: reads every field of every order message
/// into checksums, so the compiler cannot discard the decoding work.
struct DecodeOnly {
    Checksums s;
    void on_stock_directory(const StockDirectory& m) { s.header(m.header); s.directory_round_lot += m.round_lot_size; }
    void on_add(const AddOrder& m) {
        s.header(m.header); s.add_ref += m.ref; s.add_shares += m.shares; s.add_price += m.price;
        s.add_buys += m.side == 'B';
    }
    void on_executed(const OrderExecuted& m) { s.header(m.header); s.exec_shares += m.executed_shares; s.exec_match += m.match_number; }
    void on_executed_with_price(const OrderExecutedWithPrice& m) {
        s.header(m.header); s.execpx_shares += m.executed_shares; s.execpx_price += m.execution_price;
    }
    void on_cancel(const OrderCancel& m) { s.header(m.header); s.cancel_shares += m.cancelled_shares; }
    void on_delete(const OrderDelete& m) { s.header(m.header); s.delete_ref += m.ref; }
    void on_replace(const OrderReplace& m) {
        s.header(m.header); s.replace_new_ref += m.new_ref; s.replace_shares += m.shares; s.replace_price += m.price;
    }
    void on_trade(const Trade& m) { s.header(m.header); s.trade_shares += m.shares; s.trade_price += m.price; }
    void on_cross_trade(const CrossTrade& m) { s.header(m.header); s.cross_shares += m.shares; }
};

struct Options {
    std::string input = "-";
    std::size_t reserve = 1'000'000;
    std::uint64_t limit = ~std::uint64_t{0};
    bool checksum = false;
    std::string bars_path, crosses_path;
    std::size_t top = 50;
    std::size_t bench_bytes = 0;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: itch_replay FILE|- [--reserve N] [--limit N] [--checksum]\n"
                 "                  [--bars FILE --top K] [--crosses FILE] [--bench-bytes N]\n");
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options o;
    if (argc < 2) usage();
    o.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--reserve") o.reserve = std::strtoull(value(), nullptr, 10);
        else if (a == "--limit") o.limit = std::strtoull(value(), nullptr, 10);
        else if (a == "--checksum") o.checksum = true;
        else if (a == "--bars") o.bars_path = value();
        else if (a == "--top") o.top = std::strtoull(value(), nullptr, 10);
        else if (a == "--crosses") o.crosses_path = value();
        else if (a == "--bench-bytes") o.bench_bytes = std::strtoull(value(), nullptr, 10);
        else usage();
    }
    return o;
}

FILE* open_input(const std::string& path) {
    FILE* f = path == "-" ? stdin : std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::perror(path.c_str());
        std::exit(1);
    }
    return f;
}

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string hhmmss(Timestamp ts) {
    if (ts == 0) return "-";
    const std::uint64_t s = ts / kNsPerSecond;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%02" PRIu64 ":%02" PRIu64 ":%02" PRIu64 ".%06" PRIu64, s / 3600, s / 60 % 60,
                  s % 60, ts % kNsPerSecond / 1000);
    return buf;
}

int run_bench(const Options& opt) {
    FILE* f = open_input(opt.input);
    std::vector<unsigned char> data(opt.bench_bytes);
    const std::size_t got = std::fread(data.data(), 1, data.size(), f);
    data.resize(got);

    double best_parse = 1e300, best_book = 1e300;
    std::uint64_t sink = 0;  // keeps the decode results observable
    std::uint64_t messages = 0;
    std::size_t used = 0;
    for (int rep = 0; rep < 3; ++rep) {
        DecodeOnly decoder;
        Parser<DecodeOnly> p(decoder);
        auto t0 = Clock::now();
        used = p.parse(data.data(), data.size());
        best_parse = std::min(best_parse, seconds(Clock::now() - t0));
        messages = p.stats().messages;
        sink ^= decoder.s.ts ^ decoder.s.add_price;

        auto builder = std::make_unique<BookBuilder<>>(opt.reserve);
        Parser<BookBuilder<>> pb(*builder);
        t0 = Clock::now();
        pb.parse(data.data(), data.size());
        best_book = std::min(best_book, seconds(Clock::now() - t0));
        if (builder->integrity().total() != 0) {
            std::fprintf(stderr, "integrity errors during benchmark\n");
            return 1;
        }
    }
    const double mb = static_cast<double>(used) / 1e6;
    std::printf("In-memory replay of the first %.0f MB (%" PRIu64 " messages), best of 3\n", mb, messages);
    std::printf("  parse + decode every field       %7.1f M msgs/s  %6.2f ns/msg  %6.0f MB/s\n",
                static_cast<double>(messages) / best_parse / 1e6, best_parse / static_cast<double>(messages) * 1e9,
                mb / best_parse);
    std::printf("  parse + full book rebuild        %7.1f M msgs/s  %6.2f ns/msg  %6.0f MB/s\n",
                static_cast<double>(messages) / best_book / 1e6, best_book / static_cast<double>(messages) * 1e9,
                mb / best_book);
    std::printf("  (decode checksum %016" PRIx64 ")\n", sink);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    if (opt.bench_bytes > 0) return run_bench(opt);

    FILE* f = open_input(opt.input);
    auto replay = std::make_unique<Replay>(opt.reserve);
    replay->checksum = opt.checksum;
    Parser<Replay> parser(*replay);

    std::vector<unsigned char> buf(std::size_t{8} << 20);
    std::size_t have = 0;
    Clock::duration processing{};
    const auto wall0 = Clock::now();
    while (parser.frames() < opt.limit) {
        const std::size_t n = std::fread(buf.data() + have, 1, buf.size() - have, f);
        if (n == 0) break;
        have += n;
        const auto t0 = Clock::now();
        const std::size_t used = parser.parse(buf.data(), have, opt.limit);
        processing += Clock::now() - t0;
        std::memmove(buf.data(), buf.data() + used, have - used);
        have -= used;
    }
    const double wall = seconds(Clock::now() - wall0);
    const double proc = seconds(processing);

    const ParseStats& st = parser.stats();
    const auto& book = replay->book;
    const Integrity& ig = book.integrity();
    std::size_t listed = 0;
    for (std::size_t i = 1; i < 65536; ++i) listed += book.book(static_cast<Locate>(i)).listed;

    std::printf("ITCH 5.0 replay\n");
    std::printf("  bytes                 %15" PRIu64 "\n", st.bytes);
    std::printf("  messages              %15" PRIu64 "\n", st.messages);
    std::printf("  unknown type / bad length / trailing bytes  %" PRIu64 " / %" PRIu64 " / %zu\n", st.unknown_type,
                st.bad_length, have);
    std::printf("  stocks listed         %15zu\n", listed);
    std::printf("  messages by type     ");
    int col = 0;
    for (int t = 0; t < 256; ++t) {
        if (st.by_type[static_cast<std::size_t>(t)] == 0) continue;
        std::printf("%s %c %12" PRIu64, col % 4 == 0 ? "\n   " : "   ", t, st.by_type[static_cast<std::size_t>(t)]);
        ++col;
    }
    std::printf("\n");
    std::printf("Session (system events): start %s  open %s  close %s  end %s\n",
                hhmmss(replay->system_events['O']).c_str(), hhmmss(replay->system_events['Q']).c_str(),
                hhmmss(replay->system_events['M']).c_str(), hhmmss(replay->system_events['C']).c_str());
    std::printf("Book integrity (all should be 0)\n");
    std::printf("  unknown order %" PRIu64 "  duplicate %" PRIu64 "  overfill %" PRIu64 "  locate mismatch %" PRIu64
                "  level mismatch %" PRIu64 "\n",
                ig.unknown_order, ig.duplicate_order, ig.overfill, ig.locate_mismatch, ig.level_mismatch);
    std::printf("Book state\n");
    std::printf("  peak live orders %zu, live at end %zu (order map %zu slots, grew %zu times)\n",
                book.peak_live_orders(), book.live_orders(), book.orders().capacity(), book.orders().grows());
    const CrossedCounts& cx = book.crossed();
    std::printf("  updates leaving a crossed book: %" PRIu64 " while trading in regular hours, %" PRIu64
                " while halted/paused/quote-only, %" PRIu64 " outside regular hours\n",
                cx.trading_regular, cx.not_trading, cx.outside_regular);
    std::printf("  printed executions %" PRIu64 " (%" PRIu64 " shares), broken trades %" PRIu64 "\n",
                replay->book.listener().prints, replay->book.listener().printed_shares, book.broken_trades());
    std::printf("Throughput\n");
    std::printf("  processing %.2f s (%.1f M msgs/s, %.1f ns/msg); wall %.2f s including read/decompress\n", proc,
                static_cast<double>(st.messages) / proc / 1e6, proc / static_cast<double>(st.messages) * 1e9, wall);
    if (opt.checksum) replay->sums.print();

    const BarListener& bl = replay->book.listener();
    if (!opt.crosses_path.empty()) {
        FILE* out = std::fopen(opt.crosses_path.c_str(), "w");
        if (out == nullptr) { std::perror(opt.crosses_path.c_str()); return 1; }
        std::fprintf(out, "symbol,open_cross_price,open_cross_shares,close_cross_price,close_cross_shares,"
                          "continuous_regular_shares,total_shares\n");
        for (std::size_t i = 1; i < 65536; ++i) {
            const StockStats& s = bl.stocks[i];
            if (!book.book(static_cast<Locate>(i)).listed || s.total == 0) continue;
            const std::string sym(book.book(static_cast<Locate>(i)).symbol.view());
            std::fprintf(out, "%s,%.4f,%" PRIu64 ",%.4f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", sym.c_str(),
                         s.open_cross_px / 1e4, s.open_cross_shares, s.close_cross_px / 1e4, s.close_cross_shares,
                         s.continuous_regular, s.total);
        }
        std::fclose(out);
    }
    if (!opt.bars_path.empty()) {
        std::vector<std::size_t> order;
        for (std::size_t i = 1; i < 65536; ++i)
            if (bl.stocks[i].bars >= 0) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return bl.stocks[a].continuous_regular > bl.stocks[b].continuous_regular;
        });
        order.resize(std::min(order.size(), opt.top));
        FILE* out = std::fopen(opt.bars_path.c_str(), "w");
        if (out == nullptr) { std::perror(opt.bars_path.c_str()); return 1; }
        std::fprintf(out, "symbol,minute,time,volume,notional,last_price\n");
        for (std::size_t i : order) {
            const std::string sym(book.book(static_cast<Locate>(i)).symbol.view());
            const auto& bars = bl.bars[static_cast<std::size_t>(bl.stocks[i].bars)];
            for (int m = 0; m < kMinutes; ++m) {
                const Bar& b = bars[static_cast<std::size_t>(m)];
                const int clock = 9 * 60 + 30 + m;
                std::fprintf(out, "%s,%d,%02d:%02d,%" PRIu64 ",%.2f,%.4f\n", sym.c_str(), m, clock / 60, clock % 60,
                             b.volume, b.notional, b.last / 1e4);
            }
        }
        std::fclose(out);
    }
    return ig.total() == 0 ? 0 : 3;
}
