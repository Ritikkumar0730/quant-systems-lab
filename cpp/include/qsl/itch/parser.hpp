#pragma once
// Zero-copy ITCH 5.0 parser for the length-prefixed binary file format.
//
// The parser walks a byte buffer, checks each frame's length against the
// spec, decodes the message into a plain struct on the stack and calls the
// matching handler method. Handlers implement only the callbacks they need:
// a callback that is not declared is skipped at compile time (C++20
// requires-expressions), so there are no virtual calls and no dead work.

#include <array>
#include <cstddef>
#include <cstdint>

#include "qsl/itch/messages.hpp"

namespace qsl::itch {

struct ParseStats {
    std::uint64_t messages = 0;      ///< well-formed messages delivered
    std::uint64_t bytes = 0;         ///< bytes consumed, including frame lengths
    std::uint64_t unknown_type = 0;  ///< frames with an undefined type byte (skipped)
    std::uint64_t bad_length = 0;    ///< frames whose length disagrees with the spec (skipped)
    std::array<std::uint64_t, 256> by_type{};
};

namespace detail {

inline Symbol symbol_at(const unsigned char* p) noexcept { return Symbol::read(p); }

template <std::size_t N>
std::array<char, N> chars_at(const unsigned char* p) noexcept {
    std::array<char, N> out{};
    for (std::size_t i = 0; i < N; ++i) out[i] = static_cast<char>(p[i]);
    return out;
}

inline char ch(const unsigned char* p) noexcept { return static_cast<char>(*p); }

}  // namespace detail

template <class Handler>
class Parser {
public:
    explicit Parser(Handler& handler) noexcept : handler_(handler) {}

    /// Parses every complete frame in [data, data + size) and returns the
    /// number of bytes consumed. A trailing partial frame is left unconsumed,
    /// so a streaming caller keeps those bytes and appends the next chunk.
    /// Stops early once `max_frames` frames have been seen in total.
    std::size_t parse(const unsigned char* data, std::size_t size,
                      std::uint64_t max_frames = ~std::uint64_t{0}) {
        std::size_t pos = 0;
        // Handlers that declare prefetch(payload, length, stage) see each frame
        // before it is dispatched, so they can start loading the memory it will
        // touch while earlier frames are processed.
        // Two stages: stage 0 runs kPrefetchDistance frames ahead (load what the
        // frame names directly), stage 1 half as far ahead (load memory whose
        // address is only known once stage 0's loads have arrived).
        std::size_t far = 0, near = 0;
        if constexpr (kPrefetches) {
            for (int i = 0; i < kPrefetchDistance && next_frame(data, size, far, 0); ++i) {}
            for (int i = 0; i < kPrefetchDistance / 2 && next_frame(data, size, near, 1); ++i) {}
        }
        while (size - pos >= 2 && frames_ < max_frames) {
            const std::size_t length = be16(data + pos);
            if (size - pos - 2 < length) break;  // partial frame
            if constexpr (kPrefetches) {
                next_frame(data, size, far, 0);
                next_frame(data, size, near, 1);
            }
            const unsigned char* msg = data + pos + 2;
            pos += 2 + length;
            ++frames_;
            dispatch(msg, length);
        }
        stats_.bytes += pos;
        return pos;
    }

    /// Decodes one message payload (without its frame length).
    void dispatch(const unsigned char* p, std::size_t length) {
        if (length == 0) {
            ++stats_.bad_length;
            return;
        }
        const auto type = static_cast<unsigned char>(p[0]);
        const std::size_t expected = kMessageLength[type];
        if (expected == 0) {
            ++stats_.unknown_type;
            return;
        }
        if (length != expected) {
            ++stats_.bad_length;
            return;
        }
        ++stats_.messages;
        ++stats_.by_type[type];

        using namespace detail;
        const Header h = read_header(p);
        switch (type) {
            case 'S':
                call_system_event(SystemEvent{h, ch(p + 11)});
                break;
            case 'R':
                call_stock_directory(StockDirectory{h, symbol_at(p + 11), ch(p + 19), ch(p + 20), be32(p + 21),
                                                    ch(p + 25), ch(p + 26), chars_at<2>(p + 27), ch(p + 29),
                                                    ch(p + 30), ch(p + 31), ch(p + 32), ch(p + 33), be32(p + 34),
                                                    ch(p + 38)});
                break;
            case 'H':
                call_trading_action(TradingAction{h, symbol_at(p + 11), ch(p + 19), ch(p + 20), chars_at<4>(p + 21)});
                break;
            case 'A':
                call_add(AddOrder{h, be64(p + 11), ch(p + 19), be32(p + 20), symbol_at(p + 24), be32(p + 32),
                                  {' ', ' ', ' ', ' '}});
                break;
            case 'F':
                call_add(AddOrder{h, be64(p + 11), ch(p + 19), be32(p + 20), symbol_at(p + 24), be32(p + 32),
                                  chars_at<4>(p + 36)});
                break;
            case 'E':
                call_executed(OrderExecuted{h, be64(p + 11), be32(p + 19), be64(p + 23)});
                break;
            case 'C':
                call_executed_with_price(
                    OrderExecutedWithPrice{h, be64(p + 11), be32(p + 19), be64(p + 23), ch(p + 31), be32(p + 32)});
                break;
            case 'X':
                call_cancel(OrderCancel{h, be64(p + 11), be32(p + 19)});
                break;
            case 'D':
                call_delete(OrderDelete{h, be64(p + 11)});
                break;
            case 'U':
                call_replace(OrderReplace{h, be64(p + 11), be64(p + 19), be32(p + 27), be32(p + 31)});
                break;
            case 'P':
                call_trade(Trade{h, be64(p + 11), ch(p + 19), be32(p + 20), symbol_at(p + 24), be32(p + 32),
                                 be64(p + 36)});
                break;
            case 'Q':
                call_cross_trade(CrossTrade{h, be64(p + 11), symbol_at(p + 19), be32(p + 27), be64(p + 31), ch(p + 39)});
                break;
            case 'B':
                call_broken_trade(BrokenTrade{h, be64(p + 11)});
                break;
            case 'I':
                call_noii(Noii{h, be64(p + 11), be64(p + 19), ch(p + 27), symbol_at(p + 28), be32(p + 36),
                               be32(p + 40), be32(p + 44), ch(p + 48), ch(p + 49)});
                break;
            default:
                call_other(OtherMessage{h, p, length});
                break;
        }
    }

    [[nodiscard]] const ParseStats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::uint64_t frames() const noexcept { return frames_; }

private:
    static constexpr bool kPrefetches =
        requires(Handler& hd, const unsigned char* p, std::size_t n) { hd.prefetch(p, n, 0); };
    static constexpr int kPrefetchDistance = 16;

    /// Advances `ahead` past one complete frame and hands it to the handler's
    /// prefetch. Returns false when no complete frame remains.
    bool next_frame(const unsigned char* data, std::size_t size, std::size_t& ahead, int stage) {
        if (size - ahead < 2) return false;
        const std::size_t length = be16(data + ahead);
        if (size - ahead - 2 < length) return false;
        if constexpr (kPrefetches) handler_.prefetch(data + ahead + 2, length, stage);
        ahead += 2 + length;
        return true;
    }

    // Each call_* forwards to the handler only if it declares that callback.
#define QSL_ITCH_FORWARD(fn, Msg, method)                                   \
    void fn(const Msg& m) {                                                 \
        if constexpr (requires(Handler& hd, const Msg& mm) { hd.method(mm); }) \
            handler_.method(m);                                             \
    }
    QSL_ITCH_FORWARD(call_system_event, SystemEvent, on_system_event)
    QSL_ITCH_FORWARD(call_stock_directory, StockDirectory, on_stock_directory)
    QSL_ITCH_FORWARD(call_trading_action, TradingAction, on_trading_action)
    QSL_ITCH_FORWARD(call_add, AddOrder, on_add)
    QSL_ITCH_FORWARD(call_executed, OrderExecuted, on_executed)
    QSL_ITCH_FORWARD(call_executed_with_price, OrderExecutedWithPrice, on_executed_with_price)
    QSL_ITCH_FORWARD(call_cancel, OrderCancel, on_cancel)
    QSL_ITCH_FORWARD(call_delete, OrderDelete, on_delete)
    QSL_ITCH_FORWARD(call_replace, OrderReplace, on_replace)
    QSL_ITCH_FORWARD(call_trade, Trade, on_trade)
    QSL_ITCH_FORWARD(call_cross_trade, CrossTrade, on_cross_trade)
    QSL_ITCH_FORWARD(call_broken_trade, BrokenTrade, on_broken_trade)
    QSL_ITCH_FORWARD(call_noii, Noii, on_noii)
    QSL_ITCH_FORWARD(call_other, OtherMessage, on_other)
#undef QSL_ITCH_FORWARD

    Handler& handler_;
    ParseStats stats_;
    std::uint64_t frames_ = 0;
};

}  // namespace qsl::itch
