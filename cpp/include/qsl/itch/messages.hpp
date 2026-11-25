#pragma once
// NASDAQ TotalView-ITCH 5.0 message layouts.
//
// Every message starts with the same 11-byte header:
//   type (1) | stock locate (2) | tracking number (2) | timestamp (6, ns since midnight)
// All integers are big-endian. Prices are unsigned 32-bit with 4 implied
// decimals ($1 = 10000). In the binary file format each message is preceded
// by a 2-byte big-endian length.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace qsl::itch {

using Locate = std::uint16_t;
using OrderRef = std::uint64_t;
using Price4 = std::uint32_t;  ///< price x 10^4
using Timestamp = std::uint64_t;  ///< nanoseconds since midnight

inline constexpr Price4 kPriceScale = 10'000;
inline constexpr Timestamp kNsPerSecond = 1'000'000'000ULL;
inline constexpr Timestamp kNsPerMinute = 60 * kNsPerSecond;

// ---------------------------------------------------------------- byte reads
[[nodiscard]] inline std::uint16_t be16(const unsigned char* p) noexcept {
    return static_cast<std::uint16_t>((static_cast<unsigned>(p[0]) << 8) | p[1]);
}
[[nodiscard]] inline std::uint32_t be32(const unsigned char* p) noexcept {
    return (std::uint32_t{p[0]} << 24) | (std::uint32_t{p[1]} << 16) | (std::uint32_t{p[2]} << 8) | p[3];
}
[[nodiscard]] inline std::uint64_t be48(const unsigned char* p) noexcept {
    return (std::uint64_t{be16(p)} << 32) | be32(p + 2);
}
[[nodiscard]] inline std::uint64_t be64(const unsigned char* p) noexcept {
    return (std::uint64_t{be32(p)} << 32) | be32(p + 4);
}

/// 8-character, right-space-padded ASCII stock symbol.
struct Symbol {
    std::array<char, 8> chars{};

    [[nodiscard]] std::string_view view() const noexcept {
        std::size_t n = chars.size();
        while (n > 0 && chars[n - 1] == ' ') --n;
        return {chars.data(), n};
    }
    static Symbol read(const unsigned char* p) noexcept {
        Symbol s;
        for (std::size_t i = 0; i < s.chars.size(); ++i) s.chars[i] = static_cast<char>(p[i]);
        return s;
    }
};

// ------------------------------------------------------------------ messages
struct Header {
    char type;
    Locate locate;
    std::uint16_t tracking;
    Timestamp timestamp;
};

/// 'S' System Event. Codes: O start of messages, S start of system hours,
/// Q start of market hours, M end of market hours, E end of system hours,
/// C end of messages.
struct SystemEvent {
    Header header;
    char event_code;
};

/// 'R' Stock Directory: maps a stock locate code to its symbol for the day.
struct StockDirectory {
    Header header;
    Symbol stock;
    char market_category;
    char financial_status;
    std::uint32_t round_lot_size;
    char round_lots_only;
    char issue_classification;
    std::array<char, 2> issue_subtype;
    char authenticity;
    char short_sale_threshold;
    char ipo_flag;
    char luld_reference_price_tier;
    char etp_flag;
    std::uint32_t etp_leverage_factor;
    char inverse_indicator;
};

/// 'H' Stock Trading Action. State: H halted, P paused, Q quotation only, T trading.
struct TradingAction {
    Header header;
    Symbol stock;
    char trading_state;
    char reserved;
    std::array<char, 4> reason;
};

/// 'A' Add Order (no attribution) and 'F' Add Order with MPID attribution.
struct AddOrder {
    Header header;
    OrderRef ref;
    char side;  ///< 'B' buy, 'S' sell
    std::uint32_t shares;
    Symbol stock;
    Price4 price;
    std::array<char, 4> attribution;  ///< MPID for 'F', spaces for 'A'
};

/// 'E' Order Executed: a resting order traded at its own price.
struct OrderExecuted {
    Header header;
    OrderRef ref;
    std::uint32_t executed_shares;
    std::uint64_t match_number;
};

/// 'C' Order Executed With Price: traded at a price other than the order's
/// (for example in a cross). Non-printable executions are not tape volume.
struct OrderExecutedWithPrice {
    Header header;
    OrderRef ref;
    std::uint32_t executed_shares;
    std::uint64_t match_number;
    char printable;  ///< 'Y' or 'N'
    Price4 execution_price;
};

/// 'X' Order Cancel: partial cancel of a resting order.
struct OrderCancel {
    Header header;
    OrderRef ref;
    std::uint32_t cancelled_shares;
};

/// 'D' Order Delete: the whole remaining order leaves the book.
struct OrderDelete {
    Header header;
    OrderRef ref;
};

/// 'U' Order Replace: the original order is removed and a new one added with
/// a new reference, the same side, and new size and price (priority is lost).
struct OrderReplace {
    Header header;
    OrderRef original_ref;
    OrderRef new_ref;
    std::uint32_t shares;
    Price4 price;
};

/// 'P' Trade (non-cross): execution against a non-displayed order.
struct Trade {
    Header header;
    OrderRef ref;
    char side;
    std::uint32_t shares;
    Symbol stock;
    Price4 price;
    std::uint64_t match_number;
};

/// 'Q' Cross Trade: the bulk print of an opening, closing, halt or IPO cross.
struct CrossTrade {
    Header header;
    std::uint64_t shares;
    Symbol stock;
    Price4 cross_price;
    std::uint64_t match_number;
    char cross_type;  ///< 'O' opening, 'C' closing, 'H' halt/IPO, 'I' intraday
};

/// 'B' Broken Trade.
struct BrokenTrade {
    Header header;
    std::uint64_t match_number;
};

/// 'I' Net Order Imbalance Indicator.
struct Noii {
    Header header;
    std::uint64_t paired_shares;
    std::uint64_t imbalance_shares;
    char imbalance_direction;
    Symbol stock;
    Price4 far_price;
    Price4 near_price;
    Price4 current_reference_price;
    char cross_type;
    char price_variation_indicator;
};

/// Any other valid message type (Reg SHO, MWCB, LULD collar, RPII, ...),
/// delivered with its header and raw payload.
struct OtherMessage {
    Header header;
    const unsigned char* payload;
    std::size_t length;
};

// -------------------------------------------------------------- length table
/// Exact payload length (excluding the 2-byte frame length) of every
/// ITCH 5.0 message type; 0 means the type is not defined.
inline constexpr std::array<std::uint8_t, 256> kMessageLength = [] {
    std::array<std::uint8_t, 256> len{};
    len['S'] = 12;  // system event
    len['R'] = 39;  // stock directory
    len['H'] = 25;  // stock trading action
    len['Y'] = 20;  // Reg SHO restriction
    len['L'] = 26;  // market participant position
    len['V'] = 35;  // MWCB decline level
    len['W'] = 12;  // MWCB status
    len['K'] = 28;  // IPO quoting period update
    len['J'] = 35;  // LULD auction collar
    len['h'] = 21;  // operational halt
    len['A'] = 36;  // add order
    len['F'] = 40;  // add order with MPID
    len['E'] = 31;  // order executed
    len['C'] = 36;  // order executed with price
    len['X'] = 23;  // order cancel
    len['D'] = 19;  // order delete
    len['U'] = 35;  // order replace
    len['P'] = 44;  // trade (non-cross)
    len['Q'] = 40;  // cross trade
    len['B'] = 19;  // broken trade
    len['I'] = 50;  // NOII
    len['N'] = 20;  // retail price improvement indicator
    len['O'] = 48;  // direct listing with capital raise price discovery
    return len;
}();

inline constexpr std::size_t kHeaderLength = 11;

[[nodiscard]] inline Header read_header(const unsigned char* p) noexcept {
    return Header{static_cast<char>(p[0]), be16(p + 1), be16(p + 3), be48(p + 5)};
}

}  // namespace qsl::itch
