#pragma once

#include <concepts>
#include <cstdint>
#include <string_view>

namespace qsl::lob {

using OrderId = std::uint64_t;
using Price = std::int64_t;  ///< integer ticks: no floating point on the matching path
using Qty = std::uint64_t;

enum class Side : std::uint8_t { Buy, Sell };

[[nodiscard]] constexpr Side opposite(Side side) noexcept {
    return side == Side::Buy ? Side::Sell : Side::Buy;
}

enum class TimeInForce : std::uint8_t {
    GoodTillCancel,     ///< rest any unfilled remainder on the book
    ImmediateOrCancel,  ///< take what is available now, cancel the rest
    FillOrKill,         ///< fill the entire quantity now or do nothing
    PostOnly,           ///< provide liquidity only: rejected if it would trade on arrival
};

enum class Reject : std::uint8_t {
    None,
    DuplicateOrderId,
    InvalidPrice,
    InvalidQuantity,
    UnknownOrder,
    BookFull,
    PostOnlyWouldCross,
    FillOrKillUnfilled,
};

[[nodiscard]] constexpr std::string_view to_string(Reject reject) noexcept {
    switch (reject) {
        case Reject::None: return "none";
        case Reject::DuplicateOrderId: return "duplicate order id";
        case Reject::InvalidPrice: return "price outside the tick ladder";
        case Reject::InvalidQuantity: return "invalid quantity";
        case Reject::UnknownOrder: return "unknown order";
        case Reject::BookFull: return "resting-order capacity exhausted";
        case Reject::PostOnlyWouldCross: return "post-only order would cross";
        case Reject::FillOrKillUnfilled: return "fill-or-kill could not be filled";
    }
    return "unknown";
}

/// One execution. The price is always the resting (maker) order's price.
struct Trade {
    OrderId taker;  ///< aggressor: removed liquidity
    OrderId maker;  ///< resting order: provided liquidity
    Price price;
    Qty qty;
    Side taker_side;
    friend bool operator==(const Trade&, const Trade&) = default;
};

/// Outcome of an order entry or replace.
struct ExecResult {
    Reject reject = Reject::None;
    Qty filled = 0;   ///< executed immediately on arrival
    Qty resting = 0;  ///< left on the book afterwards
    [[nodiscard]] bool ok() const noexcept { return reject == Reject::None; }
    friend bool operator==(const ExecResult&, const ExecResult&) = default;
};

/// Aggregated price level as seen in a market-by-price depth snapshot.
struct LevelInfo {
    Price price;
    Qty qty;
    std::uint32_t orders;
    friend bool operator==(const LevelInfo&, const LevelInfo&) = default;
};

struct BookConfig {
    Price min_price = 1;                  ///< lowest tick on the ladder
    Price max_price = 65'536;             ///< highest tick (inclusive)
    std::uint32_t max_orders = 1u << 20;  ///< resting-order capacity, preallocated
};

/// Receives executions synchronously while the book matches. A listener must
/// not call back into the book that is notifying it.
template <typename T>
concept TradeListener = requires(T& listener, const Trade& trade) { listener.on_trade(trade); };

struct NullListener {
    void on_trade(const Trade&) noexcept {}
};

}  // namespace qsl::lob
