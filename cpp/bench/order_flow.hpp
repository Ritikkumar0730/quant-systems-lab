#pragma once
// Synthetic but realistic order flow shared by the order-book and pipeline
// benchmarks. Generated against a live book so every cancel or replace
// targets an order that is actually resting at that moment.

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "qsl/lob/order_book.hpp"

namespace qsl::bench {

using namespace qsl::lob;

struct Message {
    enum class Kind : std::uint8_t { Add, Cancel, Replace, Market } kind;
    Side side;
    TimeInForce tif;
    OrderId id;
    Price price;
    Qty qty;
};

struct ExecStats {  // order-independent fingerprint of all executions
    std::uint64_t trades = 0;
    std::uint64_t volume = 0;
    std::uint64_t notional = 0;
    void add(const Trade& t) {
        ++trades;
        volume += t.qty;
        notional += t.qty * static_cast<std::uint64_t>(t.price);
    }
    bool operator==(const ExecStats&) const = default;
};

struct StatsListener {
    ExecStats stats;
    void on_trade(const Trade& t) noexcept { stats.add(t); }
};

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    std::uint64_t below(std::uint64_t n) { return next() % n; }
};

inline const BookConfig kConfig{1, 1 << 16, 1u << 21};
constexpr std::size_t kTargetDepth = 20'000;

/// Builds the message stream against a live book so that cancels and
/// replaces always target orders that are actually resting at that moment.
inline std::vector<Message> generate(std::size_t n, std::uint64_t seed, std::size_t& peak_orders) {
    struct Collector {
        std::vector<Trade> trades;
        void on_trade(const Trade& t) { trades.push_back(t); }
    };
    OrderBook<Collector> book(kConfig);
    SplitMix rng{seed};
    std::vector<OrderId> live;
    std::unordered_map<OrderId, std::size_t> where;
    auto remove_live = [&](OrderId id) {
        const auto it = where.find(id);
        if (it == where.end()) return;
        const std::size_t pos = it->second;
        where[live.back()] = pos;
        live[pos] = live.back();
        live.pop_back();
        where.erase(id);
    };

    std::vector<Message> out;
    out.reserve(n);
    Price mid = 1 << 15;
    OrderId next_id = 1;
    peak_orders = 0;
    while (out.size() < n) {
        if (rng.uniform() < 0.02) mid += rng.uniform() < 0.5 ? 1 : -1;
        const double r = rng.uniform();
        Message m{};
        // Mix per message once the book is warm: 45% passive add, 43% cancel,
        // 7% cancel/replace, 5% aggressive (IOC through the touch or market).
        // Below kTargetDepth resting orders the generator only adds, so the
        // book hovers around a realistic depth for a liquid futures contract.
        if (r < 0.45 || live.size() < kTargetDepth) {  // passive add, geometric distance from the touch
            m.kind = Message::Kind::Add;
            m.side = rng.uniform() < 0.5 ? Side::Buy : Side::Sell;
            Price distance = 0;
            while (distance < 100 && rng.uniform() < 0.85) ++distance;
            m.price = m.side == Side::Buy ? mid - 1 - distance : mid + 1 + distance;
            m.qty = 1 + rng.below(100);
            m.tif = TimeInForce::GoodTillCancel;
            m.id = next_id++;
        } else if (r < 0.88) {
            m.kind = Message::Kind::Cancel;
            m.id = live[rng.below(live.size())];
        } else if (r < 0.93) {  // aggressive: IOC through the touch or a market order
            m.side = rng.uniform() < 0.5 ? Side::Buy : Side::Sell;
            m.qty = 1 + rng.below(200);
            m.id = next_id++;
            if (rng.uniform() < 0.5) {
                m.kind = Message::Kind::Market;
            } else {
                m.kind = Message::Kind::Add;
                m.tif = TimeInForce::ImmediateOrCancel;
                m.price = m.side == Side::Buy ? mid + 3 : mid - 3;
            }
        } else {
            m.kind = Message::Kind::Replace;
            m.id = live[rng.below(live.size())];
            const auto info = book.find(m.id);
            m.price = info->price + (rng.uniform() < 0.5 ? 1 : -1);
            m.qty = info->qty;
        }

        ExecResult res{};
        switch (m.kind) {
            case Message::Kind::Add: res = book.add(m.id, m.side, m.price, m.qty, m.tif); break;
            case Message::Kind::Market: res = book.market(m.id, m.side, m.qty); break;
            case Message::Kind::Cancel: book.cancel(m.id); remove_live(m.id); break;
            case Message::Kind::Replace: res = book.replace(m.id, m.price, m.qty); break;
        }
        if (m.kind == Message::Kind::Replace) remove_live(m.id);
        if (res.resting > 0) {
            where[m.id] = live.size();
            live.push_back(m.id);
        }
        for (const Trade& t : book.listener().trades)
            if (!book.find(t.maker)) remove_live(t.maker);
        book.listener().trades.clear();
        peak_orders = std::max(peak_orders, book.order_count());
        out.push_back(m);
    }
    return out;
}

template <typename BookT>
inline void apply(BookT& book, const Message& m) {
    switch (m.kind) {
        case Message::Kind::Add: book.add(m.id, m.side, m.price, m.qty, m.tif); break;
        case Message::Kind::Market: book.market(m.id, m.side, m.qty); break;
        case Message::Kind::Cancel: book.cancel(m.id); break;
        case Message::Kind::Replace: book.replace(m.id, m.price, m.qty); break;
    }
}

}  // namespace qsl::bench
