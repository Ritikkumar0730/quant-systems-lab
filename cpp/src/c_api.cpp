// C ABI over the C++ layer, consumed from Python through ctypes.
// Exceptions never cross this boundary: every entry point catches and
// returns a status code instead.
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <vector>

#include "qsl/lob/order_book.hpp"
#include "qsl/pricing/binomial.hpp"
#include "qsl/pricing/finite_difference.hpp"
#include "qsl/pricing/heston.hpp"

using namespace qsl;

namespace {

pricing::Vanilla make_vanilla(int is_call, int is_american, double spot, double strike, double t, double r,
                              double q, double vol) {
    return pricing::Vanilla{is_call ? pricing::OptionType::Call : pricing::OptionType::Put,
                            is_american ? pricing::Exercise::American : pricing::Exercise::European,
                            spot, strike, t, r, q, vol};
}

/// Buffers trades until Python drains them.
struct BufferListener {
    std::vector<lob::Trade> trades;
    void on_trade(const lob::Trade& t) { trades.push_back(t); }
};

struct BookHandle {
    lob::OrderBook<BufferListener> book;
    std::size_t read = 0;  // next undrained trade
};

BookHandle* as_book(void* handle) { return static_cast<BookHandle*>(handle); }
lob::Side as_side(int side) { return side == 0 ? lob::Side::Buy : lob::Side::Sell; }

/// Returned instead of a lob::Reject code when an enum argument is out of range.
constexpr int kInvalidArgument = 255;
bool bad_side(int side) { return side != 0 && side != 1; }
bool bad_tif(int tif) { return tif < 0 || tif > 3; }

}  // namespace

extern "C" {

struct qsl_cpp_trade {
    std::uint64_t taker;
    std::uint64_t maker;
    std::int64_t price;
    std::uint64_t qty;
    std::int32_t taker_side;  // 0 buy, 1 sell
};

// ------------------------------------------------------------------ pricing
// Return 0 on success, 1 on invalid input.

int qsl_cpp_binomial(int is_call, int is_american, double spot, double strike, double t, double r, double q,
                     double vol, std::uint64_t steps, int leisen_reimer, double* price) {
    try {
        *price = pricing::binomial_price(make_vanilla(is_call, is_american, spot, strike, t, r, q, vol), steps,
                                         leisen_reimer ? pricing::Tree::LeisenReimer : pricing::Tree::CoxRossRubinstein);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int qsl_cpp_fd(int is_call, int is_american, double spot, double strike, double t, double r, double q, double vol,
               std::uint64_t space_steps, std::uint64_t time_steps, double* price, double* delta, double* gamma) {
    try {
        const auto res = pricing::fd_price(make_vanilla(is_call, is_american, spot, strike, t, r, q, vol),
                                           pricing::FdGrid{space_steps, time_steps});
        *price = res.price;
        *delta = res.delta;
        *gamma = res.gamma;
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int qsl_cpp_heston(int is_call, double spot, double strike, double t, double r, double q, double v0, double kappa,
                   double theta, double xi, double rho, double* price) {
    try {
        *price = pricing::heston_price(is_call ? pricing::OptionType::Call : pricing::OptionType::Put, spot, strike,
                                       t, r, q, pricing::HestonParams{v0, kappa, theta, xi, rho});
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

// --------------------------------------------------------------- order book
// Reject codes follow qsl::lob::Reject (0 = accepted); 255 = bad side/TIF argument.

void* qsl_cpp_book_new(std::int64_t min_price, std::int64_t max_price, std::uint32_t max_orders) {
    try {
        return new BookHandle{lob::OrderBook<BufferListener>(lob::BookConfig{min_price, max_price, max_orders}), 0};
    } catch (const std::exception&) {
        return nullptr;
    }
}

void qsl_cpp_book_free(void* handle) { delete as_book(handle); }

int qsl_cpp_book_add(void* handle, std::uint64_t id, int side, std::int64_t price, std::uint64_t qty, int tif,
                     std::uint64_t* filled, std::uint64_t* resting) {
    if (bad_side(side) || bad_tif(tif)) return kInvalidArgument;
    const auto res = as_book(handle)->book.add(id, as_side(side), price, qty, static_cast<lob::TimeInForce>(tif));
    *filled = res.filled;
    *resting = res.resting;
    return static_cast<int>(res.reject);
}

int qsl_cpp_book_market(void* handle, std::uint64_t id, int side, std::uint64_t qty, std::uint64_t* filled) {
    if (bad_side(side)) return kInvalidArgument;
    const auto res = as_book(handle)->book.market(id, as_side(side), qty);
    *filled = res.filled;
    return static_cast<int>(res.reject);
}

int qsl_cpp_book_cancel(void* handle, std::uint64_t id) {
    return static_cast<int>(as_book(handle)->book.cancel(id));
}

int qsl_cpp_book_reduce(void* handle, std::uint64_t id, std::uint64_t new_qty) {
    return static_cast<int>(as_book(handle)->book.reduce(id, new_qty));
}

int qsl_cpp_book_replace(void* handle, std::uint64_t id, std::int64_t price, std::uint64_t qty,
                         std::uint64_t* filled, std::uint64_t* resting) {
    const auto res = as_book(handle)->book.replace(id, price, qty);
    *filled = res.filled;
    *resting = res.resting;
    return static_cast<int>(res.reject);
}

int qsl_cpp_book_best(void* handle, int side, std::int64_t* price) {
    const auto& book = as_book(handle)->book;
    const auto best = side == 0 ? book.best_bid() : book.best_ask();
    if (!best) return 0;
    *price = *best;
    return 1;
}

std::uint64_t qsl_cpp_book_volume_at(void* handle, int side, std::int64_t price) {
    return as_book(handle)->book.volume_at(as_side(side), price);
}

std::uint64_t qsl_cpp_book_order_count(void* handle) { return as_book(handle)->book.order_count(); }

std::size_t qsl_cpp_book_depth(void* handle, int side, std::int64_t* prices, std::uint64_t* qtys,
                               std::uint32_t* counts, std::size_t cap) {
    std::vector<lob::LevelInfo> levels(cap);
    const std::size_t n = as_book(handle)->book.depth(as_side(side), levels);
    for (std::size_t i = 0; i < n; ++i) {
        prices[i] = levels[i].price;
        qtys[i] = levels[i].qty;
        counts[i] = levels[i].orders;
    }
    return n;
}

std::size_t qsl_cpp_book_drain_trades(void* handle, qsl_cpp_trade* out, std::size_t cap) {
    BookHandle* h = as_book(handle);
    auto& trades = h->book.listener().trades;
    std::size_t n = 0;
    while (n < cap && h->read < trades.size()) {
        const lob::Trade& t = trades[h->read++];
        out[n++] = qsl_cpp_trade{t.taker, t.maker, t.price, t.qty, t.taker_side == lob::Side::Buy ? 0 : 1};
    }
    if (h->read == trades.size()) {
        trades.clear();
        h->read = 0;
    }
    return n;
}

int qsl_cpp_book_check(void* handle) { return as_book(handle)->book.check_invariants() ? 1 : 0; }

}  // extern "C"
