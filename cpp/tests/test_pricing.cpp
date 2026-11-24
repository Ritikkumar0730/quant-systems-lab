// Pricing engine tests. Every engine is checked against something
// independent: the C closed-form BSM kernel, textbook values, a theorem
// (no early exercise of calls without dividends), the other engines, and for
// Heston a Monte Carlo simulation driven by the C random number generator.
#include <cmath>
#include <vector>

#include "qsl/pricing/binomial.hpp"
#include "qsl/pricing/finite_difference.hpp"
#include "qsl/pricing/heston.hpp"
#include "qsl/qsl.h"
#include "testing.hpp"

using namespace qsl::pricing;

namespace {

double bsm(const Vanilla& o) {
    return qsl_bsm_price(o.type == OptionType::Call ? QSL_CALL : QSL_PUT, o.spot, o.strike, o.maturity,
                         o.rate, o.dividend, o.vol);
}

qsl_greeks bsm_greeks(const Vanilla& o) {
    qsl_greeks g{};
    qsl_bsm_greeks(o.type == OptionType::Call ? QSL_CALL : QSL_PUT, o.spot, o.strike, o.maturity, o.rate,
                   o.dividend, o.vol, &g);
    return g;
}

Vanilla option(OptionType type, Exercise ex, double strike, double t = 1.0, double r = 0.05, double q = 0.0,
               double vol = 0.25) {
    return Vanilla{type, ex, 100.0, strike, t, r, q, vol};
}

}  // namespace

// ------------------------------------------------------------------ binomial

TEST_CASE(pricing_binomial_hull_american_put) {
    // Hull: S0 = K = 50, r = 10%, sigma = 40%, T = 5 months, 5-step CRR tree
    // gives an American put worth 4.49.
    // Hull also quotes 30, 50, 100 and 500 steps: 4.263, 4.272, 4.278, 4.283.
    const Vanilla put{OptionType::Put, Exercise::American, 50.0, 50.0, 5.0 / 12.0, 0.10, 0.0, 0.40};
    CHECK_CLOSE(binomial_price(put, 5, Tree::CoxRossRubinstein), 4.49, 0.005);
    CHECK_CLOSE(binomial_price(put, 30, Tree::CoxRossRubinstein), 4.263, 0.0005);
    CHECK_CLOSE(binomial_price(put, 50, Tree::CoxRossRubinstein), 4.272, 0.0005);
    CHECK_CLOSE(binomial_price(put, 100, Tree::CoxRossRubinstein), 4.278, 0.0005);
    CHECK_CLOSE(binomial_price(put, 500, Tree::CoxRossRubinstein), 4.283, 0.0005);
}

TEST_CASE(pricing_binomial_converges_to_black_scholes) {
    for (const OptionType type : {OptionType::Call, OptionType::Put})
        for (const double k : {80.0, 100.0, 120.0}) {
            const Vanilla o = option(type, Exercise::European, k, 1.0, 0.05, 0.02);
            CHECK_CLOSE(binomial_price(o, 801, Tree::LeisenReimer), bsm(o), 2e-6);
            CHECK_CLOSE(binomial_price(o, 2000, Tree::CoxRossRubinstein), bsm(o), 5e-3);
        }
}

TEST_CASE(pricing_leisen_reimer_is_second_order) {
    // Error falls ~4x per doubling of steps (CRR only ~2x, with oscillation).
    const Vanilla o = option(OptionType::Call, Exercise::European, 105.0);
    const double exact = bsm(o);
    const double e1 = std::abs(binomial_price(o, 51) - exact);
    const double e2 = std::abs(binomial_price(o, 101) - exact);
    const double e3 = std::abs(binomial_price(o, 201) - exact);
    CHECK(e1 / e2 > 3.0 && e2 / e3 > 3.0);
}

TEST_CASE(pricing_no_early_exercise_for_calls_without_dividends) {
    // Merton: an American call on a non-dividend-paying stock is never
    // exercised early, so on the same tree it equals the European call.
    for (const double k : {70.0, 100.0, 140.0}) {
        Vanilla am = option(OptionType::Call, Exercise::American, k);
        Vanilla eu = am;
        eu.exercise = Exercise::European;
        CHECK_CLOSE(binomial_price(am, 500), binomial_price(eu, 500), 1e-12);
        // With a dividend yield early exercise has value.
        am.dividend = eu.dividend = 0.08;
        CHECK(binomial_price(am, 500) > binomial_price(eu, 500) + 1e-4 || k == 140.0);
    }
}

TEST_CASE(pricing_american_put_bounds) {
    for (const double k : {80.0, 100.0, 130.0}) {
        const Vanilla am = option(OptionType::Put, Exercise::American, k);
        Vanilla eu = am;
        eu.exercise = Exercise::European;
        const double a = binomial_price(am, 501);
        CHECK(a >= binomial_price(eu, 501));
        CHECK(a >= k - am.spot);  // never below intrinsic
    }
}

// ------------------------------------------------------- finite differences

TEST_CASE(pricing_fd_european_matches_black_scholes_and_greeks) {
    for (const OptionType type : {OptionType::Call, OptionType::Put})
        for (const double k : {80.0, 100.0, 125.0}) {
            const Vanilla o = option(type, Exercise::European, k, 1.0, 0.05, 0.02);
            const FdResult fd = fd_price(o, FdGrid{800, 800});
            const qsl_greeks g = bsm_greeks(o);
            CHECK_CLOSE(fd.price, g.price, 2e-4);
            CHECK_CLOSE(fd.delta, g.delta, 2e-5);
            CHECK_CLOSE(fd.gamma, g.gamma, 1e-6);
        }
}

TEST_CASE(pricing_fd_american_matches_leisen_reimer) {
    for (const double k : {85.0, 100.0, 115.0}) {
        const Vanilla put = option(OptionType::Put, Exercise::American, k, 1.0, 0.06, 0.0, 0.3);
        CHECK_CLOSE(fd_price(put, FdGrid{800, 800}).price, binomial_price(put, 2001), 5e-4);
        const Vanilla call = option(OptionType::Call, Exercise::American, k, 1.0, 0.03, 0.07, 0.3);
        CHECK_CLOSE(fd_price(call, FdGrid{800, 800}).price, binomial_price(call, 2001), 5e-4);
    }
}

TEST_CASE(pricing_fd_is_second_order) {
    // With cell-averaged payoffs and Rannacher start-up the error falls 4x
    // per grid doubling even when the strike sits between nodes.
    for (const double k : {100.0, 105.0, 125.0}) {
        const Vanilla o = option(OptionType::Call, Exercise::European, k, 1.0, 0.05, 0.02);
        const double exact = bsm(o);
        double prev = std::abs(fd_price(o, FdGrid{100, 100}).price - exact);
        for (const std::size_t n : {200u, 400u, 800u}) {
            const double err = std::abs(fd_price(o, FdGrid{n, n}).price - exact);
            CHECK(prev / err > 3.5 && prev / err < 4.5);
            prev = err;
        }
    }
}

// --------------------------------------------------------------------- Heston

TEST_CASE(pricing_heston_characteristic_function_identities) {
    const HestonParams p{0.05, 2.0, 0.04, 0.6, -0.65};
    // psi(0) = 1 (probability measure) and psi(-i) = E[S_T / F] = 1 (martingale).
    const auto at0 = heston_normalized_cf({0.0, 0.0}, 2.0, p);
    const auto at_minus_i = heston_normalized_cf({0.0, -1.0}, 2.0, p);
    CHECK_CLOSE(at0.real(), 1.0, 1e-14);
    CHECK_CLOSE(at0.imag(), 0.0, 1e-14);
    CHECK_CLOSE(at_minus_i.real(), 1.0, 1e-13);
    CHECK_CLOSE(at_minus_i.imag(), 0.0, 1e-13);
}

TEST_CASE(pricing_heston_reduces_to_black_scholes) {
    // With vanishing vol-of-vol and v0 = theta the variance is deterministic.
    const HestonParams p{0.0625, 1.0, 0.0625, 1e-4, 0.0};
    for (const double k : {70.0, 100.0, 140.0})
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            const Vanilla o = option(type, Exercise::European, k, 1.5, 0.03, 0.01, 0.25);
            CHECK_CLOSE(heston_price(type, 100.0, k, 1.5, 0.03, 0.01, p), bsm(o), 1e-6);
        }
}

TEST_CASE(pricing_heston_no_static_arbitrage_across_strikes) {
    const HestonParams p{0.04, 1.5, 0.05, 0.8, -0.8};
    std::vector<double> calls;
    for (double k = 50.0; k <= 160.0; k += 2.0) calls.push_back(heston_price(OptionType::Call, 100, k, 0.5, 0.02, 0.0, p));
    for (std::size_t i = 1; i < calls.size(); ++i) CHECK(calls[i] < calls[i - 1]);  // decreasing in K
    for (std::size_t i = 1; i + 1 < calls.size(); ++i)
        CHECK(calls[i - 1] - 2.0 * calls[i] + calls[i + 1] > -1e-10);  // butterflies >= 0
}

TEST_CASE(pricing_heston_matches_monte_carlo) {
    // Full-truncation Euler (Lord, Koekkoek & van Dijk 2010) with 250 steps,
    // antithetic pairs, normals from the C xoshiro256** generator.
    const HestonParams p{0.04, 2.0, 0.04, 0.5, -0.7};
    const double s0 = 100, k = 100, t = 1.0, r = 0.03, q = 0.0;
    const int steps = 250, pairs = 40000;
    const double dt = t / steps, sq = std::sqrt(dt), rho_c = std::sqrt(1 - p.rho * p.rho);
    qsl_rng rng;
    qsl_rng_seed(&rng, 2025);
    double sum = 0, sum2 = 0;
    for (int n = 0; n < pairs; ++n) {
        double x[2] = {std::log(s0), std::log(s0)}, v[2] = {p.v0, p.v0};
        for (int i = 0; i < steps; ++i) {
            const double z1 = qsl_rng_normal(&rng), z2 = p.rho * z1 + rho_c * qsl_rng_normal(&rng);
            for (int a = 0; a < 2; ++a) {
                const double sign = a == 0 ? 1.0 : -1.0;
                const double vp = std::max(v[a], 0.0);
                x[a] += (r - q - 0.5 * vp) * dt + std::sqrt(vp) * sq * sign * z1;
                v[a] += p.kappa * (p.theta - vp) * dt + p.xi * std::sqrt(vp) * sq * sign * z2;
            }
        }
        const double y = 0.5 * std::exp(-r * t) * (std::max(std::exp(x[0]) - k, 0.0) + std::max(std::exp(x[1]) - k, 0.0));
        sum += y;
        sum2 += y * y;
    }
    const double mean = sum / pairs;
    const double se = std::sqrt((sum2 / pairs - mean * mean) / (pairs - 1));
    const double exact = heston_price(OptionType::Call, s0, k, t, r, q, p);
    CHECK(std::abs(mean - exact) < 4.0 * se + 0.01);  // statistical error + Euler bias
}

TEST_CASE(pricing_rejects_invalid_inputs) {
    bool threw = false;
    try {
        (void)binomial_price(Vanilla{OptionType::Call, Exercise::European, -1.0}, 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)heston_price(OptionType::Call, 100, 100, 1, 0, 0, HestonParams{0.04, 1, 0.04, 0.5, 1.5});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}
