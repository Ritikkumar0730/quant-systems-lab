#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "qsl/pricing/vanilla.hpp"

namespace qsl::pricing {

enum class Tree {
    CoxRossRubinstein,  ///< u = e^{sigma sqrt(dt)}, d = 1/u: first-order, oscillating convergence
    LeisenReimer,       ///< Peizer-Pratt inversion: second-order, smooth convergence (odd steps)
};

/// Peizer-Pratt method 2 inversion: maps a normal quantile z onto a binomial
/// probability for an n-step tree (Leisen & Reimer, 1996). n must be odd.
[[nodiscard]] inline double peizer_pratt_inversion(double z, std::size_t n) noexcept {
    const double nn = static_cast<double>(n);
    const double a = z / (nn + 1.0 / 3.0 + 0.1 / (nn + 1.0));
    const double w = std::exp(-a * a * (nn + 1.0 / 6.0));
    return 0.5 + (z > 0.0 ? 0.5 : -0.5) * std::sqrt(1.0 - w);
}

/// Recombining binomial tree for European or American exercise.
/// O(N^2) time, O(N) memory; node spots come from precomputed powers of u and
/// d so there is no pow() call inside the backward induction.
[[nodiscard]] inline double binomial_price(const Vanilla& o, std::size_t steps,
                                           Tree tree = Tree::LeisenReimer) {
    validate(o);
    if (steps == 0) throw std::invalid_argument("binomial tree needs at least one step");
    const std::size_t n = (tree == Tree::LeisenReimer && steps % 2 == 0) ? steps + 1 : steps;
    const double dt = o.maturity / static_cast<double>(n);
    const double growth = std::exp((o.rate - o.dividend) * dt);

    double u;
    double d;
    double p;
    if (tree == Tree::CoxRossRubinstein) {
        u = std::exp(o.vol * std::sqrt(dt));
        d = 1.0 / u;
        p = (growth - d) / (u - d);
    } else {
        const double sd = o.vol * std::sqrt(o.maturity);
        const double d2 = (std::log(o.spot / o.strike) +
                           (o.rate - o.dividend - 0.5 * o.vol * o.vol) * o.maturity) / sd;
        const double d1 = d2 + sd;
        p = peizer_pratt_inversion(d2, n);
        u = growth * peizer_pratt_inversion(d1, n) / p;
        d = (growth - p * u) / (1.0 - p);
    }
    if (!(p > 0.0 && p < 1.0))
        throw std::domain_error("binomial tree: risk-neutral probability outside (0, 1); use more steps");

    const double disc = std::exp(-o.rate * dt);
    const double pu = disc * p;
    const double pd = disc * (1.0 - p);

    std::vector<double> up_pow(n + 1);
    std::vector<double> down_pow(n + 1);
    up_pow[0] = down_pow[0] = 1.0;
    for (std::size_t k = 1; k <= n; ++k) {
        up_pow[k] = up_pow[k - 1] * u;
        down_pow[k] = down_pow[k - 1] * d;
    }

    std::vector<double> value(n + 1);
    for (std::size_t j = 0; j <= n; ++j)
        value[j] = payoff(o.type, o.spot * up_pow[j] * down_pow[n - j], o.strike);

    const bool american = o.exercise == Exercise::American;
    for (std::size_t i = n; i-- > 0;) {  // step i has nodes j = 0..i
        for (std::size_t j = 0; j <= i; ++j) {
            const double continuation = pd * value[j] + pu * value[j + 1];
            value[j] = american
                           ? std::max(continuation, payoff(o.type, o.spot * up_pow[j] * down_pow[i - j], o.strike))
                           : continuation;
        }
    }
    return value[0];
}

}  // namespace qsl::pricing
