#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "qsl/pricing/vanilla.hpp"

namespace qsl::pricing {

struct FdGrid {
    std::size_t space_steps = 400;     ///< log-spot intervals (rounded up to even)
    std::size_t time_steps = 400;
    double width_sd = 5.0;             ///< half-width in standard deviations of ln S_T
    std::size_t rannacher_steps = 2;   ///< leading steps split into implicit half-steps
};

struct FdResult {
    double price;
    double delta;
    double gamma;
};

namespace detail {

/// Solves l[i] x[i-1] + d[i] x[i] + u[i] x[i+1] = r[i] for i = 0..n-1.
/// With `floor` non-empty it enforces x >= floor using Brennan-Schwartz
/// ordering: eliminate away from the early-exercise region, then
/// back-substitute into it, applying the constraint as each unknown is
/// fixed. That solves the American LCP exactly when the exercise region is
/// one interval touching the boundary (true for vanilla puts and calls).
inline void solve_tridiagonal(std::span<const double> l, std::span<const double> d,
                              std::span<const double> u, std::span<const double> r,
                              std::span<const double> floor, bool exercise_at_low_end,
                              std::span<double> work_d, std::span<double> work_r, std::span<double> x) {
    const std::size_t n = d.size();
    auto project = [&](std::size_t i, double v) { return floor.empty() ? v : std::max(v, floor[i]); };
    if (exercise_at_low_end) {  // eliminate the super-diagonal top-down, substitute bottom-up
        work_d[n - 1] = d[n - 1];
        work_r[n - 1] = r[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            const double m = u[i] / work_d[i + 1];
            work_d[i] = d[i] - m * l[i + 1];
            work_r[i] = r[i] - m * work_r[i + 1];
        }
        x[0] = project(0, work_r[0] / work_d[0]);
        for (std::size_t i = 1; i < n; ++i) x[i] = project(i, (work_r[i] - l[i] * x[i - 1]) / work_d[i]);
    } else {  // classic Thomas: eliminate the sub-diagonal, substitute top-down
        work_d[0] = d[0];
        work_r[0] = r[0];
        for (std::size_t i = 1; i < n; ++i) {
            const double m = l[i] / work_d[i - 1];
            work_d[i] = d[i] - m * u[i - 1];
            work_r[i] = r[i] - m * work_r[i - 1];
        }
        x[n - 1] = project(n - 1, work_r[n - 1] / work_d[n - 1]);
        for (std::size_t i = n - 1; i-- > 0;) x[i] = project(i, (work_r[i] - u[i] * x[i + 1]) / work_d[i]);
    }
}

}  // namespace detail

/// Crank-Nicolson finite differences for the Black-Scholes PDE in x = ln S,
///   dV/dtau = 0.5 sigma^2 V_xx + (r - q - 0.5 sigma^2) V_x - r V,
/// on a uniform grid centred exactly on the spot (no interpolation at S0).
/// Rannacher start-up (implicit half-steps) damps the oscillations that the
/// payoff kink otherwise causes in CN Greeks. American exercise is handled by
/// the Brennan-Schwartz projected solve. Returns price, delta and gamma.
[[nodiscard]] inline FdResult fd_price(const Vanilla& o, const FdGrid& grid = {}) {
    validate(o);
    if (grid.space_steps < 4 || grid.time_steps < 1 || !(grid.width_sd > 0.0))
        throw std::invalid_argument("fd grid too small");

    const std::size_t m = grid.space_steps + (grid.space_steps % 2);  // even: spot on node m/2
    const double sd = o.vol * std::sqrt(o.maturity);
    const double half_width = grid.width_sd * sd + std::abs(std::log(o.strike / o.spot));
    const double h = 2.0 * half_width / static_cast<double>(m);
    const double x0 = std::log(o.spot);

    std::vector<double> spot(m + 1);
    std::vector<double> value(m + 1);
    std::vector<double> exercise(m + 1);
    const double log_k = std::log(o.strike);
    for (std::size_t j = 0; j <= m; ++j) {
        const double xj = x0 + (static_cast<double>(j) - static_cast<double>(m / 2)) * h;
        spot[j] = std::exp(xj);
        exercise[j] = payoff(o.type, spot[j], o.strike);
        // Terminal condition = payoff averaged over the cell [x - h/2, x + h/2]
        // (Pooley, Vetzal & Forsyth 2003). Without it the kink's position
        // between nodes makes the error oscillate instead of falling as h^2.
        const double lo_x = xj - 0.5 * h;
        const double hi_x = xj + 0.5 * h;
        double integral;
        if (o.type == OptionType::Call) {
            const double from = std::max(lo_x, log_k);
            integral = hi_x > from ? std::exp(hi_x) - std::exp(from) - o.strike * (hi_x - from) : 0.0;
        } else {
            const double to = std::min(hi_x, log_k);
            integral = to > lo_x ? o.strike * (to - lo_x) - (std::exp(to) - std::exp(lo_x)) : 0.0;
        }
        value[j] = integral / h;
    }

    const double var = o.vol * o.vol;
    const double mu = o.rate - o.dividend - 0.5 * var;
    const double alpha = 0.5 * var / (h * h);
    const double beta = mu / (2.0 * h);
    const double a = alpha - beta;          // coefficient of V_{j-1}
    const double b = -2.0 * alpha - o.rate; // coefficient of V_j
    const double c = alpha + beta;          // coefficient of V_{j+1}

    const bool american = o.exercise == Exercise::American;
    const bool is_call = o.type == OptionType::Call;
    auto boundary = [&](double tau, std::size_t j) {
        const double fwd = spot[j] * std::exp(-o.dividend * tau) - o.strike * std::exp(-o.rate * tau);
        if (is_call) {
            if (j == 0) return 0.0;
            return american ? std::max(spot[j] - o.strike, fwd) : std::max(fwd, 0.0);
        }
        if (j == m) return 0.0;
        return american ? std::max(o.strike - spot[j], -fwd) : std::max(-fwd, 0.0);
    };

    const std::size_t n_int = m - 1;  // interior nodes 1..m-1
    std::vector<double> lo(n_int), di(n_int), up(n_int), rhs(n_int), x(n_int), wd(n_int), wr(n_int);
    const std::span<const double> floor =
        american ? std::span<const double>(exercise.data() + 1, n_int) : std::span<const double>{};

    auto theta_step = [&](double tau_old, double k, double theta) {
        const double tau_new = tau_old + k;
        const double v0 = boundary(tau_new, 0);
        const double vm = boundary(tau_new, m);
        const double el = -theta * k * a, ed = 1.0 - theta * k * b, eu = -theta * k * c;
        const double ex = (1.0 - theta) * k;
        for (std::size_t i = 0; i < n_int; ++i) {
            const std::size_t j = i + 1;
            lo[i] = el;
            di[i] = ed;
            up[i] = eu;
            rhs[i] = value[j] + ex * (a * value[j - 1] + b * value[j] + c * value[j + 1]);
        }
        rhs[0] -= el * v0;
        rhs[n_int - 1] -= eu * vm;
        detail::solve_tridiagonal(lo, di, up, rhs, floor, !is_call, wd, wr, x);
        value[0] = v0;
        value[m] = vm;
        std::copy(x.begin(), x.end(), value.begin() + 1);
    };

    const double dt = o.maturity / static_cast<double>(grid.time_steps);
    double tau = 0.0;
    for (std::size_t step = 0; step < grid.time_steps; ++step) {
        if (step < grid.rannacher_steps) {
            theta_step(tau, 0.5 * dt, 1.0);
            theta_step(tau + 0.5 * dt, 0.5 * dt, 1.0);
        } else {
            theta_step(tau, dt, 0.5);
        }
        tau += dt;
    }

    const std::size_t k = m / 2;
    const double vx = (value[k + 1] - value[k - 1]) / (2.0 * h);
    const double vxx = (value[k + 1] - 2.0 * value[k] + value[k - 1]) / (h * h);
    return FdResult{value[k], vx / o.spot, (vxx - vx) / (o.spot * o.spot)};
}

}  // namespace qsl::pricing
