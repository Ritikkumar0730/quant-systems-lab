#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>

#include "qsl/pricing/vanilla.hpp"

namespace qsl::pricing {

/// Heston (1993) stochastic volatility:
///   dS/S = (r - q) dt + sqrt(v) dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2,  d<W1,W2> = rho dt.
struct HestonParams {
    double v0 = 0.04;     ///< initial variance
    double kappa = 1.5;   ///< mean-reversion speed
    double theta = 0.04;  ///< long-run variance
    double xi = 0.5;      ///< volatility of variance ("vol of vol")
    double rho = -0.7;    ///< spot/variance correlation (equity skew when < 0)
};

namespace detail {

/// n-point Gauss-Legendre nodes and weights on [-1, 1] (Newton on P_n).
template <std::size_t N>
struct GaussLegendre {
    std::array<double, N> x{};
    std::array<double, N> w{};
    GaussLegendre() {
        for (std::size_t i = 0; i < (N + 1) / 2; ++i) {
            double z = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (static_cast<double>(N) + 0.5));
            double dp = 0.0;
            for (int iter = 0; iter < 100; ++iter) {
                double p0 = 1.0;
                double p1 = z;
                for (std::size_t k = 2; k <= N; ++k) {
                    const double kk = static_cast<double>(k);
                    const double p2 = ((2.0 * kk - 1.0) * z * p1 - (kk - 1.0) * p0) / kk;
                    p0 = p1;
                    p1 = p2;
                }
                dp = static_cast<double>(N) * (z * p1 - p0) / (z * z - 1.0);
                const double dz = p1 / dp;
                z -= dz;
                if (std::abs(dz) < 1e-16) break;
            }
            x[i] = -z;
            x[N - 1 - i] = z;
            w[i] = w[N - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
        }
    }
};

}  // namespace detail

/// psi(u) = E[exp(i u ln(S_T / F))] with F the forward, in the "little Heston
/// trap" form of Albrecher, Mayer, Schoutens & Tistaert (2007): g uses -d, so
/// the complex logarithm never crosses its branch cut and long maturities
/// stay continuous. Valid for complex u (the pricer evaluates u - i).
[[nodiscard]] inline std::complex<double> heston_normalized_cf(std::complex<double> u, double t,
                                                               const HestonParams& p) {
    using cd = std::complex<double>;
    const cd i(0.0, 1.0);
    const double xi2 = p.xi * p.xi;
    const cd beta = p.kappa - p.rho * p.xi * i * u;
    const cd d = std::sqrt(beta * beta + xi2 * (i * u + u * u));
    const cd g = (beta - d) / (beta + d);
    const cd e = std::exp(-d * t);
    const cd big_c = (p.kappa * p.theta / xi2) * ((beta - d) * t - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    const cd big_d = ((beta - d) / xi2) * ((1.0 - e) / (1.0 - g * e));
    // No i u ln F term: the forward is factored out, ln S_T = ln F + X_T.
    return std::exp(big_c + big_d * p.v0);
}

/// Semi-analytic Heston price via the Gil-Pelaez inversion written as one integral:
///   C = (S e^{-qT} - K e^{-rT}) / 2
///     + (e^{-rT} / pi) int_0^inf Re[ e^{iuk} (F psi(u - i) - K psi(u)) / (iu) ] du,
/// with k = ln(F / K). Integrated with composite 16-point Gauss-Legendre
/// panels until the tail contribution is negligible. Puts via parity.
[[nodiscard]] inline double heston_price(OptionType type, double spot, double strike, double t, double r,
                                         double q, const HestonParams& p) {
    if (!(spot > 0.0) || !(strike > 0.0) || !(t > 0.0) || !(p.v0 >= 0.0) || !(p.kappa > 0.0) ||
        !(p.theta >= 0.0) || !(p.xi > 0.0) || !(p.rho > -1.0 && p.rho < 1.0))
        throw std::invalid_argument("heston: invalid parameters");

    using cd = std::complex<double>;
    static const detail::GaussLegendre<16> gl;
    const cd i(0.0, 1.0);
    const double df = std::exp(-r * t);
    const double fwd = spot * std::exp((r - q) * t);
    const double k = std::log(fwd / strike);

    auto integrand = [&](double u) {
        const cd value = std::exp(i * u * k) * (fwd * heston_normalized_cf(cd(u, -1.0), t, p) -
                                                strike * heston_normalized_cf(cd(u, 0.0), t, p)) / (i * u);
        return value.real();
    };

    // Panels of width ~1/(total stdev) resolve the integrand's natural scale.
    const double total_sd = std::sqrt(std::max(p.v0, p.theta) * t);
    const double width = std::clamp(1.0 / std::max(total_sd, 1e-3), 0.5, 50.0);
    double integral = 0.0;
    int quiet_panels = 0;
    for (int panel = 0; panel < 4000 && quiet_panels < 4; ++panel) {
        const double lo = width * panel;
        const double mid = lo + 0.5 * width;
        double sum = 0.0;
        for (std::size_t j = 0; j < 16; ++j) sum += gl.w[j] * integrand(mid + 0.5 * width * gl.x[j]);
        sum *= 0.5 * width;
        integral += sum;
        quiet_panels = std::abs(sum) < 1e-16 * std::max(1.0, std::abs(integral)) ? quiet_panels + 1 : 0;
    }

    const double call = 0.5 * (spot * std::exp(-q * t) - strike * df) + df / std::numbers::pi * integral;
    if (type == OptionType::Call) return call;
    return call - spot * std::exp(-q * t) + strike * df;  // put-call parity
}

}  // namespace qsl::pricing
