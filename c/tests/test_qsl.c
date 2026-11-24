/* Unit tests for the C kernels: textbook values, identities, finite-difference
 * Greeks, statistical checks and round trips. Python tests add scipy and
 * QuantLib cross-checks on top of these. */
#include "qsl/qsl.h"
#include "unit.h"

#include <math.h>
#include <stdint.h>

static qsl_date ymd(int32_t y, int32_t m, int32_t d) { return (qsl_date){y, m, d}; }

/* ----------------------------------------------------------------- normal */

static void test_normal_known_values(void) {
    CHECK_CLOSE(qsl_norm_pdf(0.0), 0.3989422804014327, 1e-16);
    CHECK_CLOSE(qsl_norm_cdf(0.0), 0.5, 0.0);
    CHECK_CLOSE(qsl_norm_cdf(1.0), 0.8413447460685429, 2e-16);
    CHECK_CLOSE(qsl_norm_cdf(-1.0), 0.15865525393145707, 1e-16);
    CHECK_CLOSE(qsl_norm_inv(0.975), 1.959963984540054, 1e-14);
    CHECK_CLOSE(qsl_norm_inv(0.5), 0.0, 0.0);
    CHECK(isinf(qsl_norm_inv(0.0)) && qsl_norm_inv(0.0) < 0.0);
    CHECK(isinf(qsl_norm_inv(1.0)) && qsl_norm_inv(1.0) > 0.0);
    CHECK(isnan(qsl_norm_inv(-0.1)) && isnan(qsl_norm_inv(1.1)) && isnan(qsl_norm_inv(NAN)));
}

static void test_normal_inverse_roundtrip(void) {
    /* Phi(Phi^-1(p)) = p from p = 1e-300 up to 0.5; the relative error budget
     * grows like x^2 * eps in the far tail, hence 1e-12. */
    double worst = 0.0;
    for (int i = 0; i <= 3000; ++i) {
        const double p = pow(10.0, -300.0 + 0.1 * i);
        if (p > 0.5) break;
        const double x = qsl_norm_inv(p);
        worst = fmax(worst, fabs(qsl_norm_cdf(x) - p) / p);
    }
    CHECK(worst < 1e-12);
    /* Exact antisymmetry on the upper half, where 1 - p is exact. */
    for (double p = 0.5; p < 1.0; p += 0.0137) CHECK(qsl_norm_inv(p) == -qsl_norm_inv(1.0 - p));
    /* Monotone on a fine grid. */
    double prev = -INFINITY;
    for (int i = 1; i < 10000; ++i) {
        const double x = qsl_norm_inv(i / 10000.0);
        CHECK(x > prev);
        prev = x;
    }
}

/* ------------------------------------------------------------------- BSM */

static void test_bsm_hull_example(void) {
    /* Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%,
     * sigma=20%, T=0.5 gives c = 4.76 and p = 0.81. */
    CHECK_CLOSE(qsl_bsm_price(QSL_CALL, 42, 40, 0.5, 0.10, 0.0, 0.20), 4.76, 0.005);
    CHECK_CLOSE(qsl_bsm_price(QSL_PUT, 42, 40, 0.5, 0.10, 0.0, 0.20), 0.81, 0.005);
}

static void test_bsm_put_call_parity(void) {
    const double strikes[] = {50, 90, 100, 110, 200};
    const double times[] = {0.01, 0.25, 1.0, 5.0};
    const double vols[] = {0.05, 0.25, 0.8};
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 3; ++k) {
                const double s = 100, kk = strikes[i], t = times[j], r = 0.03, q = 0.015;
                const double c = qsl_bsm_price(QSL_CALL, s, kk, t, r, q, vols[k]);
                const double p = qsl_bsm_price(QSL_PUT, s, kk, t, r, q, vols[k]);
                CHECK_CLOSE(c - p, s * exp(-q * t) - kk * exp(-r * t), 1e-11);
                CHECK(c >= 0.0 && p >= 0.0);
            }
}

static void test_bsm_greeks_match_finite_differences(void) {
    const qsl_option_type types[] = {QSL_CALL, QSL_PUT};
    const double strikes[] = {80, 100, 125};
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b) {
            const qsl_option_type ty = types[a];
            const double s = 100, k = strikes[b], t = 0.75, r = 0.04, q = 0.01, v = 0.3;
            qsl_greeks g;
            CHECK(qsl_bsm_greeks(ty, s, k, t, r, q, v, &g) == QSL_OK);
            CHECK_CLOSE(g.price, qsl_bsm_price(ty, s, k, t, r, q, v), 1e-13);

            /* Small step for first derivatives, larger for second derivatives
             * (truncation ~h^2 versus rounding ~eps/h^2). */
            const double hd = 1e-5 * s, hs = 1e-3 * s, hv = 1e-5, ht = 1e-6, hr = 1e-6;
#define V(S_, T_, R_, Q_, SIG_) qsl_bsm_price(ty, S_, k, T_, R_, Q_, SIG_)
            const double delta = (V(s + hd, t, r, q, v) - V(s - hd, t, r, q, v)) / (2 * hd);
            const double gamma =
                (V(s + hs, t, r, q, v) - 2 * V(s, t, r, q, v) + V(s - hs, t, r, q, v)) / (hs * hs);
            const double vega = (V(s, t, r, q, v + hv) - V(s, t, r, q, v - hv)) / (2 * hv);
            const double theta = -(V(s, t + ht, r, q, v) - V(s, t - ht, r, q, v)) / (2 * ht);
            const double rho = (V(s, t, r + hr, q, v) - V(s, t, r - hr, q, v)) / (2 * hr);
            const double drho = (V(s, t, r, q + hr, v) - V(s, t, r, q - hr, v)) / (2 * hr);
            const double vanna = (V(s + hs, t, r, q, v + hv) - V(s + hs, t, r, q, v - hv) -
                                  V(s - hs, t, r, q, v + hv) + V(s - hs, t, r, q, v - hv)) /
                                 (4 * hs * hv);
            const double volga =
                (V(s, t, r, q, v + hv) - 2 * V(s, t, r, q, v) + V(s, t, r, q, v - hv)) / (hv * hv);
#undef V
            CHECK_NEAR(g.delta, delta, 1e-8);
            CHECK_NEAR(g.gamma, gamma, 1e-6);
            CHECK_NEAR(g.vega, vega, 1e-7);
            CHECK_NEAR(g.theta, theta, 1e-6);
            CHECK_NEAR(g.rho, rho, 1e-6);
            CHECK_NEAR(g.dividend_rho, drho, 1e-6);
            CHECK_NEAR(g.vanna, vanna, 1e-5);
            CHECK_NEAR(g.volga, volga, 1e-3);
        }
}

static void test_bsm_edge_cases(void) {
    /* Zero vol or zero time: discounted forward intrinsic value. */
    CHECK_CLOSE(qsl_bsm_price(QSL_CALL, 100, 90, 1.0, 0.05, 0.0, 0.0), 100 - 90 * exp(-0.05), 1e-12);
    CHECK_CLOSE(qsl_bsm_price(QSL_PUT, 100, 90, 1.0, 0.05, 0.0, 0.0), 0.0, 0.0);
    CHECK_CLOSE(qsl_bsm_price(QSL_PUT, 80, 90, 0.0, 0.05, 0.0, 0.3), 10.0, 1e-12);
    qsl_greeks g;
    CHECK(qsl_bsm_greeks(QSL_CALL, 100, 90, 0.0, 0.05, 0.0, 0.3, &g) == QSL_OK);
    CHECK_CLOSE(g.delta, 1.0, 0.0);
    CHECK_CLOSE(g.gamma, 0.0, 0.0);
    /* Invalid inputs. */
    CHECK(isnan(qsl_bsm_price(QSL_CALL, -1, 90, 1, 0.05, 0, 0.2)));
    CHECK(isnan(qsl_bsm_price(QSL_CALL, 100, 90, -1, 0.05, 0, 0.2)));
    CHECK(isnan(qsl_bsm_price((qsl_option_type)0, 100, 90, 1, 0.05, 0, 0.2)));
    CHECK(qsl_bsm_greeks(QSL_CALL, 100, 90, 1, 0.05, 0, -0.2, &g) == QSL_E_DOMAIN);
    CHECK(qsl_bsm_greeks(QSL_CALL, 100, 90, 1, 0.05, 0, 0.2, NULL) == QSL_E_INVALID_ARG);
}

static void test_implied_vol_roundtrip(void) {
    const double strikes[] = {40, 70, 95, 100, 105, 130, 200};
    const double times[] = {0.02, 0.25, 1.0, 3.0, 10.0};
    const double vols[] = {0.03, 0.12, 0.25, 0.6, 1.5};
    int solved = 0;
    for (int ty = 0; ty < 2; ++ty)
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 5; ++j)
                for (int k = 0; k < 5; ++k) {
                    const qsl_option_type type = ty ? QSL_CALL : QSL_PUT;
                    const double s = 100, t = times[j], r = 0.05, q = 0.02, v = vols[k];
                    qsl_greeks g;
                    qsl_bsm_greeks(type, s, strikes[i], t, r, q, v, &g);
                    const double intrinsic = fmax((double)type * (s * exp(-q * t) - strikes[i] * exp(-r * t)), 0.0);
                    if (g.price - intrinsic < 1e-10 * s) continue; /* no time value left to invert */
                    double iv;
                    const qsl_status st = qsl_bsm_implied_vol(type, g.price, s, strikes[i], t, r, q, &iv);
                    CHECK(st == QSL_OK);
                    CHECK_CLOSE(qsl_bsm_price(type, s, strikes[i], t, r, q, iv), g.price, 1e-11);
                    if (g.vega > 1e-3) CHECK_CLOSE(iv, v, 1e-9);
                    ++solved;
                }
    CHECK(solved >= 280); /* 350 cases; short-dated far wings carry no time value */
}

static void test_implied_vol_rejects_arbitrage(void) {
    double iv = 0.0;
    /* Below intrinsic, at the upper bound, and invalid inputs. */
    CHECK(qsl_bsm_implied_vol(QSL_CALL, 5.0, 110, 100, 1, 0.0, 0.0, &iv) == QSL_E_ARBITRAGE);
    CHECK(isnan(iv));
    CHECK(qsl_bsm_implied_vol(QSL_CALL, 100.0, 100, 100, 1, 0.0, 0.0, &iv) == QSL_E_ARBITRAGE);
    CHECK(qsl_bsm_implied_vol(QSL_PUT, 96.0, 100, 100, 1, 0.05, 0.0, &iv) == QSL_E_ARBITRAGE);
    CHECK(qsl_bsm_implied_vol(QSL_CALL, 5.0, 100, 100, 0, 0.0, 0.0, &iv) == QSL_E_DOMAIN);
    CHECK(qsl_bsm_implied_vol(QSL_CALL, 5.0, 100, 100, 1, 0.0, 0.0, NULL) == QSL_E_INVALID_ARG);
    /* Exactly intrinsic -> zero vol. */
    CHECK(qsl_bsm_implied_vol(QSL_CALL, 10.0, 110, 100, 1, 0.0, 0.0, &iv) == QSL_OK);
    CHECK_CLOSE(iv, 0.0, 0.0);
}

/* ------------------------------------------------------------------- RNG */

static void test_rng_determinism(void) {
    qsl_rng a, b, c;
    qsl_rng_seed(&a, 42);
    qsl_rng_seed(&b, 42);
    qsl_rng_seed(&c, 43);
    int same = 1, differ = 0;
    for (int i = 0; i < 1000; ++i) {
        const uint64_t x = qsl_rng_next(&a);
        same &= x == qsl_rng_next(&b);
        differ |= x != qsl_rng_next(&c);
    }
    CHECK(same);
    CHECK(differ);
}

static void test_rng_uniform_statistics(void) {
    const int n_draws = 1000000;
    const double N = (double)n_draws;
    double counts[100] = {0};
    const double bins = 100.0;
    qsl_rng rng;
    qsl_rng_seed(&rng, 2024);
    double sum = 0.0;
    for (int i = 0; i < n_draws; ++i) {
        const double u = qsl_rng_uniform(&rng);
        CHECK(u >= 0.0 && u < 1.0);
        sum += u;
        counts[(int)(u * bins)] += 1.0;
    }
    CHECK_CLOSE(sum / N, 0.5, 5.0 * sqrt(1.0 / 12.0 / N));
    double chi2 = 0.0;
    const double expected = N / bins;
    for (int b = 0; b < 100; ++b) chi2 += (counts[b] - expected) * (counts[b] - expected) / expected;
    CHECK(chi2 < (bins - 1.0) + 5.0 * sqrt(2.0 * (bins - 1.0))); /* ~5 sigma upper bound */
}

static void test_rng_normal_moments(void) {
    const int n_draws = 2000000;
    const double N = (double)n_draws;
    qsl_rng rng;
    qsl_rng_seed(&rng, 7);
    double m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (int i = 0; i < n_draws; ++i) {
        const double z = qsl_rng_normal(&rng);
        m1 += z;
        m2 += z * z;
        m3 += z * z * z;
        m4 += z * z * z * z;
    }
    m1 /= N, m2 /= N, m3 /= N, m4 /= N;
    CHECK_CLOSE(m1, 0.0, 5.0 / sqrt(N));
    CHECK_CLOSE(m2, 1.0, 5.0 * sqrt(2.0 / N));
    CHECK_CLOSE(m3, 0.0, 5.0 * sqrt(15.0 / N));
    CHECK_CLOSE(m4, 3.0, 5.0 * sqrt(96.0 / N));
}

/* ----------------------------------------------------------- Monte Carlo */

static void test_mc_european_converges_to_bsm(void) {
    const unsigned flag_sets[] = {QSL_MC_PLAIN, QSL_MC_ANTITHETIC, QSL_MC_CONTROL_VARIATE,
                                  QSL_MC_ANTITHETIC | QSL_MC_CONTROL_VARIATE};
    for (int ty = 0; ty < 2; ++ty)
        for (int f = 0; f < 4; ++f) {
            const qsl_option_type type = ty ? QSL_CALL : QSL_PUT;
            qsl_mc_result res;
            CHECK(qsl_mc_european(type, 100, 105, 1.0, 0.05, 0.02, 0.25, 200000, flag_sets[f],
                                  11 + (uint64_t)f, &res) == QSL_OK);
            const double bs = qsl_bsm_price(type, 100, 105, 1.0, 0.05, 0.02, 0.25);
            CHECK(fabs(res.price - bs) < 4.0 * res.std_error);
            CHECK(res.n_samples == 200000);
        }
}

static void test_mc_variance_reduction(void) {
    qsl_mc_result plain2n, anti, cv;
    const uint64_t n = 100000;
    /* Antithetic at equal cost (n pairs vs 2n plain paths): guaranteed gain for
     * a monotone payoff because Cov(f(Z), f(-Z)) <= 0. */
    qsl_mc_european(QSL_CALL, 100, 100, 1.0, 0.05, 0.0, 0.2, 2 * n, QSL_MC_PLAIN, 1, &plain2n);
    qsl_mc_european(QSL_CALL, 100, 100, 1.0, 0.05, 0.0, 0.2, n, QSL_MC_ANTITHETIC, 2, &anti);
    CHECK(anti.std_error < 0.98 * plain2n.std_error);
    /* Control variate on an in-the-money call: payoff and S_T are highly
     * correlated, so the residual variance collapses. */
    qsl_mc_result plain;
    qsl_mc_european(QSL_CALL, 100, 80, 1.0, 0.05, 0.0, 0.2, n, QSL_MC_PLAIN, 3, &plain);
    qsl_mc_european(QSL_CALL, 100, 80, 1.0, 0.05, 0.0, 0.2, n, QSL_MC_CONTROL_VARIATE, 3, &cv);
    CHECK(cv.std_error < 0.35 * plain.std_error);
    CHECK(cv.beta > 0.5 && cv.beta < 1.5);
}

static void test_asian_geometric_closed_form(void) {
    /* One fixing at T: the geometric Asian is a European option. */
    for (int ty = 0; ty < 2; ++ty) {
        const qsl_option_type type = ty ? QSL_CALL : QSL_PUT;
        CHECK_CLOSE(qsl_asian_geometric_price(type, 100, 95, 2.0, 0.04, 0.01, 0.3, 1),
                    qsl_bsm_price(type, 100, 95, 2.0, 0.04, 0.01, 0.3), 1e-12);
    }
    /* Monte Carlo of the geometric payoff agrees with the closed form. */
    qsl_mc_result res;
    CHECK(qsl_mc_asian(QSL_CALL, QSL_AVG_GEOMETRIC, 100, 100, 1.0, 0.05, 0.0, 0.3, 12, 200000,
                       QSL_MC_ANTITHETIC, 5, &res) == QSL_OK);
    const double exact = qsl_asian_geometric_price(QSL_CALL, 100, 100, 1.0, 0.05, 0.0, 0.3, 12);
    CHECK(fabs(res.price - exact) < 4.0 * res.std_error);
}

static void test_asian_arithmetic_control_variate(void) {
    qsl_mc_result plain, cv;
    const uint64_t n = 50000;
    CHECK(qsl_mc_asian(QSL_CALL, QSL_AVG_ARITHMETIC, 100, 100, 1.0, 0.05, 0.0, 0.3, 52, n,
                       QSL_MC_PLAIN, 9, &plain) == QSL_OK);
    CHECK(qsl_mc_asian(QSL_CALL, QSL_AVG_ARITHMETIC, 100, 100, 1.0, 0.05, 0.0, 0.3, 52, n,
                       QSL_MC_CONTROL_VARIATE, 10, &cv) == QSL_OK);
    /* Kemna-Vorst control: arithmetic and geometric averages are ~99%
     * correlated, so the standard error drops by an order of magnitude. */
    CHECK(cv.std_error < 0.1 * plain.std_error);
    CHECK(fabs(cv.price - plain.price) < 4.0 * hypot(cv.std_error, plain.std_error));
    /* AM-GM: the arithmetic-average call is worth more than the geometric. */
    CHECK(cv.price > qsl_asian_geometric_price(QSL_CALL, 100, 100, 1.0, 0.05, 0.0, 0.3, 52));
    /* The control would be the target itself. */
    CHECK(qsl_mc_asian(QSL_CALL, QSL_AVG_GEOMETRIC, 100, 100, 1, 0.05, 0, 0.3, 12, n,
                       QSL_MC_CONTROL_VARIATE, 1, &cv) == QSL_E_INVALID_ARG);
    CHECK(qsl_mc_european(QSL_CALL, 100, 100, 1, 0.05, 0, 0.3, n, 1u << 5, 1, &cv) ==
          QSL_E_INVALID_ARG);
}

/* ------------------------------------------------------------------ dates */

static void test_date_serials(void) {
    CHECK(qsl_date_to_serial(ymd(1970, 1, 1)) == 0);
    CHECK(qsl_date_to_serial(ymd(1969, 12, 31)) == -1);
    CHECK(qsl_date_to_serial(ymd(2000, 1, 1)) == 10957);
    CHECK(qsl_date_to_serial(ymd(2000, 3, 1)) - qsl_date_to_serial(ymd(2000, 2, 28)) == 2);
    CHECK(qsl_date_to_serial(ymd(1900, 3, 1)) - qsl_date_to_serial(ymd(1900, 2, 28)) == 1);
    int ok = 1;
    for (int64_t s = -1000000; s <= 1000000; ++s) {
        const qsl_date d = qsl_date_from_serial(s);
        ok &= qsl_date_is_valid(d) && qsl_date_to_serial(d) == s;
    }
    CHECK(ok);
    CHECK(qsl_is_leap_year(2000) && qsl_is_leap_year(2024));
    CHECK(!qsl_is_leap_year(1900) && !qsl_is_leap_year(2100) && !qsl_is_leap_year(2023));
    CHECK(!qsl_date_is_valid(ymd(2023, 2, 29)) && !qsl_date_is_valid(ymd(2024, 13, 1)));
}

static int same_date(qsl_date a, qsl_date b) {
    return a.year == b.year && a.month == b.month && a.day == b.day;
}

static void test_add_months(void) {
    CHECK(same_date(qsl_date_add_months(ymd(2024, 1, 31), 1, 0), ymd(2024, 2, 29)));
    CHECK(same_date(qsl_date_add_months(ymd(2024, 2, 29), 1, 0), ymd(2024, 3, 29)));
    CHECK(same_date(qsl_date_add_months(ymd(2024, 2, 29), 1, 1), ymd(2024, 3, 31)));
    CHECK(same_date(qsl_date_add_months(ymd(2024, 1, 15), -1, 0), ymd(2023, 12, 15)));
    CHECK(same_date(qsl_date_add_months(ymd(2023, 11, 30), -12, 1), ymd(2022, 11, 30)));
    CHECK(same_date(qsl_date_add_months(ymd(2030, 8, 31), -6, 1), ymd(2030, 2, 28)));
    CHECK(same_date(qsl_date_add_months(ymd(2030, 2, 28), 6, 1), ymd(2030, 8, 31)));
    CHECK(same_date(qsl_date_add_months(ymd(2030, 2, 28), 6, 0), ymd(2030, 8, 28)));
}

static void test_day_counts(void) {
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_ACT_360, ymd(2024, 1, 15), ymd(2024, 7, 15)), 182.0 / 360, 1e-15);
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_ACT_365F, ymd(2024, 1, 15), ymd(2024, 7, 15)), 182.0 / 365, 1e-15);
    /* 30/360 bond basis: D2 = 31 only becomes 30 when D1 is 30 or 31. */
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_30_360_BOND_BASIS, ymd(2024, 1, 31), ymd(2024, 3, 31)), 60.0 / 360, 1e-15);
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_30_360_BOND_BASIS, ymd(2024, 1, 15), ymd(2024, 3, 31)), 76.0 / 360, 1e-15);
    /* 30E/360: every 31st becomes the 30th. */
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_30E_360, ymd(2024, 1, 15), ymd(2024, 3, 31)), 75.0 / 360, 1e-15);
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_30E_360, ymd(2024, 2, 29), ymd(2024, 8, 31)), 181.0 / 360, 1e-15);
    /* ACT/ACT ISDA splits the period at the year boundary. */
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_ACT_ACT_ISDA, ymd(2023, 12, 15), ymd(2024, 1, 15)),
                17.0 / 365 + 14.0 / 366, 1e-15);
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_ACT_ACT_ISDA, ymd(2024, 1, 1), ymd(2025, 1, 1)), 1.0, 1e-15);
    CHECK_CLOSE(qsl_year_fraction(QSL_DC_ACT_ACT_ISDA, ymd(2025, 1, 1), ymd(2024, 1, 1)), -1.0, 1e-15);
    CHECK(isnan(qsl_year_fraction(QSL_DC_ACT_ACT_ICMA, ymd(2024, 1, 1), ymd(2024, 2, 1))));
    CHECK_CLOSE(qsl_year_fraction_icma(ymd(2024, 1, 15), ymd(2024, 4, 15), ymd(2024, 1, 15),
                                       ymd(2024, 7, 15), 2),
                91.0 / (2 * 182.0), 1e-15);
}

/* ------------------------------------------------------------------ bonds */

static qsl_bond make_bond(qsl_date maturity, double coupon, int freq, qsl_day_count dc) {
    return (qsl_bond){maturity, coupon, 100.0, freq, dc, 0};
}

static void test_bond_hull_table_4_6(void) {
    /* Hull Table 4.6: 3-year 10% semiannual bond, yield 12% continuously
     * compounded: price 94.213, duration 2.653 years. */
    const qsl_bond b = make_bond(ymd(2027, 1, 15), 0.10, 2, QSL_DC_ACT_ACT_ICMA);
    qsl_bond_metrics m;
    CHECK(qsl_bond_price(&b, ymd(2024, 1, 15), 0.12, QSL_CONTINUOUS, &m) == QSL_OK);
    CHECK_CLOSE(m.dirty_price, 94.213, 5e-4);
    CHECK_CLOSE(m.macaulay_duration, 2.653, 5e-4);
    CHECK_CLOSE(m.accrued, 0.0, 0.0);
    CHECK(m.cashflows_remaining == 6);
}

static void test_bond_par_identity(void) {
    /* On a coupon date a bond whose coupon equals its yield prices at par. */
    const qsl_day_count dcs[] = {QSL_DC_ACT_ACT_ICMA, QSL_DC_30_360_BOND_BASIS, QSL_DC_30E_360};
    const int freqs[] = {1, 2, 4, 12};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) {
            const qsl_bond b = make_bond(ymd(2034, 6, 15), 0.0725, freqs[j], dcs[i]);
            qsl_bond_metrics m;
            CHECK(qsl_bond_price(&b, ymd(2024, 6, 15), 0.0725, QSL_COMPOUNDED, &m) == QSL_OK);
            CHECK_CLOSE(m.clean_price, 100.0, 1e-10);
        }
}

static void test_bond_yield_roundtrip_and_risk(void) {
    const qsl_bond b = make_bond(ymd(2035, 11, 15), 0.0645, 2, QSL_DC_ACT_ACT_ICMA);
    const qsl_date settle = ymd(2025, 3, 7);
    const qsl_compounding comps[] = {QSL_COMPOUNDED, QSL_CONTINUOUS};
    const double yields[] = {-0.004, 0.0, 0.031, 0.0712, 0.18};
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 5; ++i) {
            qsl_bond_metrics m, up, dn, up2, dn2;
            const double y = yields[i], h1 = 1e-6, h2 = 1e-4;
            CHECK(qsl_bond_price(&b, settle, y, comps[c], &m) == QSL_OK);
            double y_back;
            CHECK(qsl_bond_yield(&b, settle, m.clean_price, comps[c], &y_back) == QSL_OK);
            CHECK_CLOSE(y_back, y, 1e-12);
            CHECK_CLOSE(m.dirty_price - m.clean_price, m.accrued, 1e-12);
            /* Central differences: small step for slope, larger for curvature. */
            qsl_bond_price(&b, settle, y + h1, comps[c], &up);
            qsl_bond_price(&b, settle, y - h1, comps[c], &dn);
            qsl_bond_price(&b, settle, y + h2, comps[c], &up2);
            qsl_bond_price(&b, settle, y - h2, comps[c], &dn2);
            const double dpdy = (up.dirty_price - dn.dirty_price) / (2 * h1);
            const double d2pdy2 = (up2.dirty_price - 2 * m.dirty_price + dn2.dirty_price) / (h2 * h2);
            CHECK_NEAR(m.modified_duration, -dpdy / m.dirty_price, 1e-8);
            CHECK_NEAR(m.convexity, d2pdy2 / m.dirty_price, 1e-5);
            CHECK_NEAR(m.dv01, -dpdy * 1e-4, 1e-9);
        }
}

static void test_bond_schedule_and_accrual(void) {
    /* Month-end schedule: Feb-28 maturity in a non-leap year rolls to Aug-31. */
    qsl_bond b = make_bond(ymd(2030, 2, 28), 0.05, 2, QSL_DC_ACT_ACT_ICMA);
    b.end_of_month = 1;
    qsl_bond_metrics m;
    CHECK(qsl_bond_price(&b, ymd(2029, 10, 1), 0.05, QSL_COMPOUNDED, &m) == QSL_OK);
    CHECK(same_date(m.previous_coupon, ymd(2029, 8, 31)));
    CHECK(same_date(m.next_coupon, ymd(2030, 2, 28)));
    CHECK(m.cashflows_remaining == 1);
    /* ACT/ACT ICMA accrued: coupon * days accrued / days in period. */
    CHECK_CLOSE(m.accrued, 2.5 * 31.0 / 181.0, 1e-12);
    /* Pull to par: one day before maturity the clean price is ~100. */
    CHECK(qsl_bond_price(&b, ymd(2030, 2, 27), 0.08, QSL_COMPOUNDED, &m) == QSL_OK);
    CHECK_CLOSE(m.clean_price, 100.0, 0.01);
    /* Errors: settlement on/after maturity, bad frequency, bad yield. */
    CHECK(qsl_bond_price(&b, ymd(2030, 2, 28), 0.05, QSL_COMPOUNDED, &m) == QSL_E_DOMAIN);
    b.frequency = 3;
    CHECK(qsl_bond_price(&b, ymd(2025, 1, 1), 0.05, QSL_COMPOUNDED, &m) == QSL_E_INVALID_ARG);
    b.frequency = 2;
    CHECK(qsl_bond_price(&b, ymd(2025, 1, 1), -2.5, QSL_COMPOUNDED, &m) == QSL_E_DOMAIN);
}

static void test_bond_zero_coupon_negative_yield(void) {
    const qsl_bond z = make_bond(ymd(2029, 6, 15), 0.0, 1, QSL_DC_ACT_ACT_ICMA);
    double y;
    CHECK(qsl_bond_yield(&z, ymd(2024, 6, 15), 101.0, QSL_COMPOUNDED, &y) == QSL_OK);
    CHECK_CLOSE(y, pow(100.0 / 101.0, 1.0 / 5.0) - 1.0, 1e-13);
    CHECK(y < 0.0);
}

int main(void) {
    printf("C kernel tests\n");
    RUN_TEST(test_normal_known_values);
    RUN_TEST(test_normal_inverse_roundtrip);
    RUN_TEST(test_bsm_hull_example);
    RUN_TEST(test_bsm_put_call_parity);
    RUN_TEST(test_bsm_greeks_match_finite_differences);
    RUN_TEST(test_bsm_edge_cases);
    RUN_TEST(test_implied_vol_roundtrip);
    RUN_TEST(test_implied_vol_rejects_arbitrage);
    RUN_TEST(test_rng_determinism);
    RUN_TEST(test_rng_uniform_statistics);
    RUN_TEST(test_rng_normal_moments);
    RUN_TEST(test_mc_european_converges_to_bsm);
    RUN_TEST(test_mc_variance_reduction);
    RUN_TEST(test_asian_geometric_closed_form);
    RUN_TEST(test_asian_arithmetic_control_variate);
    RUN_TEST(test_date_serials);
    RUN_TEST(test_add_months);
    RUN_TEST(test_day_counts);
    RUN_TEST(test_bond_hull_table_4_6);
    RUN_TEST(test_bond_par_identity);
    RUN_TEST(test_bond_yield_roundtrip_and_risk);
    RUN_TEST(test_bond_schedule_and_accrual);
    RUN_TEST(test_bond_zero_coupon_negative_yield);
    return UNIT_SUMMARY();
}
