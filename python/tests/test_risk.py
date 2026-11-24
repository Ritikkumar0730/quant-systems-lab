"""Market risk: VaR/ES definitions against brute force and closed forms,
backtests against the Basel table and their own statistical size, GARCH
parameter recovery, and range-based volatility estimators."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from qsl.risk import (OptionPosition, basel_traffic_light, christoffersen_independence, close_to_close,
                      conditional_coverage, cornish_fisher_var, ewma_variance, fit_garch11, garman_klass,
                      historical_es, historical_var, kupiec_pof, options_var, parametric_es, parametric_var,
                      parkinson, portfolio_var, rogers_satchell)


# ------------------------------------------------------------------ VaR / ES


def test_historical_var_is_the_ceil_order_statistic():
    rng = np.random.default_rng(0)
    for n in (250, 500, 1000, 1001):
        pnl = rng.standard_t(4, n)
        losses = np.sort(-pnl)
        for alpha in (0.95, 0.975, 0.99):
            k = int(np.ceil(n * alpha))  # inf{x : F_n(x) >= alpha}
            assert historical_var(pnl, alpha) == losses[k - 1]


def test_historical_es_equals_integral_of_the_empirical_quantile():
    rng = np.random.default_rng(1)
    pnl = rng.standard_t(3, 777)
    losses = np.sort(-pnl)
    n = losses.size
    for alpha in (0.95, 0.975, 0.99):
        # ES = 1/(1-alpha) * integral_alpha^1 q(u) du for the empirical quantile function.
        edges = np.arange(n + 1) / n
        lo = np.clip(edges[:-1], alpha, 1.0)
        hi = np.clip(edges[1:], alpha, 1.0)
        brute = float(np.sum(losses * (hi - lo))) / (1 - alpha)
        assert historical_es(pnl, alpha) == pytest.approx(brute, rel=1e-12)
        assert historical_es(pnl, alpha) >= historical_var(pnl, alpha)


def test_normal_closed_forms():
    mu, sigma = 0.001, 0.02
    assert parametric_var(mu, sigma, 0.99) == pytest.approx(-mu + sigma * 2.3263478740408408, rel=1e-14)
    es_numeric = integrate.quad(lambda u: -mu + sigma * norm.ppf(u), 0.975, 1.0)[0] / 0.025
    assert parametric_es(mu, sigma, 0.975) == pytest.approx(es_numeric, rel=1e-8)
    sample = np.random.default_rng(2).normal(mu, sigma, 2_000_000)
    assert historical_var(sample, 0.99) == pytest.approx(parametric_var(mu, sigma, 0.99), rel=0.01)
    assert historical_es(sample, 0.975) == pytest.approx(parametric_es(mu, sigma, 0.975), rel=0.01)
    # Basel's 97.5% ES of a normal is close to its 99% VaR.
    assert parametric_es(0, 1, 0.975) == pytest.approx(parametric_var(0, 1, 0.99), rel=0.01)


def test_cornish_fisher_captures_fat_left_tail():
    rng = np.random.default_rng(3)
    normal = rng.normal(0, 1, 200_000)
    assert cornish_fisher_var(normal, 0.99) == pytest.approx(parametric_var(0, 1, 0.99), rel=0.02)
    skewed = -rng.lognormal(0, 0.5, 200_000)  # long left tail of P&L
    skewed = (skewed - skewed.mean()) / skewed.std()
    assert cornish_fisher_var(skewed, 0.99) > parametric_var(0, 1, 0.99) * 1.1


def test_component_var_is_an_euler_allocation():
    w = np.array([1e6, -4e5, 2.5e6])
    vols = np.array([0.012, 0.02, 0.008])
    corr = np.array([[1, 0.3, -0.2], [0.3, 1, 0.5], [-0.2, 0.5, 1]])
    cov = np.outer(vols, vols) * corr
    res = portfolio_var(w, cov, 0.99)
    assert res.component.sum() == pytest.approx(res.var, rel=1e-14)
    for i in range(3):  # marginal VaR is the gradient
        h = np.zeros(3)
        h[i] = 1.0
        fd = (portfolio_var(w + h, cov).var - portfolio_var(w - h, cov).var) / 2
        assert res.marginal[i] == pytest.approx(fd, rel=1e-6)


def test_options_var_full_revaluation_vs_delta_gamma():
    # Short straddle: negative gamma, so delta-only VaR badly understates risk.
    book = [OptionPosition("call", 100, 0.25, -1000), OptionPosition("put", 100, 0.25, -1000)]
    res = options_var(book, spot=100, vol=0.3, r=0.03, horizon_days=1)
    assert abs(res.delta_gamma - res.full_revaluation) / res.full_revaluation < 0.05
    assert res.delta_only < 0.6 * res.full_revaluation
    # Over 10 days the quadratic approximation degrades but stays far better than delta.
    res10 = options_var(book, spot=100, vol=0.3, r=0.03, horizon_days=10)
    assert abs(res10.delta_gamma - res10.full_revaluation) < abs(res10.delta_only - res10.full_revaluation)


# ---------------------------------------------------------------- backtests


def test_basel_traffic_light_reproduces_the_1996_table():
    zones = [basel_traffic_light(x).zone for x in range(13)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 3
    published_cumulative = {0: 0.0811, 1: 0.2858, 2: 0.5432, 3: 0.7581, 4: 0.8922, 5: 0.9588, 6: 0.9863,
                            7: 0.9960, 8: 0.9989, 9: 0.9997}
    for x, p in published_cumulative.items():
        assert basel_traffic_light(x).cumulative_probability == pytest.approx(p, abs=5e-5)
    assert [basel_traffic_light(x).multiplier for x in (4, 5, 7, 9, 10)] == [3.0, 3.4, 3.65, 3.85, 4.0]


def test_kupiec_matches_hand_computed_likelihood_ratio():
    e = np.zeros(250, dtype=bool)
    e[:7] = True
    p, x, n = 0.01, 7, 250
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n))
    res = kupiec_pof(e, p)
    assert res.statistic == pytest.approx(lr, rel=1e-14)
    assert res.exceptions == 7 and res.rejects(0.05)
    assert not kupiec_pof(np.r_[np.ones(2, bool), np.zeros(248, bool)], p).rejects(0.05)
    assert kupiec_pof(np.zeros(250, bool), p).statistic == pytest.approx(-2 * 250 * np.log(0.99))


def test_backtests_have_correct_size_and_detect_clustering():
    rng = np.random.default_rng(4)
    trials = 2000
    # The LR tests are chi-square only asymptotically. With 5000 days the
    # empirical size of a 5% test is close to nominal...
    reject_ind = sum(christoffersen_independence(rng.random(5000) < 0.05).rejects(0.05) for _ in range(trials))
    reject_cc = sum(conditional_coverage(rng.random(5000) < 0.05, 0.05).rejects(0.05) for _ in range(trials))
    assert 0.03 < reject_ind / trials < 0.08
    assert 0.03 < reject_cc / trials < 0.08
    # ...but on ~4 years of data the independence test over-rejects (n11 is
    # tiny), a known finite-sample distortion worth knowing before you
    # reject a VaR model on it.
    small = sum(christoffersen_independence(rng.random(1000) < 0.05).rejects(0.05) for _ in range(trials))
    assert small / trials > 0.06
    clustered = np.zeros(1000, dtype=bool)
    for start in rng.choice(990, 10, replace=False):
        clustered[start:start + 5] = True  # same exception rate, arriving in bursts
    assert christoffersen_independence(clustered).rejects(0.01)


# --------------------------------------------------------------- volatility


def test_ewma_matches_explicit_recursion():
    r = np.random.default_rng(5).normal(0, 0.01, 500)
    var = ewma_variance(r, 0.94)
    ref = [np.var(r)]
    for x in r[:-1]:
        ref.append(0.94 * ref[-1] + 0.06 * x * x)
    np.testing.assert_allclose(var, ref, rtol=1e-12)


def test_garch_recovers_simulated_parameters():
    omega, alpha, beta = 2e-6, 0.08, 0.90
    rng = np.random.default_rng(6)
    n = 30_000
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    fit = fit_garch11(r)
    assert fit.alpha == pytest.approx(alpha, abs=0.015)
    assert fit.beta == pytest.approx(beta, abs=0.02)
    assert fit.long_run_variance == pytest.approx(omega / (1 - alpha - beta), rel=0.15)
    forecast = fit.forecast(2000, r[-1])
    assert forecast[-1] == pytest.approx(fit.long_run_variance, rel=1e-6)  # mean reversion
    assert 5 < fit.half_life < 100


def _simulate_ohlc(days, sigma, steps=2000, drift=0.0, seed=7):
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps
    increments = (drift - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((days, steps))
    paths = np.cumsum(increments, axis=1)
    paths = np.concatenate([np.zeros((days, 1)), paths], axis=1)
    opens = np.r_[0.0, np.cumsum(paths[:-1, -1])]  # each day opens at the previous close
    logs = paths + opens[:, None]
    return np.exp(logs[:, 0]), np.exp(logs.max(1)), np.exp(logs.min(1)), np.exp(logs[:, -1])


def test_range_estimators_are_consistent_and_more_efficient():
    sigma = 0.02  # daily
    o, h, l, c = _simulate_ohlc(4000, sigma)
    closes = np.r_[o[0], c]
    for est in (close_to_close(closes), parkinson(h, l), garman_klass(o, h, l, c), rogers_satchell(o, h, l, c)):
        assert est == pytest.approx(sigma, rel=0.05)  # discrete monitoring biases range estimators slightly low
    # Efficiency: spread of estimates across independent 60-day windows.
    windows = [_simulate_ohlc(60, sigma, steps=500, seed=s) for s in range(150)]
    cc = np.std([close_to_close(np.r_[w[0][0], w[3]]) for w in windows])
    pk = np.std([parkinson(w[1], w[2]) for w in windows])
    gk = np.std([garman_klass(*w) for w in windows])
    assert pk < 0.7 * cc and gk < 0.7 * cc
