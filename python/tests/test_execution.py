"""Execution: Almgren-Chriss checked against a direct numerical optimisation
and Monte Carlo; TWAP/VWAP schedules and the bar simulator checked against
exact identities."""
from __future__ import annotations

import numpy as np
import pytest

from qsl.execution import (ACParams, allocate, cost_variance, efficient_frontier, expected_cost, kappa, simulate,
                           cap_participation, simulate_shortfall, solve_numerically, trades, trajectory, twap, volume_profile, vwap)

# The worked example of Almgren & Chriss (2000): 1M shares of a $50 stock with
# 30% annual volatility, sold over 5 days in daily intervals.
PAPER = ACParams(shares=1e6, horizon=5, periods=5, sigma=0.95, eta=2.5e-6, gamma=2.5e-7, eps=0.0625)


@pytest.mark.parametrize("lam", [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
@pytest.mark.parametrize("periods", [1, 5, 50, 390])
def test_closed_form_trajectory_is_the_numerical_optimum(lam, periods):
    p = ACParams(shares=1e6, horizon=5, periods=periods, sigma=0.95, eta=2.5e-6, gamma=2.5e-7, eps=0.0625)
    x = trajectory(p, lam)
    assert x[0] == p.shares and x[-1] == 0
    np.testing.assert_allclose(x, solve_numerically(p, lam), rtol=1e-9, atol=1e-6 * p.shares)


def test_trajectory_is_optimal_against_random_perturbations():
    rng = np.random.default_rng(0)
    lam = 1e-6
    x = trajectory(PAPER, lam)
    best = expected_cost(PAPER, x) + lam * cost_variance(PAPER, x)
    for _ in range(500):
        y = x.copy()
        y[1:-1] += rng.normal(0, 5e3, size=len(x) - 2)
        y[1:-1] = np.clip(y[1:-1], 0, PAPER.shares)  # stay a monotone-ish sale
        if (np.diff(y) > 0).any():
            continue
        assert expected_cost(PAPER, y) + lam * cost_variance(PAPER, y) >= best - 1e-6


def test_zero_risk_aversion_is_twap_and_urgency_front_loads():
    np.testing.assert_allclose(trades(trajectory(PAPER, 0.0)), np.full(5, 2e5))
    n_lo, n_hi = trades(trajectory(PAPER, 1e-7)), trades(trajectory(PAPER, 1e-5))
    assert (np.diff(n_lo) < 0).all() and (np.diff(n_hi) < 0).all()  # sell fastest first
    assert n_hi[0] > n_lo[0] > 2e5


def test_kappa_solves_the_discrete_relation():
    for lam in (1e-8, 1e-6, 1e-4):
        k = kappa(PAPER, lam)
        lhs = 2 / PAPER.tau**2 * (np.cosh(k * PAPER.tau) - 1)
        assert lhs == pytest.approx(lam * PAPER.sigma**2 / PAPER.eta_tilde, rel=1e-12)


def test_efficient_frontier_trades_cost_for_risk():
    f = efficient_frontier(PAPER, np.logspace(-9, -4, 30))
    assert (np.diff(f[:, 1]) > 0).all()   # more risk-averse: higher expected cost
    assert (np.diff(f[:, 2]) < 0).all()   # ... and lower variance
    # The TWAP end of the frontier has the minimum expected cost of any schedule.
    twap_cost = expected_cost(PAPER, trajectory(PAPER, 0.0))
    assert twap_cost == pytest.approx(0.5 * PAPER.gamma * 1e12 + PAPER.eps * 1e6 + PAPER.eta_tilde * 5 * 2e5**2)


@pytest.mark.parametrize("lam", [0.0, 1e-6])
def test_expected_cost_and_variance_match_monte_carlo(lam):
    rng = np.random.default_rng(42)
    x = trajectory(PAPER, lam)
    sims = simulate_shortfall(PAPER, x, s0=50.0, paths=400_000, rng=rng)
    e, v = expected_cost(PAPER, x), cost_variance(PAPER, x)
    se = np.sqrt(v / len(sims))
    assert abs(sims.mean() - e) < 4 * se
    assert sims.var() == pytest.approx(v, rel=0.01)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        ACParams(shares=1, horizon=1, periods=0, sigma=1, eta=1, gamma=0)
    with pytest.raises(ValueError):
        ACParams(shares=1, horizon=10, periods=1, sigma=1, eta=1e-9, gamma=1).eta_tilde  # noqa: B018
    with pytest.raises(ValueError):
        kappa(PAPER, -1.0)


# ------------------------------------------------------------------ schedules
@pytest.mark.parametrize("total", [0, 1, 7, 999, 1_000_003])
def test_allocate_sums_exactly_and_is_proportional(total):
    rng = np.random.default_rng(total)
    w = rng.random(390) ** 3
    q = allocate(total, w)
    assert q.sum() == total and (q >= 0).all()
    assert np.abs(q - total * w / w.sum()).max() < 1.0  # never more than one share from exact


def test_twap_is_as_even_as_integers_allow():
    q = twap(1000, 390)
    assert q.sum() == 1000 and q.max() - q.min() <= 1


def test_zero_impact_vwap_with_a_perfect_forecast_hits_market_vwap():
    rng = np.random.default_rng(3)
    vol = rng.integers(100, 10_000, 390).astype(float)
    px = 100 + np.cumsum(rng.normal(0, 0.05, 390))
    notional = vol * px
    q = vwap(int(vol.sum()), vol)  # schedule identical to the realised volume
    r = simulate(q, vol, notional)
    assert r.vwap_slippage_bps == pytest.approx(0.0, abs=1e-9)


def test_zero_impact_twap_gets_the_time_weighted_average():
    rng = np.random.default_rng(4)
    vol = rng.integers(100, 10_000, 390).astype(float)
    px = 50 + np.cumsum(rng.normal(0, 0.02, 390))
    r = simulate(twap(390 * 100, 390), vol, vol * px)
    assert r.avg_price == pytest.approx(px.mean(), rel=1e-12)


def test_impact_cost_matches_the_participation_formula():
    vol = np.full(10, 1000.0)
    px = np.full(10, 20.0)
    q = np.full(10, 100)  # 10% participation in every bar
    for side in (1, -1):
        r = simulate(q, vol, vol * px, side=side, impact_coef=0.01, impact_exponent=0.5)
        assert r.vwap_slippage_bps == pytest.approx(0.01 * np.sqrt(0.1) * 1e4, rel=1e-12)
        assert r.max_participation == pytest.approx(0.1)


def test_volume_profile_weights_each_stock_equally():
    v = np.array([[1.0, 1.0, 2.0], [100.0, 300.0, 0.0]])
    np.testing.assert_allclose(volume_profile(v), [(0.25 + 0.25) / 2, (0.25 + 0.75) / 2, 0.5 / 2])


def test_simulate_rejects_trading_into_an_empty_bar():
    with pytest.raises(ValueError):
        simulate([1, 1], [10.0, 0.0], [100.0, 0.0])


def test_cap_participation_carries_excess_and_reports_unfilled():
    vol = np.array([0, 50, 0, 0, 70, 10], dtype=float)
    done, unfilled = cap_participation([10, 10, 10, 10, 10, 10], vol, 0.2)
    # caps per bar: 0, 10, 0, 0, 14, 2
    np.testing.assert_array_equal(done, [0, 10, 0, 0, 14, 2])
    assert unfilled == 60 - 26 and (done[vol == 0] == 0).all()
    done, unfilled = cap_participation([1, 2, 3], [0.0, 5.0, 0.0], np.inf)  # no cap: only empty bars defer
    np.testing.assert_array_equal(done, [0, 3, 0])
    assert unfilled == 3
