"""Pricing: C kernels and C++ engines against SciPy, an independent NumPy
implementation and QuantLib."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.special import ndtri

from qsl import pricing as P
from qsl._native import Rng, clib

KINDS = ("call", "put")


# ----------------------------------------------------------- normal / RNG


def test_inverse_normal_matches_scipy_to_machine_precision():
    lib = clib()
    p = np.concatenate([np.logspace(-300, -1, 400), np.linspace(0.01, 0.99, 197), 1 - np.logspace(-16, -1, 100)])
    ours = np.array([lib.qsl_norm_inv(x) for x in p])
    # Mixed tolerance: near p = 0.5 the quantile is ~0, so a 4e-17 absolute
    # difference (below the rounding of p itself) is a large relative one.
    np.testing.assert_allclose(ours, ndtri(p), rtol=4e-15, atol=1e-16)


def test_xoshiro256starstar_matches_reference_transliteration():
    """Re-implement SplitMix64 seeding + xoshiro256** in Python from the
    published algorithm and require bit-identical output from the C code."""
    mask = (1 << 64) - 1

    def splitmix(state):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return state, z ^ (z >> 31)

    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & mask

    state, s = 12345, []
    for _ in range(4):
        state, z = splitmix(state)
        s.append(z)
    rng = Rng()
    clib().qsl_rng_seed(rng, 12345)
    for _ in range(10_000):
        result = (rotl((s[1] * 5) & mask, 7) * 9) & mask
        t = (s[1] << 17) & mask
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        assert clib().qsl_rng_next(rng) == result


# -------------------------------------------------------------- closed form


def test_c_batch_kernel_matches_independent_numpy_implementation():
    rng = np.random.default_rng(1)
    n = 20_000
    spot, strike = rng.uniform(20, 200, n), rng.uniform(20, 200, n)
    t, r, q, vol = rng.uniform(0.01, 5, n), rng.uniform(-0.01, 0.1, n), rng.uniform(0, 0.05, n), rng.uniform(0.05, 1.0, n)
    for kind in KINDS:
        np.testing.assert_allclose(P.black_scholes(kind, spot, strike, t, r, q, vol),
                                   P.black_scholes_numpy(kind, spot, strike, t, r, q, vol), rtol=1e-11, atol=1e-11)


def test_black_scholes_broadcasting_and_invalid_rows():
    prices = P.black_scholes("call", 100.0, [90.0, 100.0, -5.0], 1.0, 0.05, 0.0, 0.2)
    assert prices.shape == (3,)
    assert np.isnan(prices[2]) and np.all(np.isfinite(prices[:2]))
    assert isinstance(P.black_scholes("put", 100, 100, 1, 0.05, 0, 0.2), float)


@pytest.mark.parametrize("kind,strike,days,vol", list(itertools.product(KINDS, [70, 100, 135], [30, 365, 1825], [0.1, 0.4])))
def test_bsm_price_and_greeks_match_quantlib(ql_env, kind, strike, days, vol):
    t, r, q, spot = days / 365, 0.045, 0.015, 100.0
    opt = ql_env.vanilla(kind, strike, days)
    opt.setPricingEngine(ql_env.ql.AnalyticEuropeanEngine(ql_env.bsm_process(spot, r, q, vol)))
    g = P.greeks(kind, spot, strike, t, r, q, vol)
    for ours, theirs in [(g.price, opt.NPV()), (g.delta, opt.delta()), (g.gamma, opt.gamma()), (g.vega, opt.vega()),
                         (g.theta, opt.theta()), (g.rho, opt.rho()), (g.dividend_rho, opt.dividendRho())]:
        assert ours == pytest.approx(theirs, rel=1e-10, abs=1e-12)


def test_implied_vol_matches_quantlib(ql_env):
    for kind, strike, vol in itertools.product(KINDS, [60, 95, 100, 110, 160], [0.08, 0.3, 0.9]):
        opt = ql_env.vanilla(kind, strike, 730)
        process = ql_env.bsm_process(100, 0.03, 0.01, vol)
        opt.setPricingEngine(ql_env.ql.AnalyticEuropeanEngine(process))
        price = opt.NPV()
        ours = P.implied_vol(kind, price, 100, strike, 2.0, 0.03, 0.01)
        theirs = opt.impliedVolatility(price, process, 1e-12, 1000, 1e-4, 5.0)
        # Vol is only determined to (price rounding) / vega.
        vega = P.greeks(kind, 100, strike, 2.0, 0.03, 0.01, vol).vega
        assert P.black_scholes(kind, 100, strike, 2.0, 0.03, 0.01, ours) == pytest.approx(price, rel=1e-13, abs=1e-14)
        assert ours == pytest.approx(vol, abs=1e-12 * 100 / vega + 1e-12)
        assert ours == pytest.approx(theirs, abs=1e-8)


def test_implied_vol_rejects_arbitrage():
    with pytest.raises(ValueError, match="no-arbitrage"):
        P.implied_vol("call", 0.5, 110, 100, 1.0, 0.0)


# ------------------------------------------------------------- C++ engines


@pytest.mark.parametrize("kind,american", list(itertools.product(KINDS, [False, True])))
def test_leisen_reimer_tree_matches_quantlib_exactly(ql_env, kind, american):
    for strike in (80, 100, 120):
        opt = ql_env.vanilla(kind, strike, 365, american)
        opt.setPricingEngine(ql_env.ql.BinomialVanillaEngine(ql_env.bsm_process(100, 0.05, 0.03, 0.3), "lr", 801))
        ours = P.binomial(kind, 100, strike, 1.0, 0.05, 0.03, 0.3, 801, american=american)
        assert ours == pytest.approx(opt.NPV(), abs=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_finite_difference_american_matches_quantlib(ql_env, kind):
    for strike in (85, 100, 115):
        opt = ql_env.vanilla(kind, strike, 365, american=True)
        opt.setPricingEngine(ql_env.ql.FdBlackScholesVanillaEngine(ql_env.bsm_process(100, 0.05, 0.03, 0.3), 1000, 1000))
        ours = P.finite_difference(kind, 100, strike, 1.0, 0.05, 0.03, 0.3, american=True,
                                   space_steps=800, time_steps=800)
        assert ours.price == pytest.approx(opt.NPV(), abs=1e-3)
        assert ours.delta == pytest.approx(opt.delta(), abs=1e-3)


HESTON_CASES = [  # (v0, kappa, theta, xi, rho, days)
    (0.04, 2.0, 0.04, 0.5, -0.7, 365),
    (0.09, 1.0, 0.05, 1.0, -0.9, 1825),
    (0.02, 4.0, 0.06, 0.3, 0.2, 90),
    (0.0625, 0.5, 0.0625, 0.8, -0.5, 3650),
    (0.04, 1.5, 0.04, 0.6, -0.3, 14),
]


@pytest.mark.parametrize("case", HESTON_CASES)
def test_heston_matches_quantlib_analytic_engine(ql_env, case):
    ql = ql_env.ql
    v0, kappa, theta, xi, rho, days = case
    process = ql.HestonProcess(ql_env.curve(0.03), ql_env.curve(0.01), ql.QuoteHandle(ql.SimpleQuote(100.0)),
                               v0, kappa, theta, xi, rho)
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    for kind, strike in itertools.product(KINDS, (70, 90, 100, 110, 140)):
        opt = ql_env.vanilla(kind, strike, days)
        opt.setPricingEngine(engine)
        ours = P.heston(kind, 100, strike, days / 365, 0.03, 0.01, P.HestonParams(v0, kappa, theta, xi, rho))
        assert ours == pytest.approx(opt.NPV(), rel=1e-7, abs=1e-9)


def test_geometric_asian_matches_quantlib(ql_env):
    ql = ql_env.ql
    for n, kind in itertools.product((1, 4, 12, 52), KINDS):
        step = 364 // n
        fixings = [ql_env.today + step * (i + 1) for i in range(n)]
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if kind == "call" else ql.Option.Put, 100.0)
        opt = ql.DiscreteAveragingAsianOption(ql.Average.Geometric, 1.0, 0, fixings, payoff,
                                              ql.EuropeanExercise(fixings[-1]))
        opt.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(ql_env.bsm_process(100, 0.05, 0.02, 0.25)))
        ours = P.asian_geometric(kind, 100, 100, step * n / 365, 0.05, 0.02, 0.25, n)
        assert ours == pytest.approx(opt.NPV(), rel=1e-12)


# --------------------------------------------------------------- Monte Carlo


def test_monte_carlo_confidence_intervals_cover_the_truth():
    """95% intervals from independent seeds should contain the BSM price
    about 95% of the time (binomial bounds for 200 trials)."""
    exact = P.black_scholes("call", 100, 100, 1.0, 0.05, 0.0, 0.2)
    hits = 0
    for seed in range(200):
        lo, hi = P.monte_carlo_european("call", 100, 100, 1.0, 0.05, 0.0, 0.2, 5_000, seed=seed).confidence_interval
        hits += lo <= exact <= hi
    assert 178 <= hits <= 199


def test_kemna_vorst_control_variate_efficiency():
    plain = P.monte_carlo_asian("call", 100, 100, 1.0, 0.05, 0.0, 0.3, 52, 20_000)
    cv = P.monte_carlo_asian("call", 100, 100, 1.0, 0.05, 0.0, 0.3, 52, 20_000, control_variate=True)
    assert cv.std_error < plain.std_error / 10
    assert abs(cv.price - plain.price) < 4 * math.hypot(cv.std_error, plain.std_error)


def test_engines_raise_on_invalid_input():
    with pytest.raises(ValueError):
        P.binomial("call", -1, 100, 1, 0.05, 0, 0.2)
    with pytest.raises(ValueError):
        P.heston("call", 100, 100, 1, 0.05, 0, P.HestonParams(0.04, 1, 0.04, 0.5, 1.5))
    with pytest.raises(ValueError):
        P.greeks("straddle", 100, 100, 1, 0.05, 0, 0.2)
