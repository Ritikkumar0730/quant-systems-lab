"""Rates: dates and day counts against datetime and QuantLib, bond analytics
against QuantLib, curve bootstrap against Hull's textbook example."""
from __future__ import annotations

import itertools
import math
from datetime import date, timedelta

import numpy as np
import pytest

from qsl._native import Date, clib
from qsl.fixed_income import (CouponBond, DayCount, Deposit, DiscountCurve, FixedRateBond, ParSwap, ZeroCouponBond,
                              fit_nss, nelson_siegel_svensson, year_fraction)

EPOCH = date(1970, 1, 1)


def random_dates(n, seed=0, start=date(1990, 1, 1), span_days=20_000):
    rng = np.random.default_rng(seed)
    return [start + timedelta(days=int(d)) for d in rng.integers(0, span_days, n)]


# ------------------------------------------------------------- dates


def test_serial_dates_match_python_ordinals():
    for d in random_dates(5000, start=date(1, 1, 1), span_days=3_600_000):
        serial = clib().qsl_date_to_serial(Date(d.year, d.month, d.day))
        assert serial == (d - EPOCH).days
        back = clib().qsl_date_from_serial(serial)
        assert (back.year, back.month, back.day) == (d.year, d.month, d.day)


def test_actual_day_counts_match_datetime():
    a, b = random_dates(2000, 1), random_dates(2000, 2)
    for d1, d2 in zip(a, b):
        days = (d2 - d1).days
        assert year_fraction(DayCount.ACT_360, d1, d2) == pytest.approx(days / 360, abs=1e-15)
        assert year_fraction(DayCount.ACT_365F, d1, d2) == pytest.approx(days / 365, abs=1e-15)


def test_thirty_360_and_act_act_match_quantlib(quantlib):
    ql = quantlib
    counters = {
        DayCount.THIRTY_360_BOND_BASIS: ql.Thirty360(ql.Thirty360.BondBasis),
        DayCount.THIRTY_E_360: ql.Thirty360(ql.Thirty360.European),
        DayCount.ACT_ACT_ISDA: ql.ActualActual(ql.ActualActual.ISDA),
    }
    # Month-ends and the 30th/31st are where the conventions differ: oversample them.
    special = [date(y, m, d) for y in (2023, 2024) for m in range(1, 13) for d in (28, 29, 30, 31)
               if d <= (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day]
    dates = random_dates(400, 3) + special
    for d1, d2 in itertools.product(dates[::7], dates[::5]):
        if d2 <= d1:
            continue
        q1, q2 = ql.Date(d1.day, d1.month, d1.year), ql.Date(d2.day, d2.month, d2.year)
        for dc, qdc in counters.items():
            assert year_fraction(dc, d1, d2) == pytest.approx(qdc.yearFraction(q1, q2), abs=1e-14), (dc, d1, d2)


# ------------------------------------------------------------- bonds


def _ql_bond(ql, bond: FixedRateBond, issue: date):
    day_counter = {DayCount.ACT_ACT_ICMA: ql.ActualActual(ql.ActualActual.ISMA),
                   DayCount.THIRTY_360_BOND_BASIS: ql.Thirty360(ql.Thirty360.BondBasis),
                   DayCount.THIRTY_E_360: ql.Thirty360(ql.Thirty360.European)}[bond.day_count]
    period = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly, 12: ql.Monthly}[bond.frequency]
    schedule = ql.Schedule(ql.Date(issue.day, issue.month, issue.year),
                           ql.Date(bond.maturity.day, bond.maturity.month, bond.maturity.year), ql.Period(period),
                           ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    return ql.FixedRateBond(0, bond.face, schedule, [bond.coupon_rate], day_counter), day_counter, period


BOND_CASES = [  # maturity, coupon, frequency, day count, issue
    (date(2035, 11, 15), 0.0645, 2, DayCount.ACT_ACT_ICMA, date(2015, 11, 15)),   # US Treasury style
    (date(2033, 6, 15), 0.0718, 2, DayCount.THIRTY_360_BOND_BASIS, date(2023, 6, 15)),  # G-Sec / corporate style
    (date(2031, 3, 15), 0.025, 1, DayCount.THIRTY_E_360, date(2021, 3, 15)),       # Eurobond style
    (date(2030, 9, 15), 0.05, 4, DayCount.ACT_ACT_ICMA, date(2020, 9, 15)),
    (date(2029, 2, 15), 0.03, 12, DayCount.THIRTY_360_BOND_BASIS, date(2024, 2, 15)),
]


@pytest.mark.parametrize("case", BOND_CASES)
def test_bond_analytics_match_quantlib(quantlib, case):
    ql = quantlib
    maturity, coupon, freq, dc, issue = case
    bond = FixedRateBond(maturity, coupon, freq, day_count=dc)
    qbond, qdc, period = _ql_bond(ql, bond, issue)
    for settle, ytm in itertools.product(random_dates(12, 5, start=date(2025, 1, 1), span_days=1200),
                                         (-0.002, 0.01, 0.0425, 0.09)):
        qs = ql.Date(settle.day, settle.month, settle.year)
        rate = ql.InterestRate(ytm, qdc, ql.Compounded, period)
        a = bond.analytics(settle, ytm)
        assert a.clean_price == pytest.approx(ql.BondFunctions.cleanPrice(qbond, rate, qs), abs=1e-9)
        assert a.accrued == pytest.approx(qbond.accruedAmount(qs), abs=1e-10)
        assert a.modified_duration == pytest.approx(ql.BondFunctions.duration(qbond, rate, ql.Duration.Modified, qs), rel=1e-9)
        assert a.convexity == pytest.approx(ql.BondFunctions.convexity(qbond, rate, qs), rel=1e-8)
        ql_ytm = ql.BondFunctions.bondYield(qbond, ql.BondPrice(a.clean_price, ql.BondPrice.Clean), qdc, ql.Compounded,
                                            period, qs, 1e-14, 200)
        assert bond.yield_to_maturity(settle, a.clean_price) == pytest.approx(ql_ytm, abs=1e-10)


def test_dv01_is_price_change_for_one_basis_point():
    bond = FixedRateBond(date(2034, 5, 15), 0.07)
    settle = date(2025, 2, 3)
    a = bond.analytics(settle, 0.068)
    up, dn = bond.analytics(settle, 0.068 + 1e-4), bond.analytics(settle, 0.068 - 1e-4)
    assert (dn.dirty_price - up.dirty_price) / 2 == pytest.approx(a.dv01, rel=1e-6)
    # Duration-convexity approximation for a 100 bp shock.
    shocked = bond.analytics(settle, 0.078).dirty_price
    approx = a.dirty_price * (1 - a.modified_duration * 0.01 + 0.5 * a.convexity * 0.01 ** 2)
    assert abs(shocked - approx) < 0.05


def test_bond_errors():
    with pytest.raises(ValueError):
        FixedRateBond(date(2030, 1, 15), 0.05).analytics(date(2031, 1, 1), 0.05)
    with pytest.raises(ValueError):
        FixedRateBond(date(2030, 1, 15), 0.05, frequency=3).analytics(date(2025, 1, 1), 0.05)


# ------------------------------------------------------------- curves


def test_bootstrap_reproduces_hull_table_4_4():
    """Hull, Options Futures and Other Derivatives, Tables 4.3/4.4: zero rates
    10.127%, 10.469%, 10.536%, 10.681%, 10.808% (continuous compounding)."""
    curve = DiscountCurve.bootstrap([
        ZeroCouponBond(0.25, 97.5), ZeroCouponBond(0.5, 94.9), ZeroCouponBond(1.0, 90.0),
        CouponBond(1.5, 0.08, 96.0), CouponBond(2.0, 0.12, 101.6)])
    expected = [0.10127, 0.10469, 0.10536, 0.10681, 0.10808]
    for t, z in zip([0.25, 0.5, 1.0, 1.5, 2.0], expected):
        assert curve.zero_rate(t) == pytest.approx(z, abs=5e-6)


def test_bootstrap_reprices_every_instrument_exactly():
    instruments = [Deposit(0.25, 0.052), Deposit(0.5, 0.053), Deposit(1.0, 0.0535),
                   *[ParSwap(t, r) for t, r in [(2, 0.051), (3, 0.049), (5, 0.047), (7, 0.0465), (10, 0.046),
                                                (15, 0.0462), (20, 0.0465), (30, 0.047)]]]
    curve = DiscountCurve.bootstrap(instruments)
    for inst in instruments:
        assert curve.price(inst) == pytest.approx(inst.price, abs=1e-13)
    # Discount factors decrease; forwards stay positive.
    grid = np.linspace(0.01, 30, 300)
    dfs = [curve.df(t) for t in grid]
    assert all(b < a for a, b in zip(dfs, dfs[1:]))
    assert min(curve.forward_rate(a, b) for a, b in zip(grid, grid[1:])) > 0


def test_curve_interpolation_is_flat_forward():
    curve = DiscountCurve([1.0, 2.0], [math.exp(-0.03), math.exp(-0.03 - 0.05)])
    assert curve.forward_rate(1.2, 1.7) == pytest.approx(0.05, abs=1e-14)
    assert curve.forward_rate(2.5, 4.0) == pytest.approx(0.05, abs=1e-14)  # flat extrapolation
    assert curve.zero_rate(1.0) == pytest.approx(0.03, abs=1e-15)


def test_nelson_siegel_svensson_fit_and_limits():
    params = (0.065, -0.02, 0.015, -0.01, 1.5, 8.0)
    t = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    fit = fit_nss(t, nelson_siegel_svensson(t, *params))
    assert fit.rmse < 1e-8
    np.testing.assert_allclose(fit(t), nelson_siegel_svensson(t, *params), atol=1e-8)
    assert nelson_siegel_svensson(1e-9, *params) == pytest.approx(params[0] + params[1], abs=1e-8)  # short end
    assert nelson_siegel_svensson(1e6, *params) == pytest.approx(params[0], abs=1e-5)              # long end
