"""Shared fixtures. QuantLib-based oracle tests skip cleanly when QuantLib
is not installed (``pip install QuantLib``)."""
from __future__ import annotations

import pytest

try:
    import QuantLib as ql  # noqa: N813
except ImportError:  # pragma: no cover - exercised only without QuantLib
    ql = None


@pytest.fixture(scope="session")
def quantlib():
    if ql is None:
        pytest.skip("QuantLib not installed")
    return ql


@pytest.fixture()
def ql_env(quantlib):
    """QuantLib helpers on a fixed evaluation date with ACT/365F flat curves,
    so QuantLib year fractions equal the plain year fractions used by qsl."""
    today = quantlib.Date(15, 1, 2025)
    quantlib.Settings.instance().evaluationDate = today
    dc = quantlib.Actual365Fixed()

    class Env:
        def __init__(self):
            self.today, self.dc, self.ql = today, dc, quantlib

        def curve(self, rate):
            return quantlib.YieldTermStructureHandle(quantlib.FlatForward(today, rate, dc))

        def bsm_process(self, spot, r, q, vol):
            return quantlib.BlackScholesMertonProcess(
                quantlib.QuoteHandle(quantlib.SimpleQuote(spot)), self.curve(q), self.curve(r),
                quantlib.BlackVolTermStructureHandle(quantlib.BlackConstantVol(today, quantlib.NullCalendar(), vol, dc)))

        def vanilla(self, kind, strike, days, american=False):
            payoff = quantlib.PlainVanillaPayoff(quantlib.Option.Call if kind == "call" else quantlib.Option.Put, strike)
            maturity = today + days
            exercise = quantlib.AmericanExercise(today, maturity) if american else quantlib.EuropeanExercise(maturity)
            return quantlib.VanillaOption(payoff, exercise)

    return Env()
