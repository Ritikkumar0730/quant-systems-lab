"""Yield-curve construction.

* ``DiscountCurve.bootstrap`` - exact bootstrap from deposits, zero-coupon
  bonds, coupon bonds and par swaps (single-curve), with log-linear
  interpolation of discount factors, i.e. piecewise-flat instantaneous
  forwards: the curve reprices every input instrument exactly.
* ``nelson_siegel_svensson`` / ``fit_nss`` - the parametric curve used by
  many central banks (and by CCIL for the Indian G-Sec zero curve).

Times are year fractions; zero rates are continuously compounded.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from scipy.optimize import brentq, least_squares

# ------------------------------------------------------------- instruments


@dataclass(frozen=True)
class Deposit:
    """Money-market deposit: simple interest, DF(T) = 1 / (1 + r T)."""
    maturity: float
    rate: float

    def cashflows(self) -> list[tuple[float, float]]:
        return [(self.maturity, 1.0 + self.rate * self.maturity)]

    price = 1.0


@dataclass(frozen=True)
class ZeroCouponBond:
    maturity: float
    price: float
    face: float = 100.0

    def cashflows(self) -> list[tuple[float, float]]:
        return [(self.maturity, self.face)]


@dataclass(frozen=True)
class CouponBond:
    """Bullet bond priced on a coupon date (dirty = clean)."""
    maturity: float
    coupon_rate: float
    price: float
    frequency: int = 2
    face: float = 100.0

    def cashflows(self) -> list[tuple[float, float]]:
        n = round(self.maturity * self.frequency)
        coupon = self.face * self.coupon_rate / self.frequency
        flows = [(self.maturity - k / self.frequency, coupon) for k in range(n - 1, -1, -1)]
        flows[-1] = (self.maturity, coupon + self.face)
        return flows


@dataclass(frozen=True)
class ParSwap:
    """Single-curve par swap: fixed leg pays rate / frequency, the floating
    leg is worth par, so  rate * sum tau DF(t_i) + DF(T) = 1."""
    maturity: float
    rate: float
    frequency: int = 1

    def cashflows(self) -> list[tuple[float, float]]:
        n = round(self.maturity * self.frequency)
        flows = [((k + 1) / self.frequency, self.rate / self.frequency) for k in range(n)]
        flows[-1] = (self.maturity, 1.0 + self.rate / self.frequency)
        return flows

    price = 1.0


Instrument = Union[Deposit, ZeroCouponBond, CouponBond, ParSwap]


# ------------------------------------------------------------------- curve


class DiscountCurve:
    """Discount factors on pillar times, log-linear in between, flat
    instantaneous forward beyond the last pillar."""

    def __init__(self, times: Sequence[float], discount_factors: Sequence[float]):
        t = [0.0, *map(float, times)]
        log_df = [0.0, *(math.log(d) for d in discount_factors)]
        if any(b <= a for a, b in zip(t, t[1:])):
            raise ValueError("pillar times must be strictly increasing and positive")
        self._t = t
        self._log_df = log_df

    @property
    def pillars(self) -> list[float]:
        return self._t[1:]

    def log_df(self, t: float) -> float:
        if t < 0:
            raise ValueError("negative time")
        ts, ys = self._t, self._log_df
        if t >= ts[-1]:  # flat forward extrapolation using the last segment
            if len(ts) == 1:
                return 0.0
            slope = (ys[-1] - ys[-2]) / (ts[-1] - ts[-2])
            return ys[-1] + slope * (t - ts[-1])
        i = bisect.bisect_right(ts, t) - 1
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        return ys[i] + w * (ys[i + 1] - ys[i])

    def df(self, t: float) -> float:
        return math.exp(self.log_df(t))

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate."""
        if t <= 0:
            return self.instantaneous_forward(0.0)
        return -self.log_df(t) / t

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be after t1")
        return (self.log_df(t1) - self.log_df(t2)) / (t2 - t1)

    def instantaneous_forward(self, t: float, h: float = 1e-6) -> float:
        return self.forward_rate(t, t + h)

    def price(self, instrument: Instrument) -> float:
        return sum(cf * self.df(t) for t, cf in instrument.cashflows())

    @classmethod
    def bootstrap(cls, instruments: Sequence[Instrument]) -> "DiscountCurve":
        """Solve pillar by pillar (sorted by maturity) for the discount factor
        that reprices each instrument exactly. Earlier cash flows only touch
        already-solved pillars, so each step is a 1-D root find."""
        ordered = sorted(instruments, key=lambda x: x.maturity)
        if len({x.maturity for x in ordered}) != len(ordered):
            raise ValueError("instrument maturities must be distinct")
        times: list[float] = []
        dfs: list[float] = []
        for inst in ordered:
            target = inst.price

            def error(log_df: float, inst=inst, target=target) -> float:
                curve = cls([*times, inst.maturity], [*dfs, math.exp(log_df)])
                return curve.price(inst) - target

            log_df = brentq(error, -20.0, 5.0, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
            times.append(inst.maturity)
            dfs.append(math.exp(log_df))
        return cls(times, dfs)


# ------------------------------------------------------ Nelson-Siegel-Svensson


def nelson_siegel_svensson(t, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float):
    """NSS zero rate. beta0 is the long end, beta0 + beta1 the short end,
    beta2/beta3 add humps at scales tau1/tau2."""
    t = np.maximum(np.asarray(t, dtype=float), 1e-12)
    x1, x2 = t / tau1, t / tau2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3


@dataclass(frozen=True)
class NssFit:
    params: tuple[float, float, float, float, float, float]
    rmse: float

    def __call__(self, t):
        return nelson_siegel_svensson(t, *self.params)


def fit_nss(times: Sequence[float], zero_rates: Sequence[float]) -> NssFit:
    """Least-squares NSS fit to observed zero rates (multi-start over tau)."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(zero_rates, dtype=float)
    best = None
    for tau1, tau2 in [(1.0, 5.0), (2.0, 10.0), (0.5, 3.0), (3.0, 12.0)]:
        x0 = [y[-1], y[0] - y[-1], 0.0, 0.0, tau1, tau2]
        res = least_squares(lambda p: nelson_siegel_svensson(t, *p) - y, x0,
                            bounds=([-1, -1, -1, -1, 0.05, 0.05], [1, 1, 1, 1, 30, 30]), xtol=1e-15, ftol=1e-15,
                            gtol=1e-15, max_nfev=20000)
        if best is None or res.cost < best.cost:
            best = res
    rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    return NssFit(tuple(float(v) for v in best.x), rmse)
