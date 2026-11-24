"""Credit default swaps under a piecewise-constant hazard rate.

Survival Q(t) = exp(-int_0^t lambda(s) ds). For a CDS with premium dates
t_i (accrual fractions tau_i) and recovery R:
    RPV01      = sum tau_i DF(t_i) Q(t_i)  + accrued premium on default
    protection = (1 - R) int DF(t) dQ(t)   (integrated on a fine grid)
    par spread = protection / RPV01
``bootstrap_hazard`` solves one hazard rate per quoted maturity so that each
quoted CDS reprices at par, the standard way credit desks build curves.
Rule of thumb ("credit triangle"): spread ~ lambda (1 - R).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Sequence

from scipy.optimize import brentq

DiscountFn = Callable[[float], float]


def flat_discount(rate: float) -> DiscountFn:
    return lambda t: math.exp(-rate * t)


@dataclass(frozen=True)
class HazardCurve:
    """Hazard lambda_k applies on (times[k-1], times[k]]; the last one extends flat."""
    times: tuple[float, ...]
    hazards: tuple[float, ...]

    def survival(self, t: float) -> float:
        integral, prev = 0.0, 0.0
        for end, lam in zip(self.times, self.hazards):
            if t <= end:
                return math.exp(-(integral + lam * (t - prev)))
            integral += lam * (end - prev)
            prev = end
        return math.exp(-(integral + self.hazards[-1] * (t - prev)))

    def hazard(self, t: float) -> float:
        i = min(bisect.bisect_left(self.times, t), len(self.hazards) - 1)
        return self.hazards[i]


@dataclass(frozen=True)
class CdsLegs:
    rpv01: float       # risky annuity per unit spread
    protection: float  # PV of the default leg per unit notional

    @property
    def par_spread(self) -> float:
        return self.protection / self.rpv01


def cds_legs(curve: HazardCurve, maturity: float, recovery: float, discount: DiscountFn, *,
             frequency: int = 4, substeps: int = 20) -> CdsLegs:
    n = max(1, round(maturity * frequency))
    dates = [maturity * (k + 1) / n for k in range(n)]
    rpv01 = protection = 0.0
    prev = 0.0
    for end in dates:
        tau = end - prev
        rpv01 += tau * discount(end) * curve.survival(end)
        # Default inside the period: protection payment plus accrued premium,
        # integrated with the midpoint rule on `substeps` sub-intervals.
        h = tau / substeps
        q_prev = curve.survival(prev)
        for j in range(substeps):
            a, b = prev + j * h, prev + (j + 1) * h
            q_b = curve.survival(b)
            mid = 0.5 * (a + b)
            dq = q_prev - q_b
            protection += (1 - recovery) * discount(mid) * dq
            rpv01 += (mid - prev) * discount(mid) * dq
            q_prev = q_b
        prev = end
    return CdsLegs(rpv01, protection)


def bootstrap_hazard(maturities: Sequence[float], spreads: Sequence[float], recovery: float,
                     discount: DiscountFn, *, frequency: int = 4) -> HazardCurve:
    """Piecewise-constant hazards that reprice each quoted par spread exactly."""
    times: list[float] = []
    hazards: list[float] = []
    for t, s in sorted(zip(maturities, spreads)):
        def error(lam: float, t=t, s=s) -> float:
            legs = cds_legs(HazardCurve((*times, t), (*hazards, lam)), t, recovery, discount, frequency=frequency)
            return legs.protection - s * legs.rpv01
        hazards.append(brentq(error, 1e-10, 5.0, xtol=1e-15))
        times.append(t)
    return HazardCurve(tuple(times), tuple(hazards))
