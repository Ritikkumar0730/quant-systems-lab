"""Merton (1974) structural credit model.

Equity is a European call on the firm's assets V with strike equal to the
face value of debt D due at T. Distance to default is d2 and the
risk-neutral default probability is N(-d2). ``calibrate_from_equity`` solves
the classic KMV-style inverse problem: recover (V, sigma_V) from observed
equity value and equity volatility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import fsolve
from scipy.stats import norm


@dataclass(frozen=True)
class MertonResult:
    asset_value: float
    asset_vol: float
    equity_value: float
    equity_vol: float
    distance_to_default: float     # d2
    default_probability: float     # N(-d2), risk neutral
    debt_value: float              # V - E
    credit_spread: float           # continuously compounded, over r
    expected_loss_rate: float      # 1 - debt value / risk-free value of debt


def merton(asset_value: float, asset_vol: float, debt_face: float, maturity: float, r: float) -> MertonResult:
    v, s, d, t = asset_value, asset_vol, debt_face, maturity
    if min(v, s, d, t) <= 0:
        raise ValueError("asset value, asset vol, debt and maturity must be positive")
    sd = s * math.sqrt(t)
    d1 = (math.log(v / d) + (r + 0.5 * s * s) * t) / sd
    d2 = d1 - sd
    equity = v * norm.cdf(d1) - d * math.exp(-r * t) * norm.cdf(d2)
    equity_vol = norm.cdf(d1) * v * s / equity
    debt = v - equity
    riskless = d * math.exp(-r * t)
    spread = -math.log(debt / d) / t - r
    return MertonResult(v, s, equity, equity_vol, d2, float(norm.cdf(-d2)), debt, spread, 1 - debt / riskless)


def calibrate_from_equity(equity_value: float, equity_vol: float, debt_face: float, maturity: float,
                          r: float) -> MertonResult:
    """Solve E = BS_call(V, sigma_V) and sigma_E E = N(d1) sigma_V V for (V, sigma_V)."""
    def equations(x):
        v, s = math.exp(x[0]), math.exp(x[1])  # log-parametrised: stays positive
        m = merton(v, s, debt_face, maturity, r)
        return [m.equity_value / equity_value - 1.0, m.equity_vol / equity_vol - 1.0]

    v0 = equity_value + debt_face * math.exp(-r * maturity)
    s0 = equity_vol * equity_value / v0
    sol, _, _, msg = fsolve(equations, [math.log(v0), math.log(s0)], full_output=True, xtol=1e-14)
    if max(abs(e) for e in equations(sol)) > 1e-10:  # judge by the residual, not the step-size flag
        raise RuntimeError(f"Merton calibration failed: {msg}")
    return merton(math.exp(sol[0]), math.exp(sol[1]), debt_face, maturity, r)
