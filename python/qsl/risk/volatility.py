"""Volatility models and estimators.

* EWMA (RiskMetrics, lambda = 0.94)
* GARCH(1,1) maximum likelihood with variance forecasts
* range-based estimators from OHLC bars: Parkinson, Garman-Klass,
  Rogers-Satchell, plus close-to-close
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


def ewma_variance(returns, lam: float = 0.94, initial: float | None = None) -> np.ndarray:
    """sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}; element t is
    the forecast for day t made with information up to t-1."""
    r = np.asarray(returns, dtype=float)
    s0 = float(np.var(r)) if initial is None else initial
    out = np.empty(r.size)
    out[0] = s0
    if r.size > 1:
        zi = np.array([lam * s0])
        out[1:], _ = lfilter([1 - lam], [1, -lam], r[:-1] ** 2, zi=zi)
    return out


@dataclass(frozen=True)
class Garch11:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    variances: np.ndarray  # conditional variance path sigma^2_t

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half."""
        return float(np.log(0.5) / np.log(self.persistence))

    def forecast(self, horizon: int, last_return: float) -> np.ndarray:
        """E[sigma^2_{T+h}], h = 1..horizon:  V_L + (a+b)^(h-1) (sigma^2_{T+1} - V_L)."""
        next_var = self.omega + self.alpha * last_return ** 2 + self.beta * self.variances[-1]
        h = np.arange(horizon)
        return self.long_run_variance + self.persistence ** h * (next_var - self.long_run_variance)


def _garch_variances(r: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    out = np.empty(r.size)
    out[0] = s0
    if r.size > 1:
        out[1:], _ = lfilter([1.0], [1.0, -beta], omega + alpha * r[:-1] ** 2, zi=np.array([beta * s0]))
    return out


def _neg_loglik(params: np.ndarray, r: np.ndarray, s0: float) -> float:
    omega, alpha, beta = params
    var = _garch_variances(r, omega, alpha, beta, s0)
    if np.any(var <= 0):
        return 1e300
    return 0.5 * float(np.sum(np.log(2 * np.pi * var) + r * r / var))


def fit_garch11(returns) -> Garch11:
    """Gaussian quasi-MLE of sigma^2_t = omega + alpha r^2_{t-1} + beta sigma^2_{t-1}
    subject to omega > 0, alpha, beta >= 0 and alpha + beta < 1 (stationarity).
    Returns should be de-meaned (daily log returns usually are close enough)."""
    r = np.asarray(returns, dtype=float)
    if r.size < 100:
        raise ValueError("need at least 100 observations")
    s0 = float(np.var(r))
    scale = s0  # optimise omega in units of the sample variance for conditioning
    objective = lambda p: _neg_loglik(np.array([p[0] * scale, p[1], p[2]]), r, s0)  # noqa: E731
    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.80), (0.03, 0.95)]:
        res = minimize(objective, x0=[1 - a0 - b0, a0, b0], method="SLSQP",
                       bounds=[(1e-8, 10.0), (0.0, 1.0), (0.0, 1.0)],
                       constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}],
                       options={"ftol": 1e-12, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x[0] * scale, best.x[1], best.x[2]
    return Garch11(omega, alpha, beta, -float(best.fun), _garch_variances(r, omega, alpha, beta, s0))


# ---------------------------------------------------- range-based estimators


def close_to_close(close) -> float:
    r = np.diff(np.log(np.asarray(close, dtype=float)))
    return float(np.std(r, ddof=1))


def parkinson(high, low) -> float:
    """Parkinson (1980): uses the high-low range; ~5x more efficient than
    close-to-close for driftless diffusion."""
    hl = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
    return float(np.sqrt(np.mean(hl ** 2) / (4 * np.log(2))))


def garman_klass(open_, high, low, close) -> float:
    o, h, l, c = (np.asarray(a, dtype=float) for a in (open_, high, low, close))
    hl, co = np.log(h / l), np.log(c / o)
    return float(np.sqrt(np.mean(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2)))


def rogers_satchell(open_, high, low, close) -> float:
    """Rogers-Satchell (1991): unbiased under non-zero drift."""
    o, h, l, c = (np.asarray(a, dtype=float) for a in (open_, high, low, close))
    return float(np.sqrt(np.mean(np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))))
