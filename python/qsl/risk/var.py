"""Value-at-Risk and Expected Shortfall.

Sign convention: inputs are P&L (gains positive); VaR and ES are reported as
positive loss amounts at confidence level ``alpha`` (e.g. 0.99).

* historical VaR = the alpha-quantile of losses, inf{x : F_n(x) >= alpha}
* historical ES  = VaR + E[(L - VaR)^+] / (1 - alpha)  (Rockafellar-Uryasev /
  Acerbi-Tasche form: coherent and exact for the empirical distribution)
* parametric (delta-normal), Cornish-Fisher and portfolio Euler allocation
* full-revaluation vs delta-gamma VaR for an options book (C BSM kernel)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import kurtosis, norm, skew

from ..pricing import black_scholes, greeks


def _losses(pnl) -> np.ndarray:
    x = np.asarray(pnl, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("pnl must be a non-empty 1-D array")
    return -x


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def historical_var(pnl, alpha: float = 0.99) -> float:
    _check_alpha(alpha)
    return float(np.quantile(_losses(pnl), alpha, method="inverted_cdf"))


def historical_es(pnl, alpha: float = 0.975) -> float:
    _check_alpha(alpha)
    losses = _losses(pnl)
    var = float(np.quantile(losses, alpha, method="inverted_cdf"))
    return var + float(np.mean(np.maximum(losses - var, 0.0))) / (1.0 - alpha)


def parametric_var(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """Delta-normal VaR of a P&L ~ N(mu, sigma^2)."""
    _check_alpha(alpha)
    return -mu + sigma * norm.ppf(alpha)


def parametric_es(mu: float, sigma: float, alpha: float = 0.975) -> float:
    _check_alpha(alpha)
    return -mu + sigma * norm.pdf(norm.ppf(alpha)) / (1.0 - alpha)


def cornish_fisher_var(pnl, alpha: float = 0.99) -> float:
    """Modified VaR: the normal quantile corrected for sample skewness and
    excess kurtosis (Cornish-Fisher expansion)."""
    _check_alpha(alpha)
    x = np.asarray(pnl, dtype=float)
    z = norm.ppf(1.0 - alpha)
    s, k = skew(x), kurtosis(x)  # kurtosis() is excess kurtosis
    w = z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s * s / 36
    return float(-(x.mean() + x.std(ddof=1) * w))


@dataclass(frozen=True)
class PortfolioVaR:
    var: float
    marginal: np.ndarray   # dVaR / dw_i
    component: np.ndarray  # w_i * marginal_i, sums to var (Euler)


def portfolio_var(positions: Sequence[float], cov: np.ndarray, alpha: float = 0.99) -> PortfolioVaR:
    """Delta-normal VaR of linear exposures with Euler risk allocation.
    VaR is homogeneous of degree one in the positions, so the component
    VaRs add up exactly to the total (Euler's theorem)."""
    _check_alpha(alpha)
    w = np.asarray(positions, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sigma = float(np.sqrt(w @ cov @ w))
    z = norm.ppf(alpha)
    marginal = z * (cov @ w) / sigma
    return PortfolioVaR(z * sigma, marginal, w * marginal)


@dataclass(frozen=True)
class OptionPosition:
    kind: str        # "call" or "put"
    strike: float
    maturity: float  # years
    quantity: float  # contracts (negative = short)


@dataclass(frozen=True)
class OptionsVaR:
    full_revaluation: float
    delta_gamma: float
    delta_only: float


def options_var(book: Sequence[OptionPosition], spot: float, vol: float, r: float, q: float = 0.0, *,
                horizon_days: float = 1.0, alpha: float = 0.99, n_scenarios: int = 200_000,
                seed: int = 7) -> OptionsVaR:
    """Monte Carlo VaR of an options book on one underlying, three ways:
    full revaluation (every scenario repriced with the C BSM kernel), the
    delta-gamma approximation, and delta only. Theta is included in all three."""
    _check_alpha(alpha)
    h = horizon_days / 365.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_scenarios)
    shocked = spot * np.exp((r - q - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z)
    ds = shocked - spot
    pnl_full = np.zeros(n_scenarios)
    pnl_dg = np.zeros(n_scenarios)
    pnl_d = np.zeros(n_scenarios)
    for pos in book:
        g = greeks(pos.kind, spot, pos.strike, pos.maturity, r, q, vol)
        later = black_scholes(pos.kind, shocked, pos.strike, pos.maturity - h, r, q, vol)
        pnl_full += pos.quantity * (later - g.price)
        pnl_dg += pos.quantity * (g.delta * ds + 0.5 * g.gamma * ds * ds + g.theta * h)
        pnl_d += pos.quantity * (g.delta * ds + g.theta * h)
    return OptionsVaR(historical_var(pnl_full, alpha), historical_var(pnl_dg, alpha), historical_var(pnl_d, alpha))
