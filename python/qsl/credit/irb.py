"""Basel IRB capital and the Vasicek one-factor model behind it.

Under the asymptotic single risk factor (ASRF) model a large homogeneous
loan book's default rate at confidence q is the Vasicek quantile
    N( (N^-1(PD) + sqrt(R) N^-1(q)) / sqrt(1 - R) ).
Basel II/III IRB capital for corporate exposures is that 99.9% loss minus
expected loss, scaled by LGD and a maturity adjustment:
    K = LGD [N(...) - PD] (1 + (M - 2.5) b) / (1 - 1.5 b),  RWA = 12.5 K EAD.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

PD_FLOOR = 0.0003  # Basel II corporate PD floor (3 bp; Basel III final raises it to 5 bp)


def vasicek_quantile(pd: float, rho: float, q: float) -> float:
    """Default-rate quantile of an infinitely granular portfolio."""
    return float(norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho)))


def corporate_correlation(pd: float) -> float:
    """Supervisory asset correlation, 12% for high PD up to 24% for low PD."""
    w = (1 - np.exp(-50 * pd)) / (1 - np.exp(-50))
    return float(0.12 * w + 0.24 * (1 - w))


def maturity_adjustment(pd: float, maturity: float) -> float:
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    return float((1 + (maturity - 2.5) * b) / (1 - 1.5 * b))


def capital_requirement(pd: float, lgd: float, maturity: float = 2.5) -> float:
    """Capital K per unit of EAD for a corporate exposure (unexpected loss at 99.9%)."""
    pd = max(pd, PD_FLOOR)
    if pd >= 1.0:
        return 0.0  # defaulted: covered by provisions (best-estimate EL) instead
    r = corporate_correlation(pd)
    return lgd * (vasicek_quantile(pd, r, 0.999) - pd) * maturity_adjustment(pd, maturity)


def risk_weighted_assets(pd: float, lgd: float, ead: float, maturity: float = 2.5) -> float:
    return 12.5 * capital_requirement(pd, lgd, maturity) * ead


def expected_loss(pd: float, lgd: float, ead: float) -> float:
    return pd * lgd * ead


def simulate_default_rates(pd: float, rho: float, n_obligors: int, n_scenarios: int, seed: int = 0) -> np.ndarray:
    """One-factor Gaussian copula: obligor i defaults when
    sqrt(rho) Z + sqrt(1 - rho) eps_i < N^-1(PD). Returns the portfolio
    default rate per scenario (exact binomial draw given Z)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_scenarios)
    conditional_pd = norm.cdf((norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
    return rng.binomial(n_obligors, conditional_pd) / n_obligors
