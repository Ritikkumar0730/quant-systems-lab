"""Almgren-Chriss optimal execution (Almgren & Chriss, "Optimal Execution of
Portfolio Transactions", J. Risk 2000), discrete-time version.

Sell X shares over N intervals of length tau = T/N. Holdings x_0 = X, ...,
x_N = 0, with n_k = x_{k-1} - x_k sold in interval k.

    price            S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau g(n_k / tau),  g(v) = gamma v
    execution price  S~_k = S_{k-1} - h(n_k / tau),                         h(v) = eps sgn(v) + eta v

The implementation shortfall IS = X S_0 - sum_k n_k S~_k has

    E[IS]   = gamma X^2 / 2 + eps sum|n_k| + (eta_tilde / tau) sum n_k^2,   eta_tilde = eta - gamma tau / 2
    Var[IS] = sigma^2 tau sum_k x_k^2

and minimising E + lambda Var gives x_j = X sinh(kappa (T - t_j)) / sinh(kappa T), where
(2 / tau^2)(cosh(kappa tau) - 1) = lambda sigma^2 / eta_tilde. lambda -> 0 recovers the
straight line (TWAP); larger risk aversion front-loads the selling.

Units: shares, dollars, and one time unit for T, tau and sigma (e.g. days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACParams:
    shares: float   #: X, shares to sell (use a negative number to buy)
    horizon: float  #: T, in time units
    periods: int    #: N, number of trading intervals
    sigma: float    #: price volatility, $/share per sqrt(time unit)
    eta: float      #: temporary impact, $/share per (share/time unit)
    gamma: float    #: permanent impact, $/share per share
    eps: float = 0.0  #: fixed cost per share (half spread plus fees), $/share

    @property
    def tau(self) -> float:
        return self.horizon / self.periods

    @property
    def eta_tilde(self) -> float:
        et = self.eta - 0.5 * self.gamma * self.tau
        if et <= 0:
            raise ValueError("eta - gamma*tau/2 must be positive (temporary impact too small for this tau)")
        return et

    def __post_init__(self):
        if self.periods < 1 or self.horizon <= 0:
            raise ValueError("need periods >= 1 and horizon > 0")
        if self.sigma < 0 or self.eta <= 0 or self.gamma < 0 or self.eps < 0:
            raise ValueError("need sigma >= 0, eta > 0, gamma >= 0, eps >= 0")


def kappa(p: ACParams, risk_aversion: float) -> float:
    """Urgency kappa (per time unit) solving (2/tau^2)(cosh(kappa tau) - 1) = lambda sigma^2 / eta_tilde."""
    if risk_aversion < 0:
        raise ValueError("risk aversion must be non-negative")
    kt2 = risk_aversion * p.sigma**2 / p.eta_tilde
    return float(np.arccosh(1.0 + 0.5 * kt2 * p.tau**2) / p.tau)


def trajectory(p: ACParams, risk_aversion: float) -> np.ndarray:
    """Optimal holdings x_0..x_N (x_0 = X, x_N = 0)."""
    t = np.linspace(0.0, p.horizon, p.periods + 1)
    k = kappa(p, risk_aversion)
    if k * p.horizon < 1e-12:
        return p.shares * (1.0 - t / p.horizon)
    # sinh ratio written with exponentials scaled by exp(-kT) so large kT cannot overflow
    num = np.exp(-k * t) - np.exp(-k * (2 * p.horizon - t))
    den = 1.0 - np.exp(-2 * k * p.horizon)
    x = p.shares * num / den
    x[-1] = 0.0
    return x


def trades(holdings: np.ndarray) -> np.ndarray:
    """Shares sold in each interval, n_k = x_{k-1} - x_k."""
    return -np.diff(holdings)


def expected_cost(p: ACParams, holdings: np.ndarray) -> float:
    n = trades(holdings)
    return float(0.5 * p.gamma * p.shares**2 + p.eps * np.abs(n).sum() + p.eta_tilde / p.tau * (n**2).sum())


def cost_variance(p: ACParams, holdings: np.ndarray) -> float:
    return float(p.sigma**2 * p.tau * (holdings[1:] ** 2).sum())


def efficient_frontier(p: ACParams, risk_aversions) -> np.ndarray:
    """Rows of (lambda, E[IS], Var[IS]) for each risk aversion."""
    rows = []
    for lam in risk_aversions:
        x = trajectory(p, lam)
        rows.append((lam, expected_cost(p, x), cost_variance(p, x)))
    return np.array(rows)


def solve_numerically(p: ACParams, risk_aversion: float) -> np.ndarray:
    """Minimises E + lambda Var directly over x_1..x_{N-1}, without the closed form.

    For a monotone sale the objective is a quadratic in the interior
    holdings: (eta_tilde/tau) sum (x_{k-1} - x_k)^2 + lambda sigma^2 tau sum x_k^2
    (+ constants). Setting the gradient to zero gives a tridiagonal linear
    system, solved here densely. Used to validate ``trajectory``.
    """
    n = p.periods - 1
    if n == 0:
        return np.array([p.shares, 0.0])
    a = p.eta_tilde / p.tau
    b = risk_aversion * p.sigma**2 * p.tau
    m = np.zeros((n, n))
    np.fill_diagonal(m, 2 * a + b)
    idx = np.arange(n - 1)
    m[idx, idx + 1] = m[idx + 1, idx] = -a
    rhs = np.zeros(n)
    rhs[0] = a * p.shares  # x_0 = X enters the first equation; x_N = 0 adds nothing
    interior = np.linalg.solve(m, rhs)
    return np.concatenate(([p.shares], interior, [0.0]))


def simulate_shortfall(p: ACParams, holdings: np.ndarray, s0: float, paths: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo implementation shortfall of a schedule under the model's price dynamics."""
    n = trades(holdings)
    xi = rng.standard_normal((paths, p.periods))
    # S_{k-1} for k = 1..N: arithmetic random walk minus accumulated permanent impact
    shocks = p.sigma * np.sqrt(p.tau) * xi
    perm = p.gamma * n
    s = s0 + np.cumsum(shocks, axis=1) - np.cumsum(perm)[None, :]
    s_prev = np.concatenate([np.full((paths, 1), s0), s[:, :-1]], axis=1)
    exec_px = s_prev - (p.eps * np.sign(n) + p.eta / p.tau * n)[None, :]
    return p.shares * s0 - (exec_px * n[None, :]).sum(axis=1)
