"""Avellaneda & Stoikov (2008) optimal market making.

A dealer quoting around mid-price s with inventory q and risk aversion gamma
skews quotes around the reservation price
    r = s - q gamma sigma^2 (T - t)
and quotes a total spread
    delta_a + delta_b = gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k).
Market orders hit a quote at distance delta with intensity A exp(-k delta).

``simulate`` reproduces the paper's experiment (vectorised across paths) and
compares the inventory-aware strategy with a symmetric strategy quoting the
same average spread around the mid: inventory skewing trades a little
expected P&L for a much tighter P&L and inventory distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MarketMakingParams:
    s0: float = 100.0
    horizon: float = 1.0
    sigma: float = 2.0
    dt: float = 0.005
    gamma: float = 0.1
    k: float = 1.5
    a: float = 140.0


@dataclass(frozen=True)
class SimulationResult:
    pnl: np.ndarray             # terminal wealth per path: cash + q * S_T
    inventory: np.ndarray       # terminal inventory per path
    average_spread: float

    def summary(self) -> dict[str, float]:
        return {"mean_pnl": float(self.pnl.mean()), "std_pnl": float(self.pnl.std()),
                "mean_inventory": float(self.inventory.mean()), "std_inventory": float(self.inventory.std()),
                "spread": self.average_spread,
                "sharpe_like": float(self.pnl.mean() / self.pnl.std())}


def optimal_spread(p: MarketMakingParams, time_left: float) -> float:
    return p.gamma * p.sigma ** 2 * time_left + (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)


def simulate(p: MarketMakingParams = MarketMakingParams(), strategy: Literal["inventory", "symmetric"] = "inventory",
             n_paths: int = 1000, seed: int = 0) -> SimulationResult:
    rng = np.random.default_rng(seed)
    n_steps = int(round(p.horizon / p.dt))
    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    spreads = []
    for step in range(n_steps):
        time_left = p.horizon - step * p.dt
        spread = optimal_spread(p, time_left)
        spreads.append(spread)
        centre = s - q * p.gamma * p.sigma ** 2 * time_left if strategy == "inventory" else s
        ask, bid = centre + spread / 2, centre - spread / 2
        prob_ask = np.clip(p.a * np.exp(-p.k * (ask - s)) * p.dt, 0.0, 1.0)
        prob_bid = np.clip(p.a * np.exp(-p.k * (s - bid)) * p.dt, 0.0, 1.0)
        sold = rng.random(n_paths) < prob_ask    # our ask lifted: we sell one
        bought = rng.random(n_paths) < prob_bid  # our bid hit: we buy one
        cash += np.where(sold, ask, 0.0) - np.where(bought, bid, 0.0)
        q += bought.astype(float) - sold.astype(float)
        s = s + p.sigma * np.sqrt(p.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
    return SimulationResult(cash + q * s, q, float(np.mean(spreads)))
