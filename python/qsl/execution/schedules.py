"""TWAP and VWAP execution schedules and a bar-level execution simulator.

A parent order is split into child orders, one per bar (for example one per
minute). TWAP trades equal amounts per bar; VWAP trades in proportion to a
forecast intraday volume profile, so that the average fill tracks the
market's volume-weighted average price. Child sizes are integers that always
sum exactly to the parent size (largest-remainder rounding).

The simulator fills each child at the bar's own VWAP plus a temporary impact
that grows with the child's share of the bar's volume (participation):

    fill = bar_vwap * (1 + side * coef * participation ** exponent)

exponent = 0.5 is the empirical square-root law of market impact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def allocate(total: int, weights) -> np.ndarray:
    """Splits an integer `total` in proportion to non-negative weights.

    Largest-remainder method: floors of the exact shares, then one extra unit
    to the largest fractional parts, so the result sums to `total` exactly.
    """
    w = np.asarray(weights, dtype=float)
    if total < 0 or (w < 0).any() or w.sum() <= 0:
        raise ValueError("need total >= 0 and non-negative weights with a positive sum")
    exact = total * w / w.sum()
    out = np.floor(exact).astype(np.int64)
    short = total - int(out.sum())
    if short:
        order = np.argsort(-(exact - out), kind="stable")
        out[order[:short]] += 1
    return out


def twap(total: int, bars: int) -> np.ndarray:
    return allocate(total, np.ones(bars))


def vwap(total: int, profile) -> np.ndarray:
    """Schedule proportional to a forecast volume profile (any positive scale)."""
    return allocate(total, profile)


def cap_participation(schedule, bar_volume, max_rate: float) -> tuple[np.ndarray, int]:
    """Limits each child order to `max_rate` of its bar's market volume,
    carrying the excess into later bars, as a VWAP or POV algo with a
    participation cap does. Returns (executed shares per bar, unfilled shares).

    Bars without volume absorb nothing, so their orders also carry forward:
    on a single venue even liquid stocks have minutes with no print (on
    30 Dec 2019 INTC had no NASDAQ execution from 12:11:52 to 12:13:30).
    """
    if max_rate <= 0:
        raise ValueError("max_rate must be positive")
    q = np.asarray(schedule, dtype=np.int64)
    vol = np.asarray(bar_volume, dtype=float)
    done = np.zeros_like(q)
    carry = 0
    for i in range(len(q)):
        want = int(q[i]) + carry
        room = int(np.floor(max_rate * vol[i])) if np.isfinite(max_rate) else (want if vol[i] > 0 else 0)
        done[i] = min(want, room)
        carry = want - int(done[i])
    return done, carry


def volume_profile(volumes: np.ndarray) -> np.ndarray:
    """Average intraday profile from a (stocks x bars) volume matrix: each
    stock's day is normalised to sum to one before averaging, so heavily
    traded names do not dominate the shape."""
    v = np.asarray(volumes, dtype=float)
    day = v.sum(axis=1, keepdims=True)
    keep = day[:, 0] > 0
    return (v[keep] / day[keep]).mean(axis=0)


@dataclass(frozen=True)
class ExecutionResult:
    avg_price: float
    market_vwap: float
    arrival_price: float
    vwap_slippage_bps: float      #: cost vs market VWAP (positive = worse than VWAP)
    shortfall_bps: float          #: cost vs arrival price (implementation shortfall)
    max_participation: float


def simulate(schedule, bar_volume, bar_notional, *, side: int = 1, impact_coef: float = 0.0,
             impact_exponent: float = 0.5, arrival_price: float | None = None) -> ExecutionResult:
    """Fills `schedule` (shares per bar) against market bars.

    side = +1 buys, -1 sells; costs are reported as positive when the fill is
    worse than the benchmark for that side. Bars with no volume cannot absorb
    a child order, so the schedule must be zero there.
    """
    q = np.asarray(schedule, dtype=float)
    vol = np.asarray(bar_volume, dtype=float)
    notional = np.asarray(bar_notional, dtype=float)
    if q.shape != vol.shape or vol.shape != notional.shape:
        raise ValueError("schedule and bars must have the same length")
    if ((q > 0) & (vol <= 0)).any():
        raise ValueError("schedule trades in a bar with no market volume")
    if side not in (1, -1):
        raise ValueError("side must be +1 (buy) or -1 (sell)")
    active = vol > 0
    bar_vwap = np.zeros_like(vol)
    bar_vwap[active] = notional[active] / vol[active]
    part = np.zeros_like(vol)
    part[active] = q[active] / vol[active]
    fill = bar_vwap * (1.0 + side * impact_coef * part**impact_exponent)
    avg = float((fill * q).sum() / q.sum())
    mkt = float(notional.sum() / vol.sum())
    arrival = float(bar_vwap[np.argmax(active)]) if arrival_price is None else float(arrival_price)
    return ExecutionResult(avg, mkt, arrival, side * (avg - mkt) / mkt * 1e4, side * (avg - arrival) / arrival * 1e4,
                           float(part.max()))
