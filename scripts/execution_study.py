#!/usr/bin/env python3
"""TWAP vs VWAP on real NASDAQ minute bars (30 Dec 2019, rebuilt from ITCH).

For each of the 100 most traded symbols, a buy order for a fraction of the
stock's NASDAQ regular-hours volume is scheduled across the 390 minutes
from 09:30 to 16:00 and filled at each minute's own VWAP:

  * TWAP: equal shares per minute.
  * VWAP: shares in proportion to a forecast intraday profile. The forecast
    for each stock is the average profile of the *other* 99 stocks
    (leave-one-out), so the stock's own volume is never used to schedule it.
  * Oracle VWAP: the stock's realised profile, an unattainable lower bound.

Every schedule runs under the same participation cap (at most 25% of each
minute's NASDAQ volume; the excess carries to later minutes, and anything
left at 16:00 is reported as unfilled). Costs are measured against the
stock's market VWAP for the day, first without market impact (pure tracking
error) and then with square-root temporary impact
(cost = 1% x sqrt(participation), an illustrative coefficient).

Input: docs/data/nasdaq_itch_20191230_minute_bars_top100.csv (from itch_replay --bars).
Output: docs/results/execution_study.txt
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from qsl.execution import cap_participation, simulate, twap, volume_profile, vwap  # noqa: E402

BARS = ROOT / "docs" / "data" / "nasdaq_itch_20191230_minute_bars_top100.csv"
OUT = ROOT / "docs" / "results" / "execution_study.txt"
IMPACT = 0.01
MAX_RATE = 0.25
FRACTIONS = (0.01, 0.05, 0.20)


def load():
    vol, notional = defaultdict(lambda: np.zeros(390)), defaultdict(lambda: np.zeros(390))
    with BARS.open() as f:
        for r in csv.DictReader(f):
            m = int(r["minute"])
            vol[r["symbol"]][m] = float(r["volume"])
            notional[r["symbol"]][m] = float(r["notional"])
    syms = list(vol)
    return syms, np.array([vol[s] for s in syms]), np.array([notional[s] for s in syms])


def clock(minute: int) -> str:
    t = 9 * 60 + 30 + int(minute)
    return f"{t // 60:02d}:{t % 60:02d}"


def main() -> int:
    syms, vol, notional = load()
    n = len(syms)
    profile = volume_profile(vol)
    lines = [f"TWAP vs VWAP on real NASDAQ minute bars: {n} most traded symbols, 30 Dec 2019 (ITCH replay)", ""]
    first, mid, last = profile[:30].sum(), profile[30:-30].sum(), profile[-30:].sum()
    lines += ["Average intraday profile (each stock weighted equally):",
              f"  first 30 min {first:6.1%}   middle 330 min {mid:6.1%}   last 30 min {last:6.1%}",
              f"  busiest minute {clock(profile.argmax())} is "
              f"{profile.max() / profile.mean():.1f}x the average minute; quietest is "
              f"{profile.min() / profile.mean():.2f}x", ""]
    lines.append(f"Order = a fraction of the stock's NASDAQ regular-hours volume; participation capped at {MAX_RATE:.0%}"
                 " of each minute.")
    lines.append(f"{'order':>6s} {'schedule':>12s} | {'|slippage| vs VWAP, no impact':^29s} | "
                 f"{'cost vs VWAP, sqrt impact':^25s} | {'unfilled':>8s}")
    lines.append(f"{'':>6s} {'':>12s} | {'median':>11s} {'mean':>8s} {'p90':>8s} | {'median':>11s} {'mean':>11s} |"
                 f" {'mean':>8s}")
    for frac in FRACTIONS:
        results = {"TWAP": [], "VWAP (LOO)": [], "VWAP oracle": []}
        for i in range(n):
            q = int(round(frac * vol[i].sum()))
            loo = volume_profile(np.delete(vol, i, axis=0))
            plans = {"TWAP": twap(q, 390), "VWAP (LOO)": vwap(q, loo), "VWAP oracle": vwap(q, vol[i])}
            for name, plan in plans.items():
                done, unfilled = cap_participation(plan, vol[i], MAX_RATE)
                free = simulate(done, vol[i], notional[i])
                hit = simulate(done, vol[i], notional[i], impact_coef=IMPACT)
                results[name].append((abs(free.vwap_slippage_bps), hit.vwap_slippage_bps, unfilled / q))
        for name, rows in results.items():
            a = np.array(rows)
            lines.append(f"{frac:>6.0%} {name:>12s} | {np.median(a[:, 0]):8.2f} bp {a[:, 0].mean():5.2f} bp"
                         f" {np.percentile(a[:, 0], 90):5.1f} bp | {np.median(a[:, 1]):8.2f} bp {a[:, 1].mean():8.2f} bp |"
                         f" {a[:, 2].mean():8.2%}")
        lines.append("")
    # Split the no-impact tracking error by how much the stock moved: on a
    # news-driven day no generic volume curve can track the price path.
    rows = []
    for i in range(n):
        q = int(round(0.01 * vol[i].sum()))
        loo = volume_profile(np.delete(vol, i, axis=0))
        t = simulate(cap_participation(twap(q, 390), vol[i], MAX_RATE)[0], vol[i], notional[i]).vwap_slippage_bps
        v = simulate(cap_participation(vwap(q, loo), vol[i], MAX_RATE)[0], vol[i], notional[i]).vwap_slippage_bps
        active = vol[i] > 0
        px = notional[i][active] / vol[i][active]
        rows.append((abs(t), abs(v), px.max() / px.min() - 1))
    r = np.array(rows)
    calm, wild = r[r[:, 2] <= 0.10], r[r[:, 2] > 0.10]
    lines += ["No-impact tracking error at 1% of volume, split by the stock's intraday high/low range:",
              f"{'':24s} {'stocks':>6s} {'TWAP median':>12s} {'VWAP median':>12s} {'TWAP p90':>9s} {'VWAP p90':>9s}"
              f" {'VWAP better':>12s}"]
    for label, g in (("range <= 10%", calm), ("range > 10% (news days)", wild)):
        lines.append(f"{label:24s} {len(g):6d} {np.median(g[:, 0]):9.2f} bp {np.median(g[:, 1]):9.2f} bp"
                     f" {np.percentile(g[:, 0], 90):6.1f} bp {np.percentile(g[:, 1], 90):6.1f} bp"
                     f" {int((g[:, 1] < g[:, 0]).sum()):>5d} of {len(g)}")
    lines += ["",
              "Reading the results:",
              "  * Tracking error (no impact): for ordinary stocks a volume profile borrowed from other stocks",
              "    beats TWAP on both the median and the tail. On news days (18 of the 100 symbols moved more than",
              "    10% intraday) neither schedule can track VWAP and the generic profile can be worse, which is why",
              "    production VWAP models forecast each stock's profile from its own history and adapt intraday.",
              "    One day of data cannot do that, so these VWAP numbers are a pessimistic estimate.",
              "  * Impact cost: VWAP is clearly cheaper. TWAP trades the same amount in quiet midday minutes,",
              "    so its participation, and square-root impact, peak where liquidity is thinnest. Under the",
              "    25% cap, large orders (20% of volume) also go unfilled more often with TWAP than with VWAP.",
              "  * The oracle row (the stock's own realised profile) is the floor for a perfect forecast."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
