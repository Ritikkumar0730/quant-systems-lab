#!/usr/bin/env python3
"""Checks the closing-cross prices rebuilt from NASDAQ's ITCH file against an
independent source: the official daily closes published by Yahoo Finance.

For NASDAQ-listed securities the NASDAQ Closing Cross sets the official
closing price, so the 'Q' cross print with cross type 'C' in the ITCH feed
must equal the official close. Yahoo's daily bars are split-adjusted, so the
script also fetches each symbol's split history and undoes later splits.

Inputs: data/nasdaq_20191230_crosses.csv, written by
    gzip -dc data/12302019.NASDAQ_ITCH50.gz | build/bin/itch_replay - --crosses data/nasdaq_20191230_crosses.csv
Output: docs/results/itch_closing_cross_vs_official.txt

Needs network access (one small request per symbol).
"""
from __future__ import annotations

import csv
import json
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROSSES = ROOT / "data" / "nasdaq_20191230_crosses.csv"
OUT = ROOT / "docs" / "results" / "itch_closing_cross_vs_official.txt"
DAY = datetime(2019, 12, 30, tzinfo=timezone.utc)
TOP = 80  # largest closing crosses by shares
# Yahoo quotes to the cent (more digits below $1) and the cross can print at
# a half cent, so agreement within half a cent is agreement to the source's precision.
TOLERANCE = 0.005 + 1e-9
# Symbols whose ticker now belongs to a different company, so Yahoo's history
# under that ticker is not the 2019 security.
REUSED_TICKERS = {
    "CZR": "Caesars Entertainment Corp in 2019; Eldorado Resorts took the ticker after the July 2020 merger",
}


def _tls_context() -> ssl.SSLContext:
    """Verified TLS. python.org builds on macOS ship without a CA bundle, so
    fall back to certifi or the system bundle; verification is never disabled."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"] == 0:
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except ImportError:
            ctx.load_verify_locations("/etc/ssl/cert.pem")
    return ctx


_TLS = _tls_context()


def official_close(symbol: str) -> float | None:
    """Unadjusted official close on DAY from Yahoo's chart endpoint."""
    start = int(DAY.timestamp())
    now = int(time.time())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={now}"
           f"&interval=1d&events=split")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=_TLS) as resp:
        res = json.load(resp)["chart"]["result"][0]
    stamps = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    day0 = next((c for t, c in zip(stamps, closes) if datetime.fromtimestamp(t, timezone.utc).date() == DAY.date()),
                None)
    if day0 is None:
        return None
    factor = 1.0
    for split in res.get("events", {}).get("splits", {}).values():
        if split["date"] > start + 86400:
            factor *= split["numerator"] / split["denominator"]
    # Yahoo serves single-precision floats (19.54 arrives as 19.540000915...),
    # so round away the float noise before comparing.
    return round(day0 * factor, 4)


def main() -> int:
    if not CROSSES.exists():
        print(f"missing {CROSSES}; run itch_replay with --crosses first", file=sys.stderr)
        return 1
    with CROSSES.open() as f:
        rows = [r for r in csv.DictReader(f) if float(r["close_cross_price"]) > 0]
    rows.sort(key=lambda r: int(r["close_cross_shares"]), reverse=True)

    lines = ["NASDAQ Closing Cross (rebuilt from the 30 Dec 2019 ITCH file) vs official close (Yahoo Finance,",
             f"split-adjustment undone), for the {TOP} largest closing crosses by shares.", "",
             f"{'symbol':8s} {'ITCH cross':>11s} {'official':>11s} {'diff':>8s}  shares in cross"]
    matched, mismatched, unavailable, reused = [], [], [], []
    for r in rows[:TOP]:
        sym, cross = r["symbol"], float(r["close_cross_price"])
        if sym in REUSED_TICKERS:
            reused.append(sym)
            lines.append(f"{sym:8s} {cross:11.4f} {'skipped':>11s}           ticker reused: {REUSED_TICKERS[sym]}")
            continue
        try:
            official = official_close(sym)
        except Exception as exc:  # delisted or renamed since 2019: report, do not hide
            official, reason = None, type(exc).__name__
        else:
            reason = "no bar for that day"
        time.sleep(0.3)
        if official is None:
            unavailable.append(sym)
            lines.append(f"{sym:8s} {cross:11.4f} {'n/a':>11s}           ({reason}: delisted or renamed since 2019)")
            continue
        diff = cross - official
        (matched if abs(diff) <= TOLERANCE else mismatched).append(sym)
        lines.append(f"{sym:8s} {cross:11.4f} {official:11.4f} {diff:8.4f}  {int(r['close_cross_shares']):>12,d}")
    checked = len(matched) + len(mismatched)
    lines += ["",
              f"agree to the source's precision: {len(matched)} of {checked} symbols with an official close available",
              f"no official close available (delisted/renamed): {len(unavailable)}; ticker reused, skipped: {len(reused)}",
              f"mismatches: {', '.join(mismatched) if mismatched else 'none'}"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if checked and not mismatched else 1


if __name__ == "__main__":
    sys.exit(main())
