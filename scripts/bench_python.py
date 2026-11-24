#!/usr/bin/env python3
"""Native vs NumPy benchmarks behind the README's pricing rows.

1. Black-Scholes on 1,000,000 random options: the C batch kernel (via ctypes)
   against vectorised NumPy/SciPy and a plain Python loop.
2. American put on a 2001-step CRR tree: the C++ engine (via ctypes) against
   the fastest NumPy version (vectorised across each time step).

Each timing is the best of 5 runs. Before any timing is reported, the fast and
slow versions are checked to give the same prices, so a speed-up can never
come from a wrong answer.

Run from the repository root after ``make``:
    python3 scripts/bench_python.py            # print only
    python3 scripts/bench_python.py --write    # also update docs/results/bench_bsm.txt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from qsl import pricing as P  # noqa: E402

REPS = 5


def best_of(fn, reps: int = REPS):
    best, result = math.inf, None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call_scalar(s: float, k: float, t: float, r: float, q: float, v: float) -> float:
    sd = v * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q) * t) / sd + 0.5 * sd
    return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d1 - sd)


def bench_black_scholes(n: int = 1_000_000, loop_n: int = 50_000) -> str:
    rng = np.random.default_rng(0)
    args = [
        rng.uniform(50, 150, n),   # spot
        rng.uniform(50, 150, n),   # strike
        rng.uniform(0.1, 2.0, n),  # maturity (years)
        np.full(n, 0.03),          # rate
        np.full(n, 0.01),          # dividend yield
        rng.uniform(0.1, 0.6, n),  # volatility
    ]

    t_c, px_c = best_of(lambda: P.black_scholes("call", *args))
    t_np, px_np = best_of(lambda: P.black_scholes_numpy("call", *args))

    sub = [a[:loop_n] for a in args]
    t_loop, px_loop = best_of(lambda: [bs_call_scalar(*row) for row in zip(*sub)], reps=1)
    t_loop *= n / loop_n  # scale the subset timing to n options

    np.testing.assert_allclose(px_c, px_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(px_c[:loop_n], px_loop, rtol=1e-10, atol=1e-10)

    return (
        f"Black-Scholes, {n:,} options (best of {REPS}; prices agree to 1e-10)\n"
        f"  C batch kernel via ctypes   {t_c * 1e3:8.1f} ms   {t_c / n * 1e9:7.1f} ns/option\n"
        f"  NumPy/SciPy vectorised      {t_np * 1e3:8.1f} ms   {t_np / n * 1e9:7.1f} ns/option\n"
        f"  pure Python loop (scaled)   {t_loop * 1e3:8.1f} ms   {t_loop / n * 1e9:7.1f} ns/option\n"
    )


def numpy_crr_american_put(s, k, t, r, q, v, steps):
    dt = t / steps
    u = math.exp(v * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    values = np.maximum(k - s * u**j * d ** (steps - j), 0.0)
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        continuation = disc * (p * values[1 : i + 2] + (1.0 - p) * values[: i + 1])
        values = np.maximum(continuation, k - s * u**j * d ** (i - j))
    return float(values[0])


def bench_tree(steps: int = 2001) -> str:
    s, k, t, r, q, v = 100.0, 105.0, 1.0, 0.05, 0.02, 0.3
    t_cpp, px_cpp = best_of(lambda: P.binomial("put", s, k, t, r, q, v, steps, american=True, tree="crr"))
    t_np, px_np = best_of(lambda: numpy_crr_american_put(s, k, t, r, q, v, steps))

    if not math.isclose(px_cpp, px_np, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"tree prices disagree: C++ {px_cpp!r} vs NumPy {px_np!r}")

    return (
        f"American put, CRR tree with {steps} steps (best of {REPS})\n"
        f"  C++ engine via ctypes       {t_cpp * 1e3:7.2f} ms   price {px_cpp:.10f}\n"
        f"  NumPy, vectorised per step  {t_np * 1e3:7.2f} ms   price {px_np:.10f}   ({t_np / t_cpp:.1f}x slower)\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="update docs/results/bench_bsm.txt")
    opts = parser.parse_args()
    out = bench_black_scholes() + "\n" + bench_tree()
    print(out, end="")
    if opts.write:
        (ROOT / "docs/results/bench_bsm.txt").write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
