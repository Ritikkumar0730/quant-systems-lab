# Design notes

Each section covers why the code is the way it is, which is usually the follow-up question in an
interview.

## Why three languages

- **C** for numerical kernels behind a stable ABI. Plain structs and status codes make them callable from
  anything (Python via ctypes, C++, Excel add-ins, legacy bank systems). There is no global state, so every
  function is thread-safe, and the only mutable object is the caller-owned RNG.
- **C++20** where data-structure design and zero-cost abstraction matter: the matching engine is a template
  over its trade listener (static dispatch, no virtual calls on the hot path), and concepts constrain the
  templates.
- **Python** for research, risk and glue, and as the place where independent oracles live (SciPy, QuantLib,
  pure-Python reference implementations).

Measured consequence: for element-wise closed forms NumPy is as fast as the C kernel, because both are
compiled loops. Native code earns its keep on sequential algorithms: a 2001-step American tree is 45x faster
in C++ than the best vectorised NumPy, and order matching cannot be vectorised at all.

## Matching engine (`cpp/include/qsl/lob`)

| Decision | Reason |
|---|---|
| Integer tick prices | Exact comparisons; no floating point on the matching path |
| Flat array of price levels, one per tick | O(1) level lookup. Bids and asks share the array because a tick can never hold both (the book is never crossed) |
| Hierarchical bitset of non-empty levels per side | When the best level empties, the next is found in O(log64 ticks) via `countr_zero`/`countl_zero`: 3 levels for 2^18 ticks instead of a linear scan |
| Intrusive doubly-linked FIFO per level, 32-bit indices into a preallocated pool | O(1) cancel; 32-byte orders (two per 64-byte line); no allocation after construction |
| Open-addressing id → slot map, linear probing, backward-shift deletion | Cache-friendly lookups, no tombstones, so probe lengths do not degrade under cancel-heavy flow |
| Reject before any state change (including a conservative `BookFull` check for GTC) | Every rejection is side-effect free, which keeps the state machine easy to reason about and test |
| Trades are always at the maker's price; a replace keeps priority only when it shrinks at the same price | Standard exchange semantics |

**Testing approach.** `ReferenceBook` re-implements the same contract in the most obvious way possible
(`std::map` and `std::list`). Random order flow drives both books and compares every result, every trade and
the full depth. The run also records coverage, so the test fails if the random flow stops reaching any
rejection path. `scripts/mutation_test.py` then injects six realistic bugs and checks that the suite catches
all of them.

## SPSC queue (`cpp/include/qsl/concurrency/spsc_queue.hpp`)

- Each index is written by one thread only, so acquire/release ordering is enough: no CAS, no locks,
  wait-free.
- Each side caches the other side's index and reloads it only when the queue looks full or empty.
- Indices are padded to 128 bytes. Apple M-series cores use 128-byte lines, and x86 prefetches 64-byte lines
  in pairs.
- **Measured insight:** with one message per push, throughput is capped by core-to-core latency (~26-40 ns
  per cache-line handoff), not by the queue code; a spin `yield` hint made no difference. Bulk push/pop
  publish many slots with one release store and reach 1+ billion msgs/s. In the pipeline benchmark the
  matching thread drains in bursts, so the queue costs almost nothing over single-threaded matching.

## Pricing engines

- **BSM (C):** `erfc` keeps lower-tail accuracy. The probit is Acklam's approximation plus one Halley step,
  and it reflects through 1 - p (exact for p ≥ 0.5) for exact antisymmetry. Implied vol uses Newton inside
  a maintained bisection bracket, so it cannot diverge, with a Manaster-Koehler starting guess.
- **Monte Carlo (C):** Welford-style single-pass moments, so there is no sample storage and no
  catastrophic cancellation. The control-variate beta is estimated by regression. For arithmetic Asians the
  control is the geometric Asian with its closed form (Kemna-Vorst).
- **Trees (C++):** Leisen-Reimer uses Peizer-Pratt inversion and converges at second order. Spot powers are
  precomputed, so the backward induction has no `pow()`.
- **Crank-Nicolson (C++):** log-spot grid centred exactly on the spot, Rannacher implicit half-steps at
  start-up, a cell-averaged payoff, and a Brennan-Schwartz projected tridiagonal solve for American exercise
  (exact for one exercise region touching the boundary).
- **Heston (C++):** Gil-Pelaez inversion written as one integral with the forward factored out (minimal
  oscillation), using the Albrecher et al. "little trap" characteristic function so the complex log stays on
  its principal branch. Integration is composite Gauss-Legendre on panels scaled to the total standard
  deviation, stopping when the tail is negligible.

## Risk and credit (Python)

- Historical VaR uses the standard quantile definition, `inf{x : F(x) ≥ α}`. ES uses the
  Rockafellar-Uryasev form, which is coherent and exact for the empirical distribution.
- Component VaR is the Euler allocation, so the components sum to the total VaR.
- Options VaR compares full revaluation (every scenario repriced through the C kernel) with the delta-gamma
  and delta-only approximations, which shows why negative-gamma books need full revaluation.
- Backtests: Kupiec POF, Christoffersen independence, conditional coverage and the Basel traffic light
  (zones derived from the binomial distribution, not hard-coded).
- IRB follows the Basel corporate formula, validated both against the published risk weight and against a
  one-factor Gaussian-copula simulation.

## Known limitations

- Bonds use unadjusted schedules with no holiday calendars, and assume regular coupon periods (no short or
  long first coupon).
- Curves are single-curve. Production rates desks use OIS discounting plus separate forwarding curves.
- The implied-vol solver is robust but is not Jäckel's "Let's Be Rational"; extreme far-wing quotes with
  tiny vega lose accuracy.
- One order book holds one instrument and the engine is single-threaded by design (threading happens at
  the queue boundary). There is no self-trade prevention and no persistence.
- CI (`.github/workflows/ci.yml`) builds and tests on Ubuntu and macOS with clang, including the
  sanitizer builds and the mutation test. Benchmarks in `docs/results` are from an Apple M4 laptop, not CI.
