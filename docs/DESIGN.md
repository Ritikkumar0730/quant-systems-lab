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
compiled loops. Native code earns its keep on sequential algorithms: a 2001-step American tree is about 43x faster
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
rejection path. `scripts/mutation_test.py` then injects six realistic bugs into the engine (and seven into
the ITCH handler) and checks that the suite catches all of them.

## ITCH 5.0 feed handler (`cpp/include/qsl/itch`)

ITCH is a market-by-order feed: NASDAQ reports every add, execution, cancel, delete and replace by order
reference, after it has done the matching. Rebuilding the book is bookkeeping, not matching, so this is a
separate component from the matching engine. It shares the same habits: integer prices, no allocation in
steady state, and a naive reference implementation for differential tests.

| Decision | Reason |
|---|---|
| Zero-copy parser; each message decoded into a small struct on the stack | No allocation and no virtual calls. Decoding every field costs 5 ns/msg (193M msgs/s) |
| Optional callbacks through C++20 `requires` expressions | A handler implements only what it needs. Missing callbacks compile away |
| Frame length checked against the spec for every type; bad frames counted and skipped | The stream stays in sync after corruption, and nothing is silently misread |
| Order store indexed by the order reference itself, data inline in the slot | NASDAQ references mostly increase and most updates hit young orders (median age: ~430 adds), so recent orders sit in a small, cache-resident region |
| Price levels in a sorted vector per side, best at the back, searched from the top | Two thirds of real updates land in the top four levels |
| Integrity counters instead of asserts (unknown order, overfill, locate mismatch...) | Replaying a real day doubles as a check of the parser offsets: one wrong offset makes these explode |

**Optimisation log (same 1 GB of real data, Apple M4, in memory):**

| Step | ns/msg | What the profiler showed |
|---|---|---|
| Hashed `OrderIndex` (SplitMix) + separate order pool | 85 | Two DRAM misses per message: index slot, then pool slot |
| Identity-indexed map, order data inline | 69 | Price-level binary search dominates |
| Scan levels from the top first; 16-byte level entries | 58 | First touch of each stock's ladder |
| Two-stage software prefetch, 16 and 8 frames ahead | 54 | Remaining time is deep-book searches |
| Load factor capped at 1/4 (was 1/2) | 50 | With the table ~45% full the same input took 143 ns/msg: references arrive in dense runs, which form long probe clusters under identity indexing |

The full day runs at 18.3M msgs/s (25 s wall time, including gunzip in a separate process).

**Validation.** The C++ tests check:
- every message type against bytes written field by field from the spec;
- framing across random chunk splits;
- the book against a naive `std::map` reference after every message, including deliberately invalid events;
- the order map against `std::unordered_map`, with keys chosen so that every one collides.

On real data:
- a separate Python decoder (struct layouts written from the spec) agrees on every field checksum over 20M
  messages;
- the full day replays with zero integrity errors and zero orders left at the close;
- the rebuilt closing-cross prices equal the official closes (52 of 52 symbols that still have an official
  close available).

## Execution (`python/qsl/execution`)

- **Almgren-Chriss** uses the discrete-time solution, written with exponentials so that large κT cannot
  overflow. It is validated against a direct minimisation of the same objective: setting the gradient to
  zero gives a tridiagonal system. A Monte Carlo of the model's price dynamics checks E[IS] and Var[IS].
- **Schedules** use largest-remainder rounding, so child orders are integers that always sum to the parent
  size.
- **The simulator** fills each child at its minute's VWAP plus square-root impact in participation. A
  participation cap carries any excess forward, as a real VWAP/POV algorithm does.
- **On real data** (`scripts/execution_study.py`), each stock's VWAP forecast is the average profile of the
  other 99 stocks, so the stock's own volume is never used to schedule it:
  - For ordinary stocks it beats TWAP on median and tail tracking error.
  - On news days (an intraday range above 10%) no generic profile can track VWAP. The study reports that
    split rather than averaging it away.

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
- The ITCH handler rebuilds displayed NASDAQ orders only. Hidden orders appear only as 'P' trade prints,
  so the book is the displayed book, as on any ITCH consumer. It reads files; there is no MoldUDP64/SoupBinTCP
  network layer.
- The execution study uses one day of data, so the VWAP forecast is cross-sectional (other stocks), not
  each stock's own history; the impact coefficient is illustrative, not calibrated.
- CI (`.github/workflows/ci.yml`) builds and tests on Ubuntu and macOS with clang, including the
  sanitizer builds and the mutation test. It does not download the 3.5 GB NASDAQ file; the ITCH code is
  tested in CI on generated feeds, and the real-data results in `docs/results` are from an Apple M4 laptop.
