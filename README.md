# Quant Systems Lab

Pricing, risk, rates, credit, execution and market-microstructure infrastructure in **C, C++ and Python**,
where every number is checked against something independent: QuantLib, textbook worked examples,
closed-form identities, brute-force oracles, simulation, or **a full day of real NASDAQ market data**.

```
            Python 3.10+   research & risk layer
   pricing.py · fixed_income · risk · credit · microstructure · execution
        |  ctypes                              |  ctypes
        v                                      v
   C11  libqsl                          C++20  libqslcpp
   normal dist · BSM + Greeks · IV      limit order book / matching engine
   xoshiro256** · Monte Carlo (CV)      NASDAQ ITCH 5.0 feed handler + book rebuild
   dates · day counts · bond math       lock-free SPSC queue
                                        binomial (CRR, Leisen-Reimer) · Crank-Nicolson · Heston
```

No third-party code is vendored. The only runtime dependencies are NumPy and SciPy, and QuantLib is used
by the tests as a reference only.

## Headline results (Apple M4)

**Real data: one full NASDAQ trading day (30 Dec 2019, 268.7M ITCH messages, 8,906 symbols)**

| What | Result |
|---|---|
| Full order-book rebuild of every NASDAQ stock, one thread | **18.3M msgs/s (55 ns/msg)**; the whole day in 25 s including gunzip |
| Parse + decode every field | 193M msgs/s (5.2 ns/msg) |
| Book integrity over the whole day | **0** unknown orders, overfills or mismatches; **0** orders left on the book at the close |
| Rebuilt closing-cross prices vs official closes | **52 / 52** symbols agree, to the source's precision (cents) |
| C++ parser vs an independent Python decoder (20M real messages) | all 21 field checksums and 17 per-type counts identical |
| VWAP vs TWAP on real minute bars, 100 most traded symbols | VWAP cuts median tracking error 25% (6.5 → 4.8 bp) on ordinary stocks and impact cost by more than half for a 1% order (29 → 12 bp mean) |

Details: [itch_replay_20191230.txt](docs/results/itch_replay_20191230.txt),
[closing crosses](docs/results/itch_closing_cross_vs_official.txt),
[cross-check](docs/results/itch_crosscheck.txt), [execution study](docs/results/execution_study.txt).
Where the data comes from, and other free sources: [docs/DATA.md](docs/DATA.md).

**Engines (`make bench`)**

| What | Result |
|---|---|
| Matching engine, 5M realistic messages, 20k resting orders | **41.5M msgs/s, 24 ns/msg**; p50 42 ns, p99 125 ns, p99.99 250 ns |
| Same flow through a `std::map`/`std::list` book | 3.7M msgs/s, **11x slower**, identical executions |
| SPSC queue, one message per push / batches of 64 | 38M msgs/s / **1.1-1.6 billion msgs/s** |
| Gateway thread → SPSC → matching thread | ~same throughput as single-threaded matching |
| American option tree, 2001 steps: C++ vs NumPy | 0.67 ms vs 29.0 ms (**43x**), same price to 10 dp |
| 1M Black-Scholes prices: C batch vs vectorised NumPy | 43 ns vs 36 ns per option: NumPy is already compiled code, so C only pays off on sequential algorithms |

Full outputs are in [docs/results](docs/results). Timer resolution on Apple Silicon is 41.7 ns, so latency
percentiles are quantised to that step. Thread benchmarks vary roughly ±30% between runs because macOS does
not allow pinning threads to cores.

## What is validated, and against what

| Component | Independent reference | Agreement |
|---|---|---|
| BSM price + delta, gamma, vega, theta, rho, dividend rho | QuantLib `AnalyticEuropeanEngine`, 36 cases | ≤ 1e-10 relative |
| Inverse normal CDF | SciPy `ndtri`, p from 1e-300 to 1 - 1e-16 | ≤ 4e-15 relative |
| Implied volatility | QuantLib, plus price round trip | 1e-8 vs QL; price reproduced to 1e-13 |
| Leisen-Reimer tree, European and American | QuantLib `BinomialVanillaEngine("lr")` | ≤ 1e-9 |
| Cox-Ross-Rubinstein tree | Hull's American put: 4.49 (5 steps), 4.263 / 4.272 / 4.278 / 4.283 | all printed digits |
| Crank-Nicolson PDE (Rannacher, cell-averaged payoff) | BSM: error falls exactly 4x per grid doubling; QuantLib FD American | 2e-4 price, 2e-5 delta |
| Heston (little-trap characteristic function) | QuantLib `AnalyticHestonEngine`, 50 cases incl. 2-week and 10-year expiries; BS limit; Monte Carlo | ≤ 1e-7 relative |
| Geometric Asian (Kemna-Vorst) | QuantLib analytic discrete engine | ≤ 1e-12 relative |
| Monte Carlo | BSM within 4 SE; 95% CIs cover the true price ~95% of the time; control variate cuts Asian SE more than 10x | statistical |
| xoshiro256** RNG | Python transliteration of the published algorithm | bit-identical |
| Bond clean price, accrued, duration, convexity, YTM | QuantLib `BondFunctions`: 5 bonds (ACT/ACT ICMA, 30/360, 30E/360; annual to monthly coupons) × 48 settlement/yield scenarios | ≤ 1e-9 |
| ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT ISDA | Python `datetime`, QuantLib day counters | ≤ 1e-14 |
| Yield-curve bootstrap | Hull Table 4.4 zero rates; exact repricing of every input | 5e-6; 1e-13 |
| Order book | Naive C++ reference book (720k random ops) and a pure-Python book via ctypes (60k ops) | identical trades and depth |
| ITCH 5.0 parser | Every type vs bytes written field by field from the spec; an independent Python decoder on generated feeds and on 20M real messages | every field identical |
| ITCH book rebuild | Naive `std::map` book after every message (incl. invalid events); a full real NASDAQ day | identical depth; 0 integrity errors in 268.7M messages |
| NASDAQ closing cross | Official closes (Yahoo Finance) for the 80 largest crosses | 52/52 available agree (within the source's half-cent rounding); 27 delisted/renamed and 1 reused ticker reported, not hidden |
| Almgren-Chriss trajectory | Direct numerical minimisation of E + λVar (tridiagonal solve) | ≤ 1e-9 |
| Almgren-Chriss E[IS], Var[IS] | Monte Carlo of the model's price dynamics | mean within 4 SE; variance within 1% |
| TWAP / VWAP simulator | Exact identities: perfect forecast hits VWAP, TWAP hits the time average, impact formula | 1e-9 |
| Test strength | 13 injected bugs: 6 in the matching engine, 7 in the ITCH handler (`scripts/mutation_test.py`) | **13/13 caught** |
| SPSC queue | 2-thread stress tests under ThreadSanitizer | no data races |
| Historical VaR / ES | Brute-force order statistic and integral of the quantile function | exact |
| Basel traffic light | 1996 Basel table of cumulative probabilities and plus factors | 5e-5 |
| Kupiec / Christoffersen | Hand-computed LR; empirical test size by simulation | size 5.7% at n = 5000 |
| GARCH(1,1) MLE | Recovers parameters of a simulated process | α ± 0.015, β ± 0.02 |
| Merton model | Hull's worked example (V = 12.40, σV = 21.23%, PD = 12.7%) | printed digits |
| Basel IRB | Risk weight 92.32% at PD 1%, LGD 45%, M 2.5; Vasicek quantile vs one-factor MC | 1e-4; 3% |
| CDS | Credit triangle (exact at r = 0); bootstrap reprices every quote | 1e-5; 1e-12 |
| Interview solutions (Py / C++ / C) | Exhaustive brute force on thousands of random inputs | identical |

Extra checks: everything builds with `-Wall -Wextra -Wpedantic -Wconversion -Werror` (plus
`-Wsign-conversion -Wold-style-cast` for C++), and the native suites pass under AddressSanitizer,
UndefinedBehaviorSanitizer and ThreadSanitizer.

Findings from validation that are worth quoting in interviews:
1. **Christoffersen's independence test over-rejects on short samples.** On 1,000 days of truly
   independent exceptions a 5% test rejects about 9% of the time; it only approaches 5% by around 5,000
   days. The test suite asserts both facts.
2. **Crank-Nicolson only converges at second order once the payoff is cell-averaged.** Without it, the
   strike falling between grid nodes made the error oscillate, and it got *worse* going from a 400 to an
   800 grid. With averaging it falls exactly 4x per grid doubling, now locked in by a test.
3. **The feed handler's speed is a memory-layout problem, not a parsing problem.** Decoding costs 5 ns per
   message, but rebuilding the book started at 85 ns because every lookup missed cache. Indexing orders
   by their (mostly increasing) NASDAQ reference, scanning price levels from the top and prefetching
   took it to 50 ns. Letting the order table fill to ~45% made the same data 3x slower again: references
   arrive in dense runs that form long probe clusters. The steps are in [DESIGN.md](docs/DESIGN.md).
4. **Real data does not look like textbook data.** INTC, one of the most liquid stocks, had no NASDAQ
   execution at all from 12:11:52 to 12:13:30, although its book updated 804 times in the 12:12 minute alone.
   Crossed books while trading appeared in only 10 symbols, 631 of 642 times right after auction fills. And a generic VWAP profile fails on news days (18 of the top 100
   symbols moved more than 10% intraday), which the execution study reports separately.

## Map by role

| If you are interviewing for... | Look at |
|---|---|
| **Quant developer / HFT / low-latency C++** | ITCH feed handler: [`parser.hpp`](cpp/include/qsl/itch/parser.hpp), [`book.hpp`](cpp/include/qsl/itch/book.hpp), [`order_map.hpp`](cpp/include/qsl/itch/order_map.hpp), [`itch_replay.cpp`](cpp/tools/itch_replay.cpp); matching engine: [`order_book.hpp`](cpp/include/qsl/lob/order_book.hpp), [`hierarchical_bitset.hpp`](cpp/include/qsl/lob/hierarchical_bitset.hpp); [`spsc_queue.hpp`](cpp/include/qsl/concurrency/spsc_queue.hpp), [benchmarks](cpp/bench) |
| **Quant researcher / strats (derivatives)** | [`binomial.hpp`](cpp/include/qsl/pricing/binomial.hpp), [`finite_difference.hpp`](cpp/include/qsl/pricing/finite_difference.hpp), [`heston.hpp`](cpp/include/qsl/pricing/heston.hpp), [`bsm.c`](c/src/bsm.c), [`mc.c`](c/src/mc.c) |
| **Quant trader / market maker** | [`market_making.py`](python/qsl/microstructure/market_making.py) (Avellaneda-Stoikov), [order book](python/qsl/microstructure/lob.py), [puzzles](interview/python/puzzles.py) |
| **Execution / algo trading** | [`almgren_chriss.py`](python/qsl/execution/almgren_chriss.py), [`schedules.py`](python/qsl/execution/schedules.py) (TWAP, VWAP, participation cap), [real-data study](scripts/execution_study.py) |
| **Market risk (bank)** | [`var.py`](python/qsl/risk/var.py), [`backtest.py`](python/qsl/risk/backtest.py), [`volatility.py`](python/qsl/risk/volatility.py) |
| **Credit risk / model validation** | [`merton.py`](python/qsl/credit/merton.py), [`irb.py`](python/qsl/credit/irb.py), [`cds.py`](python/qsl/credit/cds.py), and the validation approach in this README |
| **Rates / fixed income** | [`bond.c`](c/src/bond.c), [`dates.c`](c/src/dates.c), [`curves.py`](python/qsl/fixed_income/curves.py) |
| **Coding rounds** | [`interview/`](interview): LeetCode 121, 122, 123, 188, 146, 239, 295, 309, 480, 714, 901, 1801, 2034 and max drawdown, with brute-force tests |

[docs/DESIGN.md](docs/DESIGN.md) explains the design decisions and trade-offs.
[docs/GLOSSARY.md](docs/GLOSSARY.md) maps quant terms (DV01, vanna, ES, RWA...) to the code that computes them.

## Build and test

Requirements: a C11/C++20 compiler (clang ≥ 15 or gcc ≥ 12), GNU make, Python ≥ 3.10 with NumPy and SciPy.

```bash
make                 # libraries, test binaries, benchmarks
```
```bash
python3 -m pip install numpy scipy pytest QuantLib   # QuantLib is optional: its tests skip without it
```
```bash
make test            # C, C++, interview (C++ and C) and Python suites
```
```bash
make sanitize        # ASan + UBSan build of the native tests
```
```bash
make tsan            # ThreadSanitizer build of the C++ tests
```
```bash
make bench           # order book, SPSC and native-vs-NumPy pricing benchmarks
```
```bash
python3 scripts/mutation_test.py   # inject 13 bugs (order book + ITCH), expect 13/13 caught
```

Real NASDAQ data (3.5 GB download, optional; see [docs/DATA.md](docs/DATA.md)):

```bash
scripts/fetch_nasdaq_itch.sh
```
```bash
gzip -dc data/12302019.NASDAQ_ITCH50.gz | build/bin/itch_replay - --crosses data/nasdaq_20191230_crosses.csv
```
```bash
python3 scripts/itch_crosscheck.py         # C++ vs independent Python decoder on 20M real messages
```
```bash
python3 scripts/itch_validate_closes.py    # rebuilt closing crosses vs official closes (needs network)
```
```bash
python3 scripts/execution_study.py         # TWAP vs VWAP on the committed real minute bars
```

Python example (after `make`, run `python3 -m pip install -e .` once so `import qsl` works from anywhere):

```python
from qsl import pricing
pricing.greeks("call", spot=100, strike=105, t=1.0, r=0.05, q=0.02, sigma=0.25)
pricing.heston("put", 100, 90, 0.5, 0.03, 0.0, pricing.HestonParams(v0=0.04, kappa=2, theta=0.04, xi=0.5, rho=-0.7))
pricing.binomial("put", 100, 105, 1.0, 0.05, 0.02, 0.3, steps=801, american=True)
```

## Layout

```
c/            C11 kernels: include/qsl/qsl.h, src/, tests/
cpp/          C++20 engines: include/qsl/{lob,itch,concurrency,pricing}, src/c_api.cpp, tools/itch_replay.cpp,
              tests/, bench/
python/qsl/   Python package: pricing, fixed_income, risk, credit, microstructure (incl. ITCH codec), execution
python/tests/ pytest suite (QuantLib, SciPy and textbook oracles)
interview/    LeetCode-style problems in Python, C++ and C, plus probability puzzles
scripts/      mutation testing, benchmarks, NASDAQ data fetch, real-data cross-checks, execution study
docs/         design notes, glossary, data sources, results, derived real-data sample (docs/data)
```

## License

MIT
