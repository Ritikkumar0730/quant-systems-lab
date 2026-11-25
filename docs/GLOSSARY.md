# Glossary: quant terms and where they live in the code

| Term | Meaning | Code |
|---|---|---|
| Delta, gamma | dV/dS, d²V/dS² | `qsl_bsm_greeks` in [c/src/bsm.c](../c/src/bsm.c); grid Greeks in [finite_difference.hpp](../cpp/include/qsl/pricing/finite_difference.hpp) |
| Vega, volga (vomma), vanna | dV/dσ, d²V/dσ², d²V/dS dσ | [c/src/bsm.c](../c/src/bsm.c) |
| Theta, rho, dividend rho | -dV/dT, dV/dr, dV/dq | [c/src/bsm.c](../c/src/bsm.c) |
| Implied volatility | σ that reproduces a market price | `qsl_bsm_implied_vol` |
| Put-call parity, no-arbitrage bounds | C - P = S e^{-qT} - K e^{-rT}; intrinsic ≤ price < upper bound | C tests; implied-vol bounds check |
| Early exercise premium | American minus European value | [binomial.hpp](../cpp/include/qsl/pricing/binomial.hpp), Brennan-Schwartz solve |
| Stochastic volatility, vol of vol, skew | Heston v_t; ξ; ρ < 0 | [heston.hpp](../cpp/include/qsl/pricing/heston.hpp) |
| Variance reduction (antithetic, control variate) | Lower Monte Carlo error at equal cost | [c/src/mc.c](../c/src/mc.c) |
| Clean / dirty price, accrued interest | Quoted price vs price paid | `qsl_bond_price` in [c/src/bond.c](../c/src/bond.c) |
| YTM, Macaulay / modified duration, convexity | Yield and its first two price sensitivities | [c/src/bond.c](../c/src/bond.c) |
| DV01 / PV01 | Price change for a 1 bp yield move | `dv01` field, [bonds.py](../python/qsl/fixed_income/bonds.py) |
| Day count (ACT/360, 30/360, ACT/ACT) | Accrual year-fraction conventions | [c/src/dates.c](../c/src/dates.c) |
| Bootstrapping, zero / forward rates | Building a discount curve from market quotes | [curves.py](../python/qsl/fixed_income/curves.py) |
| Nelson-Siegel-Svensson | Parametric yield curve (used for the Indian G-Sec zero curve by CCIL) | `fit_nss` |
| VaR, Expected Shortfall (CVaR) | Loss quantile; mean loss beyond it | [var.py](../python/qsl/risk/var.py) |
| Component / marginal VaR | Euler risk allocation | `portfolio_var` |
| Full revaluation vs delta-gamma | Repricing scenarios vs Taylor approximation | `options_var` |
| Backtesting, Kupiec POF, Christoffersen | Checking VaR exception frequency and clustering | [backtest.py](../python/qsl/risk/backtest.py) |
| Basel traffic light, plus factor | Regulatory capital multiplier from exceptions | `basel_traffic_light` |
| EWMA (RiskMetrics), GARCH(1,1) | Conditional volatility models | [volatility.py](../python/qsl/risk/volatility.py) |
| Parkinson / Garman-Klass / Rogers-Satchell | Range-based realised volatility | [volatility.py](../python/qsl/risk/volatility.py) |
| PD, LGD, EAD, EL | Probability of default, loss given default, exposure at default, expected loss | [irb.py](../python/qsl/credit/irb.py) |
| Distance to default | Merton d2 | [merton.py](../python/qsl/credit/merton.py) |
| Asset correlation, ASRF, Vasicek | One-factor credit portfolio model | [irb.py](../python/qsl/credit/irb.py) |
| RWA, capital requirement K | Basel IRB risk-weighted assets | `risk_weighted_assets` |
| Hazard rate, survival probability, CDS spread, RPV01 | Reduced-form credit and CDS pricing | [cds.py](../python/qsl/credit/cds.py) |
| Credit triangle | spread ≈ λ(1 - R) | CDS tests |
| Limit order book, price-time priority | FIFO matching at each price level | [order_book.hpp](../cpp/include/qsl/lob/order_book.hpp) |
| Maker / taker, aggressor | Liquidity provider / remover | `Trade` in [types.hpp](../cpp/include/qsl/lob/types.hpp) |
| GTC, IOC, FOK, post-only | Time-in-force instructions | `TimeInForce` |
| Market-by-price depth | Aggregated quantity per level | `OrderBook::depth` |
| Tick-to-trade pipeline, feed handler | Gateway thread → queue → matching thread | [bench_spsc.cpp](../cpp/bench/bench_spsc.cpp) |
| Market data feed, ITCH 5.0 | NASDAQ's binary market-by-order protocol: every add, execute, cancel, delete and replace | [parser.hpp](../cpp/include/qsl/itch/parser.hpp), [itch.py](../python/qsl/microstructure/itch.py) |
| Market-by-order (L3) vs market-by-price (L2) | Individual orders vs aggregated size per price | [book.hpp](../cpp/include/qsl/itch/book.hpp) |
| Order reference number, stock locate | ITCH's per-day order id and per-day symbol index | [messages.hpp](../cpp/include/qsl/itch/messages.hpp) |
| Book building / reconstruction | Rebuilding every stock's book from the feed | `BookBuilder` in [book.hpp](../cpp/include/qsl/itch/book.hpp) |
| Opening / closing cross, NOII | NASDAQ's auctions and their imbalance messages; the closing cross sets the official close | [itch_closing_cross_vs_official.txt](results/itch_closing_cross_vs_official.txt) |
| Locked / crossed book | Best bid equal to or above best ask | `CrossedCounts` in [book.hpp](../cpp/include/qsl/itch/book.hpp) |
| Software prefetching | Loading memory a few messages ahead to hide DRAM latency | `prefetch` in [book.hpp](../cpp/include/qsl/itch/book.hpp) |
| False sharing, cache line | Cross-core contention on one line | [spsc_queue.hpp](../cpp/include/qsl/concurrency/spsc_queue.hpp) |
| Latency percentiles (p50 / p99 / p99.9) | Tail latency | [bench_order_book.cpp](../cpp/bench/bench_order_book.cpp) |
| Reservation price, inventory risk | Avellaneda-Stoikov market making | [market_making.py](../python/qsl/microstructure/market_making.py) |
| Implementation shortfall, arrival price | Execution cost vs the price when the order arrived | [almgren_chriss.py](../python/qsl/execution/almgren_chriss.py), `simulate` |
| Almgren-Chriss, efficient frontier, urgency κ | Optimal liquidation trading expected cost against variance | [almgren_chriss.py](../python/qsl/execution/almgren_chriss.py) |
| Permanent vs temporary impact | Lasting vs transient price move caused by trading | `ACParams.gamma`, `ACParams.eta` |
| TWAP, VWAP, volume profile | Equal-time and volume-weighted schedules; the U-shaped intraday volume curve | [schedules.py](../python/qsl/execution/schedules.py) |
| Participation rate, POV, square-root law | Share of market volume traded; impact ∝ sqrt(participation) | `cap_participation`, `simulate` |
| VWAP slippage / tracking error | Fill price vs market VWAP | [execution_study.txt](results/execution_study.txt) |
| Kelly criterion | Growth-optimal bet size | [puzzles.py](../interview/python/puzzles.py) |
| Max drawdown | Largest peak-to-trough loss | [stock_trading.py](../interview/python/stock_trading.py) |
