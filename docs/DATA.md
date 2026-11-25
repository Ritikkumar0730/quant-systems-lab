# Real market data: what this project uses, and where to get more

## Used here

**NASDAQ TotalView-ITCH 5.0, full day, 30 Dec 2019.** NASDAQ publishes a few complete days of its
market-by-order feed as free samples for developers at `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/`. The
file used here is `12302019.NASDAQ_ITCH50.gz`: 3.5 GB compressed, 8.25 GB raw and 268.7 million messages.
It contains every order added, executed, cancelled and replaced on NASDAQ that day, across 8,906 symbols.

```bash
scripts/fetch_nasdaq_itch.sh                                   # downloads into data/ (git-ignored)
```
```bash
gzip -dc data/12302019.NASDAQ_ITCH50.gz | build/bin/itch_replay -
```

What is derived from it and committed:

| File | Contents |
|---|---|
| `docs/results/itch_replay_20191230.txt` | message counts, book integrity, throughput for the full day |
| `docs/results/itch_crosscheck.txt` | C++ parser vs independent Python decoder on 20M real messages |
| `docs/results/itch_closing_cross_vs_official.txt` | rebuilt closing-cross prices vs official closes |
| `docs/data/nasdaq_itch_20191230_minute_bars_top100.csv` | regular-hours minute volume, notional and last price for the 100 most traded symbols (input to the execution study) |
| `docs/results/execution_study.txt` | TWAP vs VWAP on those minute bars |

The raw feed is not committed. Only small aggregates derived from it are.

**Official closing prices** for the closing-cross check come from Yahoo Finance's public chart endpoint,
one small request per symbol (`scripts/itch_validate_closes.py`).

## Other free sources worth knowing

| Source | What you get | Good for |
|---|---|---|
| NASDAQ ITCH samples (above) | Full-depth, every-order feed for a few days | Feed handlers, book rebuilding, microstructure, execution research |
| [Binance public data](https://data.binance.vision) | Daily files of every trade, plus order-book snapshots for crypto pairs | Tick-level work on many consecutive days, for free |
| [LOBSTER](https://lobsterdata.com) | Reconstructed NASDAQ order books. The free samples cover a few stocks for one day | Academic LOB research (the full archive is paid or through universities) |
| [NSE India](https://www.nseindia.com) | Daily bhavcopy (end-of-day prices for all stocks and F&O contracts) and option chains | Indian equities and derivatives, end of day |
| [US Treasury](https://home.treasury.gov/resource-center/data-chart-center/interest-rates) | Daily par yield curve | Real inputs for the curve bootstrap and NSS fit |
| [FRED](https://fred.stlouisfed.org) | Rates, spreads, macro series | Rates and credit models, factor research |
| [Kenneth French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Factor returns (market, size, value, momentum...) | Alpha research and factor models |
| Databento, Polygon, IEX Cloud | Paid tick and quote data (some offer free credits) | Multi-day tick studies once free samples run out |

Through a university, WRDS can provide TAQ (every US trade and quote) and OptionMetrics (option prices and
implied volatilities), the standard academic datasets. Check whether your institution subscribes.

Always read each provider's terms before redistributing data. This repository commits only small derived
aggregates, never raw feeds.
