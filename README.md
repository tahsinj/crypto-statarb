# Statistical Arbitrage in Cryptocurrencies

The goal: find momentum
and/or reversal strategies in crypto, backtest them unconstrained with
realistic costs (20 bps for market orders, 7 bps for limit orders), combine the
ones that work with proper weighting, and report returns, volatility, Sharpe,
drawdowns and alpha/beta.

The write-up is [`reports/REPORT.pdf`](reports/REPORT.pdf). The notebooks hold
the full research trail, including the ideas that failed.

## How the research was run

- Data comes straight from Binance's public API: daily bars for the top 150
  USDT pairs since 2018, hourly bars for the 60 most liquid, and perp funding
  rates.
- Each day's universe is the 100 most liquid coins as of the previous day, so
  no backtest trades a coin it could not have known about.
- Weights set at the close of day t earn day t+1's return, and every trade is
  charged: 20 bps for the momentum sleeves (market orders), 7 bps for the
  others (limit orders).
- The history is split in time, roughly 70/15/15, the usual train, validation
  and test split. Parameters were chosen on a development window (2020-01 to
  2024-07) and checked on a gate window (2024-08 to 2025-06). The lockbox year
  (2025-07 to 2026-07) was run once, in notebook 06, after every choice had
  been made.
- All 48 configurations tried are in a trial registry, and the deflated Sharpe
  ratio accounts for that many tries.

## Results

The book combines three sleeves: momentum with weekday exposure halved
(Seasonality), a 10-day taker-imbalance follower (Orderflow) and a funding
carry trade (Carry). The main version uses walk-forward mean-variance weights;
an equal-weight version runs alongside.

Over the whole sample, 2020-01 to 2026-07:

| | Ann. return | Ann. vol | Sharpe | Max DD | Beta BTC | Alpha (t) |
|---|---:|---:|---:|---:|---:|---:|
| Book, equal weight | 25.6% | 13.4% | 1.76 | -10.1% | -0.02 | 23.6% (4.4) |
| Book, walk-forward (from 2021-10) | 8.1% | 7.9% | 1.02 | -8.1% | -0.01 | 8.3% (2.3) |
| Seasonality | 13.6% | 9.6% | 1.37 | -11.5% | -0.01 | 14.5% (3.8) |
| Orderflow | 21.5% | 21.3% | 1.02 | -29.6% | +0.02 | 19.6% (2.4) |
| Carry | 36.6% | 31.3% | 1.15 | -35.6% | -0.08 | 36.6% (2.9) |
| Baseline momentum | 18.4% | 15.5% | 1.17 | -19.6% | -0.02 | 20.2% (3.2) |
| Baseline pairs | -8.0% | 20.2% | -0.31 | -55.9% | -0.04 | -5.4% (-0.7) |
| BTC | 39.3% | 60.8% | 0.86 | -76.6% | | |

These mix the windows the strategies were chosen on with the windows they were
tested on. Split by window, the book looks like this:

| | Dev | Gate | Lockbox |
|---|---:|---:|---:|
| Walk-forward Sharpe | +0.40* | +2.52 | +1.46 |
| Equal-weight Sharpe | +1.68 | +2.90 | +1.23 |
| Walk-forward return / vol | 2.9% / 8.1% | 21.6% / 7.9% | 11.4% / 7.6% |
| Walk-forward max drawdown | -8.1% | -3.9% | -4.7% |
| Walk-forward beta to BTC | +0.00 | -0.05 | -0.02 |

\* The walk-forward book needs 756 days of history, so its dev figure only
covers 2021-10 to 2024-07.

On the lockbox the walk-forward book has an alpha of 9.5% a year (Newey-West
t = 1.25) and a block-bootstrap 95% interval for its Sharpe of [-0.42, 3.44].
The deflated Sharpe is 0.49 on the full sample and 0.20 on the lockbox, far
below the usual 0.95. The book made money with almost no market exposure, but
the evidence is not strong enough to call it proven.

Two notes on the alphas. For the momentum-based sleeves most of the alpha is
market timing: trend following flips between long and short, so its average
beta is near zero, and a regression that allows for the timing leaves almost
no alpha (report section 2). And alpha scales with volatility: Carry runs at
about 31% volatility, so its alphas look large; at 10% volatility its lockbox
alpha would be about 8% (report section 4).

**Correction found in an audit after the lockbox.** The Seasonality backtest
scaled each day's return by 0.5 or 1.0 but never charged for the trades that
resize the book twice a week. With those trades charged, the sleeve falls to
dev +1.51 and gate +0.99, below the untilted momentum sleeve on dev, so it
should not have been selected. Notebook 07 re-scores the same frozen book with
the trades charged: lockbox Sharpe 1.49 (walk-forward) and 1.16 (equal
weight), deflated Sharpe 0.21. The conclusion stays the same.

**Forward test.** Notebook 08 runs the frozen book, unchanged, on the 82 days
after the data ends (2026-07-07 to 2026-09-26):

| | Return over the 82 days | Sharpe | Beta BTC |
|---|---:|---:|---:|
| Book, walk-forward | -5.9% | -1.01 | -0.12 |
| Book, equal weight | +5.1% | +0.63 | -0.20 |
| Seasonality | +2.6% | +1.17 | +0.07 |
| Orderflow | -10.8% | -2.56 | -0.12 |
| Carry | +11.3% | +0.90 | -0.55 |
| BTC | +31.8% | +3.25 | |

The walk-forward book lost money. Orderflow, most of its weight, had a bad
run (about one 82-day stretch in twenty since 2020 was worse), and a beta of
-0.12 during BTC's rally accounts for about two thirds of the loss. All of
Carry's gain came from DEXE, which fell 82.5% in a day; its funding went so
negative that Carry's weights, which have no cap per coin, put half the sleeve
into it. An 82-day Sharpe has a standard error of about 2, so this neither
confirms nor overturns the lockbox, but it shows that one coin can take over
the carry sleeve. The book stays as frozen, since a cap chosen now would be
fitted to this window (report section 6).

## What worked and what didn't

- Baseline time-series momentum decayed: Sharpe +1.65 on dev, +0.43 on the
  gate. Pairs trading lost money on dev (-0.40).
- Halving momentum's weekday exposure looked better on both windows (dev
  +1.79, gate +1.24) but lost money on the lockbox (-0.56), and with correct
  costs it would not have passed selection (above).
- Following the 10-day taker-buy imbalance held up best: dev +0.89, gate
  +2.12, lockbox +1.95, then -10.8% over the forward test. Every reversal
  variant built on order flow lost on dev.
- Shorting high-funding perps against low-funding ones: dev +1.01, gate
  +2.19, lockbox +0.88. On dev, 63% of its P&L is funding income. Its forward
  gain came from a single coin.
- Hourly cross-sectional reversal has a real gross edge (gross Sharpe +4.83 at
  a 4-hour lookback), but hourly trading costs 461% a year against a 272%
  gross return. No hourly config passed.
- Equal weight beat walk-forward weights on every window except the lockbox.
- Run in twsq on a fixed list of 20 large coins, the three sleeves are close
  to flat (Sharpe +0.07, -0.15 and +0.22 over 2024-08 to 2026-07). The
  order-flow edge seems to live in the smaller names of the top 100.

## Repository layout

```
quantlib/                 shared library used by every notebook
  fetch.py                Binance fetchers and the local cache
  data.py                 panels and the point-in-time universe
  signals.py              z-scores, calendar masks, weights
  backtest.py             vectorised backtest: one-day weight lag, turnover costs
  pairs.py                pairs selection and trading (reversal baseline)
  strategies.py           sleeve definitions, costs as parameters
  metrics.py              performance stats, significance tests, alpha/beta
  trials.py               trial registry behind the deflated Sharpe
  robustness.py           walk-forward weights and other portfolio helpers
  plotting.py             chart helpers and the plot style
notebooks/
  00_data                 fetch and cache Binance data; build the panels
  01_baselines            baseline momentum and pairs
  02_seasonality          calendar tilts on momentum
  03_orderflow            taker-imbalance signals
  04_carry                funding carry and crowding
  05_fastrev              hourly cross-sectional reversal
  06_portfolio            the combined book and the one lockbox run
  07_posthoc              checks added after the lockbox, whole-sample figures
  08_forward              the frozen book on data after the freeze
alphas/                   the three sleeves as twsq alphas, with backtest CSVs
tests/                    look-ahead, neutrality, cost and fetch tests
reports/
  build_report.py         builds REPORT.pdf and checks its numbers
  REPORT.pdf              the written report
```

`data/` is not in the repository; notebook 00 rebuilds it.

## Reproducing

```bash
# environment (Python 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt

# tests
for f in tests/test_*.py; do .venv/bin/python "$f"; done

# notebooks, in order
for nb in 00_data 01_baselines 02_seasonality 03_orderflow 04_carry 05_fastrev 06_portfolio 07_posthoc 08_forward; do
  .venv/bin/jupytext --to notebook notebooks/$nb.py
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done

# report (stops without writing the PDF if a number disagrees with the notebooks)
.venv/bin/python reports/build_report.py
```

Things to know before rerunning:

- Notebook 00 downloads from Binance's global API, which refuses connections
  from US IP addresses. The data in this project was fetched on 2026-07-06.
  `END` in notebook 00 pins the sample to that date, but a new fetch picks the
  top 150 pairs by volume on the day it runs, so the universe, and the
  numbers, can differ a little.
- Notebook 08 downloads 2026-05-01 to 2026-09-26 for the same coins (the dates
  are pinned) and caches it under `data/raw/forward_*`. It checks the overlap
  with the cached data before using anything new.
- The trial registry (`data/processed/trial_registry.csv`) skips configs it
  already holds, so rerunning a notebook does not add trials. Notebook 06
  expects exactly 48 research rows.
- The twsq alphas run in a separate environment; see `alphas/README.md`.

The notebooks are stored as paired `.py` files (jupytext) for readable diffs;
the `.ipynb` files keep the executed outputs.

## Limitations

The report lists them in full. The main ones: the universe comes from coins
Binance still listed in 2026, so earlier delistings are missing (the hourly
and funding panels, picked by volume at fetch time, lean further toward
survivors); everything is Binance-only; the carry backtest adds funding to
spot returns and ignores the perp basis; Carry's weights have no cap per coin;
Orderflow and Carry assume their limit orders always fill; and a one-year
lockbox and an 82-day forward test are short.
