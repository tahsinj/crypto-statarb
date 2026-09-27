# Statistical Arbitrage in Cryptocurrencies

The goal: find momentum
and/or reversal strategies in crypto, backtest them unconstrained with
realistic costs (20 bps for market orders, 7 bps for limit orders), combine the
ones that work with proper weighting, and report returns, volatility, Sharpe,
drawdowns and alpha/beta.

The write-up is [`reports/REPORT.pdf`](reports/REPORT.pdf). The notebooks hold
the full research trail, including the ideas that failed.

## What this project shows

- A research protocol that can fail: development, gate and lockbox windows in
  time order, a registry of all 53 configurations tried, deflated Sharpe
  ratios, a lockbox opened once, and a second version registered in the
  repository before it was tested.
- A survivorship check on its own result. The original coin list, the 150
  most-traded pairs of July 2026, only held survivors: 46% of the historical
  top-100 universe was missing from it. Rebuilt from Binance's public archive
  with all 585 coins, delisted ones included, the book's lockbox Sharpe falls
  from 1.46 to 0.36.
- Execution measured, not assumed. 99% of limit orders fill, but the misses
  are the days the price runs away, which costs Orderflow 43% of its price
  P&L on dev. Under a square-root impact model its edge is mostly gone by $1M.
- An edge that holds up on the full universe on dev and gate: funding carry
  weighted by rank on perp prices, Sharpe 1.76 and 2.41. The second version of
  the book is built on it and has a monthly forward test.
- A tested library, a backtester that cannot look ahead, and a report whose
  every number is checked against the notebooks before it is written.

## Results at a glance

| Sharpe ratio | Dev | Gate | Lockbox | Forward |
|---|---:|---:|---:|---:|
| Frozen book, walk-forward, research coin list | +0.40 | +2.52 | +1.46 | -1.01 |
| Frozen book, walk-forward, every pair | -0.25 | +1.87 | +0.36 | -0.91 |
| Frozen book, equal weight, research coin list | +1.68 | +2.90 | +1.23 | +0.63 |
| Frozen book, equal weight, every pair | +0.78 | -1.46 | -0.62 | +1.26 |
| v2 book, every pair | +1.39 | +1.12 | not run | from 2026-09-28 |
| v2 Carry sleeve, every pair | +1.76 | +2.41 | not run | from 2026-09-28 |

Dev is 2020-01 to 2024-07, the gate 2024-08 to 2025-06, the lockbox 2025-07 to
2026-07 and the forward window 2026-07-07 to 2026-09-26. "Every pair" is the
daily top 100 chosen from all Binance pairs, delisted ones included. v2 is
never run on the lockbox or the forward window, which had both been seen
before it was written.

## How the research was run

- Data comes from Binance: daily bars for the top 150 USDT pairs since 2018,
  hourly bars, and perp funding rates. Notebook 09 rebuilds the daily and
  hourly data from Binance's public archive with every pair that ever traded,
  delisted ones included.
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
- Every configuration tried is in a trial registry, 53 in all, and the
  deflated Sharpe ratio accounts for them.

## The book on the original coin list

The book combines three sleeves: momentum with weekday exposure halved
(Seasonality), a 10-day taker-imbalance follower (Orderflow) and a funding
carry trade (Carry), with walk-forward mean-variance weights and an
equal-weight version alongside. As run on the research coin list, over
2020-01 to 2026-07:

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

On the lockbox the walk-forward book had a Sharpe of 1.46, an alpha of 9.5% a
year (Newey-West t = 1.25) and a beta to BTC of -0.02. Its deflated Sharpe was
0.20, far below 0.95, so the result was never statistically proven, and the
checks below show that most of it came from the coin list. Report section 4
splits every number by window and explains why the alphas look large:
momentum's is mostly market timing, and Carry runs at about 31% volatility.

## Making it realistic

Notebooks 09 and 11 rebuild the data from Binance's public archive and rerun
the same frozen book one fix at a time (report section 5):

| Lockbox Sharpe | Walk-forward | Equal weight |
|---|---:|---:|
| As run (research coin list) | +1.46 | +1.23 |
| Same list, archive data | +1.45 | +1.23 |
| Pegged assets and tokenized stocks out | +1.26 | +1.20 |
| Every pair, delisted coins included | +0.36 | -0.62 |
| and Carry on perp prices | +0.02 | -0.46 |
| and limit orders that have to fill | +0.14 | -0.69 |

The research coin list, picked by volume in July 2026, left out 46% of the
historical universe (EOS, MATIC, XMR, VET, SAND and many more), and the coins
that did the damage once they are back (OM, VIDT, BNX and others) were mostly
missing from it. Orderflow's research edge sat in its smaller coins, which
were survivors. Requiring limit orders to fill costs it most of what is left,
and under a square-root impact model its edge is mostly gone by $1M.
Notebook 05's hourly reversal, rerun on every pair, has a bigger gross
edge, about equal to its trading cost, but still loses money after costs in
every configuration.

Adding the 82-day forward test (notebook 08) to the lockbox year, the frozen
book's whole out-of-sample record is a Sharpe of 0.37 over 453 days.

## v2 and the forward log

Notebook 10 registers a second version in the repository before testing it:
Orderflow plus a Carry sleeve weighted by rank and measured on perp prices, on
every pair, with equal weights and no Seasonality. On dev and gate it has a
Sharpe of 1.39 and 1.12, nearly all from Carry (1.76 and 2.41); its Orderflow
sleeve fails the gate (-0.78). Its test is every day from 2026-09-28, and
notebook 12 adds each month next to the frozen book.

## What worked and what didn't

- Funding carry weighted by rank and measured on perp prices holds up across
  every pair on dev and gate (Sharpe 1.76 and 2.41). With z-score weights it
  piles into crashing coins and loses (report section 6.1 has one case).
- Following the 10-day taker-buy imbalance looked best on the research coin
  list (lockbox +1.95), but on every pair it loses on the gate and the
  lockbox.
- Baseline time-series momentum decayed (dev +1.65, gate +0.43). Halving its
  weekday exposure looked better until its resizing trades were charged
  (report appendix A).
- Pairs trading lost money on dev (-0.40), and every reversal variant built on
  order flow lost on dev.
- Hourly reversal has a real gross edge (gross Sharpe +4.83 at a 4-hour
  lookback) but costs 461% a year against a 272% gross return.
  On every pair the gross edge grows to about the size of the cost, and every
  configuration still loses after costs.
- Equal weight beat walk-forward weights on every window except the lockbox.
- Run in twsq on 20 large coins, the frozen book's three sleeves are close to flat
  (Sharpe +0.07, -0.15 and +0.22).

## Repository layout

```
quantlib/                 shared library used by every notebook
  fetch.py                Binance fetchers, the public data archive, the local cache
  data.py                 panels and the point-in-time universe
  signals.py              z-scores, calendar masks, weights
  backtest.py             vectorised backtest: one-day weight lag, turnover costs,
                          and a fill model for limit orders
  pairs.py                pairs selection and trading (reversal baseline)
  strategies.py           sleeve definitions (frozen book and v2), costs as parameters
  metrics.py              performance stats, significance tests, alpha/beta
  trials.py               trial registry behind the deflated Sharpe
  robustness.py           walk-forward weights, market impact, other helpers
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
  09_archive_data         every Binance pair from the public archive, checked
                          against the research data
  10_v2                   v2, registered before testing; dev and gate only
  11_realism              the frozen book re-measured one fix at a time
  12_forward_log          the monthly out-of-sample record, frozen book and v2
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
for nb in 00_data 01_baselines 02_seasonality 03_orderflow 04_carry 05_fastrev 06_portfolio 07_posthoc \
          08_forward 09_archive_data 10_v2 11_realism 12_forward_log; do
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
- Notebook 09 downloads about 100,000 small files from data.binance.vision
  the first time (around an hour), cached under `data/raw/archive/`.
- Notebook 12 is the forward log. Once a month, set `END` to the last
  complete UTC day, run it and commit it, so each month's numbers are on
  record before the next month's data exists.
- The trial registry (`data/processed/trial_registry.csv`) skips configs it
  already holds, so rerunning a notebook does not add trials. It has 48
  research rows, the two combination rules and three v2 rows; notebook 06
  expects exactly 48 research rows.
- The twsq alphas run in a separate environment; see `alphas/README.md`.

The notebooks are stored as paired `.py` files (jupytext) for readable diffs;
the `.ipynb` files keep the executed outputs.

## Limitations

Binance is the only exchange, so taker flow has no second source; the samples
are short (a one-year lockbox, an 82-day forward test, and v2 not yet tested);
the fill and impact models are approximations; and the tokenized-stock list is
kept by hand. Report section 9 has them.
