# Statistical Arbitrage in Cryptocurrencies

A research project: find profitable
momentum and/or reversal strategies in crypto, backtest them unconstrained and
net of realistic costs (20 bps market / 7 bps limit orders), combine them with
proper weighting, and report returns, volatility, Sharpe, drawdown and
alpha/beta.

Every backtest is point-in-time, tuned only on a development window, checked
out of sample and charged transaction costs. Ideas that failed are reported
along with the ones that worked.

## Headline result

A low-beta book of three sleeves: calendar-scaled momentum (Seasonality),
taker-imbalance following (Orderflow) and funding carry (Carry).

| Window | WF Sharpe | EW Sharpe | BTC Beta |
|---|---:|---:|---:|
| Development (2020-01 to 2024-07) | +0.40* | +1.68 | - |
| Gate (2024-08 to 2025-06) | +2.52 | +2.90 | - |
| **Lockbox (2025-07 to 2026-07)** | **+1.458** | **+1.233** | **-0.020** |

*The walk-forward book needs 756 days of training, so its dev figure only
starts in 2021-10.*

Lockbox annualised alpha 9.48% (HAC t = 1.25). Deflated Sharpe is 0.490 on
the full sample and 0.205 on the lockbox against 48 enumerated trials:
positive, but not statistically decisive.

## What the research found

- Baseline momentum decayed: dev Sharpe +1.65, gate +0.43.
- Three new sleeves passed the dev and gate screens: Seasonality
  (weekday/weekend scaling of momentum), Orderflow (10-day taker imbalance,
  cross-sectional) and Carry (short high-funding perps). All three have gate
  Sharpes above 0.8.
- Hourly cross-sectional reversal fails on costs. The 4-hour lookback has a
  gross Sharpe of +4.83, but hourly rebalancing costs 461% a year against a
  272% gross return. That ratio (0.59) is far below the 3x threshold set in
  advance.
- Equal weight beats mean-variance on most windows. With three nearly
  uncorrelated sleeves there is little for mean-variance to add, and it only
  comes out ahead on the lockbox.

## Repository layout

```
quantlib/                 shared library used by every notebook
  fetch.py                Binance fetchers and the local cache
  data.py                 panels and the point-in-time universe
  signals.py              ranks, z-scores, calendar masks, weights
  backtest.py             vectorised backtest with a one-day weight lag and costs
  pairs.py                pairs selection and trading (reversal baseline)
  strategies.py           sleeve definitions, costs as parameters
  metrics.py              performance stats, significance tests, alpha/beta
  trials.py               trial registry used for the deflated Sharpe
  robustness.py           walk-forward weights, capacity and regime tables
  plotting.py             chart helpers and plot style
notebooks/
  00_data                 fetch and cache Binance data; build daily/hourly/funding panels
  01_baselines            baseline momentum and pairs
  02_seasonality          calendar tilts on momentum; survivor: weekday/weekend scaling
  03_orderflow            taker-imbalance signals; survivor: 10-day follow
  04_carry                funding carry; survivor: 7-day smoothed, short high funding
  05_fastrev              hourly cross-sectional reversal; no survivor (costs)
  06_portfolio            walk-forward combination and the one-time lockbox evaluation
alphas/                   twsq versions (SeasonalMomentum, OrderflowFollow, FundingCarry)
tests/                    tests for look-ahead, neutrality, costs and the fetch layer
reports/
  build_report.py         rebuilds REPORT.pdf and checks its key numbers
  REPORT.pdf              the written report
data/raw/                 Binance cache (gitignored, rebuilt on demand)
data/processed/           panels written by notebooks 00-06
```

## Reproducing

```bash
# 1. environment (Python 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt

# 2. tests (look-ahead, dollar neutrality, costs)
for f in tests/test_*.py; do .venv/bin/python "$f"; done

# 3. notebooks in order (00 fetches the data and writes data/processed/;
#    01-06 read those panels)
for nb in 00_data 01_baselines 02_seasonality 03_orderflow 04_carry 05_fastrev 06_portfolio; do
  .venv/bin/jupytext --to notebook notebooks/$nb.py
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done

# 4. rebuild the PDF (reads data/processed/ and checks the key numbers
#    before writing)
.venv/bin/python reports/build_report.py
```

The notebooks are stored as paired `.py` files (jupytext, light format) for
readable diffs; the `.ipynb` files keep the executed outputs.

## Data

Raw data comes from Binance's public APIs (no API key) through notebook 00 and
is cached under `data/raw/` (gitignored). `quantlib.fetch.refresh` refetches
when the cache is more than a day old.

- Daily klines: top-150 USDT pairs, 2018 to present.
- Hourly klines: top-60 pairs by trailing ADV, 2020 to present.
- Perpetual funding rates from Binance fapi, 2019-09 onward.

Validation windows:

| Window | Dates | Purpose |
|---|---|---|
| Development | 2020-01-01 to 2024-07-31 | parameter search, model selection |
| Gate | 2024-08-01 to 2025-06-30 | held out during development |
| Lockbox | 2025-07-01 to end | evaluated once, in notebook 06 |

Survivorship bias is a known limitation. The universe is rebuilt from the
pairs Binance lists today, so coins delisted before the fetch date are
missing. The point-in-time liquidity screen and the gate split reduce the
effect but cannot remove it.

## Execution costs

These costs map directly onto the engine: turnover times bps, with
20 bps for market orders (7 bps commission plus 13 bps slippage) and 7 bps for
limit orders (commission only). The seasonality sleeve takes liquidity on
trend entries, so it pays the market-order rate; orderflow and carry
rebalance passively and pay the limit rate. The three-sleeve book has not been
stress-tested at higher costs yet (see the report's limitations).
