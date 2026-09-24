# alphas/: the three sleeves in twsq

The book's three sleeves written as `Alpha` classes for twsq, an
execution and backtesting framework. The research lives in `quantlib/` and the notebooks; these
files run the same signals through twsq's order and fill simulation.

- `seasonal_momentum.py`: time-series momentum (sign of the 30-day return,
  skipping the latest day, equal size per coin), scaled to 15% volatility from
  an EWMA of the book's own returns, at half size on weekdays and full size at
  weekends. Market orders, 20 bps.
- `orderflow_follow.py`: cross-sectional z-score of the 10-day mean taker-buy
  share minus 0.5; long the names with the most aggressive buying, short the
  names with the least. Limit orders, 7 bps.
- `funding_carry.py`: minus the z-score of the 7-day mean perp funding rate;
  short the names where longs pay the most funding, long the ones where they
  pay the least. Limit orders, 7 bps.
- `_binance_data.py`: downloads taker volume and funding history from Binance
  (standard library and pandas only, no quantlib import).
- `run_twsq.py`: runs the three backtests and writes the CSVs below.

The orderflow and carry signals only use days that have closed: a rebalance at
the start of day D sees data up to D-1. Limit orders are placed at the last
close, and any still unfilled the next day are cancelled before new ones go in.

## Results

All three runs cover the same 700 days, 2024-08-06 to 2026-07-07, on $1M of
capital each:

| Alpha | Ann. return | Ann. vol | Sharpe | Max DD | Turnover/day |
|---|---:|---:|---:|---:|---:|
| SeasonalMomentum | 0.7% | 9.9% | +0.07 | -10.9% | 8% |
| OrderflowFollow | -2.3% | 15.5% | -0.15 | -27.6% | 29% |
| FundingCarry | 3.8% | 17.2% | +0.22 | -20.6% | 21% |

Each alpha's folder holds `backtest/orders.csv` (every order and fill),
`backtest/pos_pnl.csv` (daily portfolio value, P&L and positions) and
`stats.csv` (the summary above, plus fees and order counts).

## How these differ from the research backtests

- Universe. The alphas trade a fixed list of 20 large coins. The research uses
  a point-in-time top 100 by volume. On the 20 large coins none of the three
  sleeves does much, which suggests the order-flow edge in the research comes
  from smaller names.
- Costs. twsq charges every trade, including the weekday/weekend resizing that
  the frozen Seasonality backtest left uncharged (notebook 07 covers this).
- Funding. twsq holds spot, so FundingCarry earns the price leg only. In the
  research, funding income is 63% of the carry sleeve's P&L on dev and 39% on
  the gate.
- Window. Two years, against 2020 onward in the research, so the Sharpe ratios
  are not directly comparable.

twsq's backtester prices and fills on Binance daily bars (its live trading
goes through Kraken), so prices and signals come from the same exchange.

## Reproducing

twsq is not part of the research environment. Install it once in its own
venv (the package ships as `twsq.zip`):

```bash
unzip twsq.zip && cd twsq
uv venv --python 3.12 .venv
uv pip install -e . pyyaml ccxt
```

Then, from the `alphas/` directory:

```bash
/path/to/twsq/.venv/bin/python run_twsq.py
```

The window is fixed in `run_twsq.py`, so a rerun reproduces these CSVs as long
as Binance serves the same history.
