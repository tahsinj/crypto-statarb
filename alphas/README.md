# alphas/: twsq versions of the three sleeves

These are the three sleeves of the final book written as twsq `Alpha` classes,
the format the `twsq` framework runs. The research itself lives in
`quantlib/` and the notebooks; these files show the same strategies running in
the twsq execution framework.

- `seasonal_momentum.py`: vol-targeted time-series momentum with a
  weekday/weekend overlay (half size on weekdays, full size at weekends).
  Follows `quantlib.strategies.seasonal_momentum_sleeve`. Market orders,
  20 bps.
- `orderflow_follow.py`: cross-sectional z-score of the 10-day mean taker-buy
  share minus 0.5. Long the names with the most buying pressure, short the
  names with the least. Follows `quantlib.strategies.orderflow_sleeve`. Limit
  orders, 7 bps.
- `funding_carry.py`: minus the z-score of the 7-day mean perp funding rate.
  Short the names where longs pay the most funding, long the names where they
  pay the least. Follows `quantlib.strategies.carry_sleeve`. Limit orders,
  7 bps.
- `_binance_data.py`: small stdlib + pandas helper that the two Binance-signal
  alphas share. It does not import quantlib.
- `run_twsq.py`: runs all three backtests and writes the CSVs below.

## Backtest outputs

`SeasonalMomentum/`, `OrderflowFollow/` and `FundingCarry/` each hold:

- `backtest/orders.csv`: every order and its fill.
- `backtest/pos_pnl.csv`: daily `port_val`, `pnl` and positions per symbol.
- `stats.csv`: Sharpe, volatility, max drawdown and total P&L over the run.

## Reproducing

twsq is not part of the research venv. Set up a separate environment once
(the package ships as `twsq.zip`):

```bash
unzip twsq.zip && cd twsq
uv venv --python 3.12 .venv
uv pip install -e . pyyaml ccxt
```

Then, from the `alphas/` directory:

```bash
/path/to/twsq/.venv/bin/python run_twsq.py
```

## Differences from the research backtests

**Window.** The twsq runs cover the last 700 days on Binance daily bars (the
backtester's price source), while the research runs from 2020. The Sharpe
ratios will not match the notebooks. The CSVs show that the strategies run
under the twsq framework and its cost model; they are not a replication of
the 2020-onward research numbers.

**No funding income in FundingCarry.** A spot backtest cannot hold perps, so
it misses the funding payments that make up much of the carry sleeve's P&L.
The `FundingCarry` stats understate the sleeve.

**Fixed universe.** The twsq runs trade a fixed list of 20 large coins. The
research uses a point-in-time top-100 universe (trailing ADV, lagged a day)
that changes over time, so the twsq results carry some survivorship bias.
