"""Run the three twsq alphas and save their backtest CSVs.

twsq is not part of the research environment. Install it once in its own venv
(the package ships as twsq.zip):

    unzip twsq.zip && cd twsq
    uv venv --python 3.12 .venv
    uv pip install -e . pyyaml ccxt

Then, from this alphas/ directory:

    /path/to/twsq/.venv/bin/python run_twsq.py

twsq's backtester prices and fills on Binance daily bars, downloaded through
ccxt with no API key (live trading would go through Kraken). All three runs
cover the same fixed 700 days, 2024-08-06 to 2026-07-07, which ends where the
research data ends. The signal data (taker volume, funding) also comes from
Binance's public API, fetched once per alpha in prepare().
"""
import os
import sys

# twsq's progress bar calls os.get_terminal_size(), which needs a TTY. Stub it
# so the script runs non-interactively; this only affects progress rendering.
os.get_terminal_size = lambda *a, **k: os.terminal_size((120, 40))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from seasonal_momentum import SeasonalMomentum, symbols as SM_SYMBOLS
from orderflow_follow import OrderflowFollow, symbols as OF_SYMBOLS
from funding_carry import FundingCarry, symbols as FC_SYMBOLS

START, END = "20240806", "20260707"
SIGNAL_END = "2026-07-07"   # last day of taker/funding history to download
CAPITAL = 1_000_000         # the alphas size positions off this fixed amount
TRADING_DAYS = 365


def _stats(pos_pnl: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Summary in return terms on the alpha's capital (P&L is not reinvested)."""
    r = (pos_pnl["pnl"] / CAPITAL).dropna()
    equity = 1.0 + r.cumsum()
    sd = r.std()
    filled = orders[orders["status"] == "closed"]
    return pd.DataFrame({
        "start": [pos_pnl["Date"].iloc[0]],
        "end": [pos_pnl["Date"].iloc[-1]],
        "n_days": [len(r)],
        "ann_return": [r.mean() * TRADING_DAYS],
        "ann_vol": [sd * np.sqrt(TRADING_DAYS)],
        "sharpe": [r.mean() / sd * np.sqrt(TRADING_DAYS) if sd else np.nan],
        "max_drawdown": [(equity / equity.cummax() - 1.0).min()],
        "total_return": [r.sum()],
        "avg_daily_turnover": [filled["ntn_filled"].sum() / CAPITAL / len(r)],
        "fees": [filled["fee"].sum() / CAPITAL],
        "orders_filled": [len(filled)],
        "orders_cancelled": [int((orders["status"] == "canceled").sum())],
    })


def run(alpha_cls, out_dir, **kwargs):
    print(f"\n=== {alpha_cls.__name__}: {START} -> {END} ===")
    res = alpha_cls.run_backtest(start_ts=START, end_ts=END, freq="1d", **kwargs)

    dest = os.path.join(HERE, out_dir, "backtest")
    os.makedirs(dest, exist_ok=True)
    res.save_pos_pnl(dest)
    res.save_orders(dest)
    pos_pnl = pd.read_csv(os.path.join(dest, "pos_pnl.csv"))
    orders = pd.read_csv(os.path.join(dest, "orders.csv"))
    stats = _stats(pos_pnl, orders)
    stats.to_csv(os.path.join(HERE, out_dir, "stats.csv"), index=False)
    print(stats.T.to_string(header=False))
    print(f"saved -> {out_dir}/backtest/{{pos_pnl,orders}}.csv and {out_dir}/stats.csv")
    return res


if __name__ == "__main__":
    # SeasonalMomentum: market orders, 20 bps all-in.
    run(
        SeasonalMomentum, "SeasonalMomentum",
        taker_fee=7e-4, slip=13e-4,
        symbols=SM_SYMBOLS,
        lookback=30, halflife=30, target_vol=0.15,
        weekend_scale=1.0, weekday_scale=0.5,
    )

    # OrderflowFollow: limit orders, 7 bps; Binance taker data downloaded in prepare().
    run(
        OrderflowFollow, "OrderflowFollow",
        maker_fee=7e-4, slip=0.0,
        symbols=OF_SYMBOLS, smooth=10,
        hist_days=730, end=SIGNAL_END,
    )

    # FundingCarry: limit orders, 7 bps; Binance funding downloaded in prepare().
    # Spot-only backtest: no funding income, so this understates the sleeve.
    run(
        FundingCarry, "FundingCarry",
        maker_fee=7e-4, slip=0.0,
        symbols=FC_SYMBOLS, smooth=7,
        hist_days=730, end=SIGNAL_END,
    )
