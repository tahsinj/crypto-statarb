"""Run the three twsq alphas and save their backtest CSVs.

twsq is not part of the research environment. Install it once in its own venv
(the package ships as twsq.zip):

    unzip twsq.zip && cd twsq
    uv venv --python 3.12 .venv
    uv pip install -e . pyyaml ccxt

Then, from this alphas/ directory:

    /path/to/twsq/.venv/bin/python run_twsq.py

twsq's backtester prices and fills on Binance daily bars, downloaded through
ccxt with no API key. Each run covers the last 700 days. The signal data
(taker volume, funding) also comes from Binance's public API, fetched once per
alpha in prepare() rather than on every rebalance.
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

TRADING_DAYS = 365


def _stats(pnl: pd.Series) -> pd.DataFrame:
    """Minimal performance summary from a daily P&L series (no quantlib dep)."""
    pnl = pnl.dropna()
    equity = pnl.cumsum()
    dd = equity - equity.cummax()
    sd = pnl.std()
    sharpe = (pnl.mean() / sd * np.sqrt(TRADING_DAYS)) if sd else np.nan
    return pd.DataFrame(
        {
            "sharpe": [sharpe],
            "avg_pnl": [pnl.mean()],
            "volatility": [sd * np.sqrt(TRADING_DAYS)],
            "max_drawdown": [dd.min()],
            "total_pnl": [pnl.sum()],
            "n_days": [len(pnl)],
        }
    )


def run(alpha_cls, out_dir, start_days, **kwargs):
    end = pd.Timestamp.now("UTC").normalize().tz_localize(None)
    start = end - pd.Timedelta(days=start_days)
    print(f"\n=== {alpha_cls.__name__}: {start.date()} -> {end.date()} ===")
    res = alpha_cls.run_backtest(
        start_ts=start.strftime("%Y%m%d"),
        end_ts=end.strftime("%Y%m%d"),
        freq="1d",
        **kwargs,
    )

    dest = os.path.join(HERE, out_dir, "backtest")
    os.makedirs(dest, exist_ok=True)
    pp = res.save_pos_pnl(dest)
    res.save_orders(dest)
    if pp is not None and "pnl" in pp:
        stats = _stats(pp["pnl"])
        stats.to_csv(os.path.join(HERE, out_dir, "stats.csv"), index=False)
        print(stats.to_string(index=False))
    print(f"saved -> {out_dir}/backtest/{{pos_pnl,orders}}.csv")
    return res


if __name__ == "__main__":
    # SeasonalMomentum: market orders, 20 bps all-in.
    run(
        SeasonalMomentum, "SeasonalMomentum", start_days=700,
        taker_fee=7e-4, slip=13e-4,
        symbols=SM_SYMBOLS,
        lookback=30, vol_window=30, target_vol=0.15,
        weekend_scale=1.0, weekday_scale=0.5,
    )

    # OrderflowFollow: limit orders, 7 bps; Binance taker data pre-fetched in prepare().
    run(
        OrderflowFollow, "OrderflowFollow", start_days=700,
        maker_fee=7e-4, slip=0.0,
        symbols=OF_SYMBOLS, smooth=10,
        hist_days=730,  # days of taker history to pre-fetch
    )

    # FundingCarry: limit orders, 7 bps; Binance funding pre-fetched in prepare().
    # Spot-only backtest: no funding income, so this understates the sleeve.
    run(
        FundingCarry, "FundingCarry", start_days=700,
        maker_fee=7e-4, slip=0.0,
        symbols=FC_SYMBOLS, smooth=7,
        hist_days=730,  # days of funding history to pre-fetch
    )
