"""Sleeve definitions shared by the notebooks.

momentum_sleeve and reversal_sleeve are the two baseline sleeves. Costs are
parameters so the same definitions can be re-run at other rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest, pairs, signals


def momentum_sleeve(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    lookback: int = 30,
    target_vol: float = 0.15,
    cost_bps: float = 20.0,
    borrow_bps_annual: float = 0.0,
) -> pd.Series:
    """Vol-targeted time-series (trend) momentum.

    Each coin in the universe is held long or short by the sign of its own
    trailing lookback-day return (skipping the latest day), all at the same
    size, and the book is scaled to target_vol annualised. The book is not
    dollar-neutral: its net exposure follows the share of coins in uptrends.
    Market orders, so 20 bps by default. borrow_bps_annual charges carry on
    shorts.
    """
    sig = np.sign(signals.trailing_return(price, lookback, 1)).where(universe)
    w = signals.signal_to_weights(sig, universe, long_short=False, gross_leverage=1.0)
    raw = backtest.run(w, returns, cost_bps=cost_bps, borrow_bps_annual=borrow_bps_annual)
    return backtest.vol_target(raw.net_returns, target_vol).rename("momentum")


def reversal_sleeve(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    cost_bps: float = 7.0,
    borrow_bps_annual: float = 0.0,
    entry: float = 2.0,
    exit: float = 0.0,
    zwin: int = 30,
    min_corr: float = 0.8,
    top_k: int = 20,
    start: str = "2020-01-01",
) -> pd.Series:
    """Dynamic pairs trading. Limit orders, so 7 bps by default.

    borrow_bps_annual charges carry on the short leg of each pair.
    """
    net, _turn, _info = pairs.backtest_pairs(
        price, returns, universe, start=start, cost_bps=cost_bps,
        borrow_bps_annual=borrow_bps_annual,
        entry=entry, exit=exit, zwin=zwin, min_corr=min_corr, top_k=top_k,
    )
    return net.rename("reversal")
