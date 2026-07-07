"""Sleeve definitions shared by the notebooks.

momentum_sleeve and reversal_sleeve are the two baselines tested in notebook
01. seasonal_momentum_sleeve, orderflow_sleeve and carry_sleeve are frozen
copies of the configurations selected in notebooks 02-04; notebook 06 checks
each against its saved research parquet before it reads any lockbox data.
Costs are parameters so the same definitions can be re-run at other rates.
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


def seasonal_momentum_sleeve(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    lookback: int = 30,
    target_vol: float = 0.15,
    cost_bps: float = 20.0,
    weekend: float = 1.0,
    weekday: float = 0.5,
) -> pd.Series:
    """Momentum sleeve with weekday/weekend scaling (notebook 02 selection).

    Returns signals.seasonal_scale(momentum_sleeve(...), weekend, weekday). The
    defaults (weekend=1.0, weekday=0.5) are the configuration that passed
    notebook 02's gate check. Notebook 06 compares the output with
    sleeve_seasonality.parquet.
    """
    mom = momentum_sleeve(price, returns, universe,
                          lookback=lookback, target_vol=target_vol,
                          cost_bps=cost_bps)
    return signals.seasonal_scale(mom, weekend=weekend, weekday=weekday).rename("seasonality")


def orderflow_sleeve(
    taker_imbalance: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    smooth: int = 10,
    cost_bps: float = 7.0,
) -> pd.Series:
    """Taker-imbalance follow signal, 10-day smoothing (notebook 03 selection).

    Signal = cross_sectional_zscore((taker_imbalance - 0.5).rolling(smooth).mean(),
    universe), long the high buy-share names. With smooth=1 the rolling mean
    is skipped. Dollar-neutral long/short, 7 bps by default. Notebook 06
    compares the output with sleeve_orderflow.parquet.
    """
    imb_c = taker_imbalance - 0.5
    raw = imb_c.rolling(smooth).mean() if smooth > 1 else imb_c
    sig = signals.cross_sectional_zscore(raw, universe)
    w = signals.signal_to_weights(sig, universe, long_short=True)
    return backtest.run(w, returns, cost_bps).net_returns.rename("orderflow")


def carry_sleeve(
    funding: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    smooth: int = 7,
    cost_bps: float = 7.0,
) -> pd.Series:
    """Short high funding, long low funding (notebook 04 selection, P1 k=7).

    Trades the universe names that have funding data (fuv). Signal =
    -cross_sectional_zscore(funding.rolling(smooth).mean(), fuv). The smoothing
    is only in the signal: the funding P&L uses the actual daily funding,
    because that is what a perp position pays or receives. Dollar-neutral
    long/short, 7 bps by default. Notebook 06 compares the output with
    sleeve_carry.parquet.
    """
    fuv = universe & funding.notna()
    smooth_fund = funding.rolling(smooth).mean()
    sig = -signals.cross_sectional_zscore(smooth_fund, fuv)
    w = signals.signal_to_weights(sig, fuv, long_short=True)
    res = backtest.run(w, returns, cost_bps)
    fund_pnl = -(w.shift(1) * funding).sum(axis=1)
    net = res.net_returns + fund_pnl
    return net.rename("carry")
