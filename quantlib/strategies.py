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
    charge_resizing: bool = False,
) -> pd.Series:
    """Momentum sleeve with weekday/weekend scaling (notebook 02 selection).

    Returns signals.seasonal_scale(momentum_sleeve(...), weekend, weekday). The
    defaults (weekend=1.0, weekday=0.5) are the configuration that passed
    notebook 02's gate check. Notebook 06 compares the output with
    sleeve_seasonality.parquet.

    The default version charges costs on the momentum weights only, not on the
    trades that resize the book when the multiplier or the vol-target scale
    changes. charge_resizing=True rebuilds the daily positions actually held
    (weights x vol scale x day multiplier) and charges cost_bps on every change
    in them. Notebook 07 uses it to re-score the book; the frozen sleeve keeps
    the default.
    """
    if not charge_resizing:
        mom = momentum_sleeve(price, returns, universe,
                              lookback=lookback, target_vol=target_vol,
                              cost_bps=cost_bps)
        return signals.seasonal_scale(mom, weekend=weekend, weekday=weekday).rename("seasonality")

    sig = np.sign(signals.trailing_return(price, lookback, 1)).where(universe)
    w = signals.signal_to_weights(sig, universe, long_short=False, gross_leverage=1.0)
    raw = backtest.run(w, returns, cost_bps=cost_bps)
    idx = raw.net_returns.index
    scale = backtest.vol_target_scale(raw.net_returns, target_vol)
    mult = pd.Series(np.where(idx.dayofweek >= 5, weekend, weekday), index=idx)
    held = w.shift(1).reindex(idx).mul(scale * mult, axis=0)   # position held over day t
    gross = (held * returns.reindex(idx)).sum(axis=1)
    trades = (held - held.shift(1)).abs().sum(axis=1)           # trades made at close t-1
    net = gross - trades * (cost_bps / 1e4)
    return net.where(scale.notna()).rename("seasonality")


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
    w = orderflow_weights(taker_imbalance, universe, smooth)
    return backtest.run(w, returns, cost_bps).net_returns.rename("orderflow")


def orderflow_weights(taker_imbalance: pd.DataFrame, universe: pd.DataFrame, smooth: int = 10) -> pd.DataFrame:
    """Orderflow target weights: long the names with the highest recent buy share."""
    imb_c = taker_imbalance - 0.5
    raw = imb_c.rolling(smooth).mean() if smooth > 1 else imb_c
    sig = signals.cross_sectional_zscore(raw, universe)
    return signals.signal_to_weights(sig, universe, long_short=True)


def carry_weights(
    funding: pd.DataFrame,
    universe: pd.DataFrame,
    smooth: int = 7,
    weighting: str = "zscore",
) -> pd.DataFrame:
    """Carry target weights: short high funding, long low funding.

    weighting='zscore' is the frozen research rule. weighting='rank' uses the
    cross-sectional rank instead (v2): with 40 or more coins no single coin gets
    more than about 5% of the sleeve, though with only a handful, as in the
    first weeks of funding data, each side is still one or two coins.
    """
    fuv = universe & funding.notna()
    smooth_fund = funding.rolling(smooth).mean()
    if weighting == "zscore":
        sig = -signals.cross_sectional_zscore(smooth_fund, fuv)
    elif weighting == "rank":
        sig = -signals.cross_sectional_rank(smooth_fund, fuv)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return signals.signal_to_weights(sig, fuv, long_short=True)


def carry_sleeve(
    funding: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    smooth: int = 7,
    cost_bps: float = 7.0,
    weighting: str = "zscore",
    pnl_funding: pd.DataFrame | None = None,
) -> pd.Series:
    """Short high funding, long low funding (notebook 04 selection, P1 k=7).

    Trades the universe names that have funding data (fuv). Signal =
    -cross_sectional_zscore(funding.rolling(smooth).mean(), fuv). The smoothing
    is only in the signal: the funding P&L uses the actual daily funding,
    because that is what a perp position pays or receives. Dollar-neutral
    long/short, 7 bps by default. Notebook 06 compares the output with
    sleeve_carry.parquet. ``returns`` can be spot or perp returns; the research
    used spot. weighting='rank' is the v2 rule (see carry_weights).

    ``pnl_funding`` is the funding a held position pays or receives, when it
    differs from the funding the signal reads. The every-pair panels drop a
    perp's data on days it trades more than 20% away from spot, which decides
    what the sleeve can pick, but a position held into such a day still paid
    or received its contract's funding and moved with its contract's price:
    pass the panels' funding_traded here and perp_returns_traded as
    ``returns`` (data.archive_panels). The default reads ``funding`` for both,
    as the first runs did.
    """
    w = carry_weights(funding, universe, smooth, weighting)
    res = backtest.run(w, returns, cost_bps)
    paid = funding if pnl_funding is None else pnl_funding.reindex_like(funding)
    fund_pnl = -(w.shift(1) * paid).sum(axis=1)
    net = res.net_returns + fund_pnl
    return net.rename("carry")


# ---------------------------------------------------------------------------
# Version 2, registered on 2026-09-27 before any v2 result was computed
# ---------------------------------------------------------------------------
# Changes from the frozen book, each for a reason found before the lockbox or
# in the data audit, none chosen by performance:
#   - Seasonality is dropped: charged for its resizing trades it fails
#     notebook 02's own dev/gate rule (notebook 07).
#   - Carry weights by rank, not z-score: z-scores let one coin take a whole
#     side of the sleeve (SOL in November 2022 on dev).
#   - Carry is measured on perp prices, which is what the trade holds.
#   - The universe comes from every Binance pair, delisted ones included, with
#     pegged assets and tokenized stocks out and JUP/SYRUP back in (notebook 09).
#   - The two sleeves are combined with equal weights, which beat the
#     walk-forward weights on both dev and gate.
# The test is the data from V2_START on (notebook 12).

V2_START = "2026-09-28"

# Added after v2's first run, once its test window had opened but before any
# test result was computed: how a held Carry position is measured on a day the
# 20% rule drops its perp's data. It first took the coin's spot move
# (committed 2026-09-28 UTC); since 2026-09-29 it earns the contract's own
# return and funding, or nothing once the contract has stopped trading
# (v2_sleeves' pnl_funding with the *_traded panels), which notebook 12 uses.
# That changes what v2 earns on a few days, not what it picks. The other fix,
# filling two holes in the archive's 2022 perp files (2026-09-29), is in the
# data (notebook 09); it gives Carry funding it was missing, so it also
# changes what Carry holds for about a week after each hole. Added later:
# "none chosen by performance" above means none was chosen on v2's own
# results. The equal weights were picked on the frozen book's dev and gate
# results, and rank weights also answer the DEXE case of the forward window
# (notebook 08).


def v2_sleeves(
    taker_imbalance: pd.DataFrame,
    returns: pd.DataFrame,
    perp_returns: pd.DataFrame,
    funding: pd.DataFrame,
    universe: pd.DataFrame,
    cost_bps: float = 7.0,
    pnl_funding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The two v2 sleeves: Orderflow (unchanged rule) and rank-weighted Carry on perp prices.

    As registered and first run (notebook 10), ``perp_returns`` and ``funding``
    are the every-pair panels, which drop a perp's data on days it trades more
    than 20% away from spot, so a coin held into such a day earns nothing. The
    test (notebook 12) passes the panels' perp_returns_traded as
    ``perp_returns`` and funding_traded as ``pnl_funding``: a held perp earns its
    contract's own return and funding on every day it traded, and nothing once
    it stopped. The signal still reads ``funding`` (see carry_sleeve).
    """
    of = orderflow_sleeve(taker_imbalance, returns, universe, cost_bps=cost_bps)
    ca = carry_sleeve(funding, perp_returns, universe, cost_bps=cost_bps, weighting="rank",
                      pnl_funding=pnl_funding)
    return pd.concat([of, ca], axis=1)


def v2_book(sleeves: pd.DataFrame) -> pd.Series:
    """Equal weight of the v2 sleeves, rebalanced daily."""
    return sleeves.dropna().mean(axis=1).rename("v2")
