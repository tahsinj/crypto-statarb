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
    cross-sectional rank instead, so no single coin can take more than a few
    percent of the book (v2).
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
    fallback_returns: pd.DataFrame | None = None,
) -> pd.Series:
    """Short high funding, long low funding (notebook 04 selection, P1 k=7).

    Trades the universe names that have funding data (fuv). Signal =
    -cross_sectional_zscore(funding.rolling(smooth).mean(), fuv). The smoothing
    is only in the signal: the funding P&L uses the actual daily funding,
    because that is what a perp position pays or receives. Dollar-neutral
    long/short, 7 bps by default. Notebook 06 compares the output with
    sleeve_carry.parquet. ``returns`` can be spot or perp returns; the research
    used spot. weighting='rank' is the v2 rule (see carry_weights).

    ``fallback_returns`` fills in a coin's return on the first day of each gap
    in ``returns`` (backtest.fill_gap_starts). The every-pair perp panels drop a
    contract's data on days it trades more than 20% away from spot, so a coin
    the sleeve holds on such a day earns nothing, even on a crash day like
    LUNA's in May 2022. Passing the spot returns gives it the coin's spot move
    instead. On those panels the sleeve cannot hold a coin past that first
    day, since the coin's funding is dropped on the same days. The default
    leaves the gaps, as the first runs did.
    """
    w = carry_weights(funding, universe, smooth, weighting)
    if fallback_returns is not None:
        returns = backtest.fill_gap_starts(returns, fallback_returns)
    res = backtest.run(w, returns, cost_bps)
    fund_pnl = -(w.shift(1) * funding).sum(axis=1)
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

# Added after v2's first run and before any test result was computed: the
# measurement fix spot_fallback=True, which notebook 12 uses. It changes how
# P&L is counted on the few days a held perp's data is dropped, not what the
# sleeves trade. The other fix made then, filling two holes in the archive's
# 2022 perp files, is in the data (notebook 09); it gives Carry funding it was
# missing, so it also changes what Carry holds for about a week after each
# hole. Added later: "none chosen by performance" above means none was chosen
# on v2's own results. The equal weights were picked on the frozen book's dev
# and gate results, and rank weights also answer the DEXE case of the forward
# window (notebook 08).


def v2_sleeves(
    taker_imbalance: pd.DataFrame,
    returns: pd.DataFrame,
    perp_returns: pd.DataFrame,
    funding: pd.DataFrame,
    universe: pd.DataFrame,
    cost_bps: float = 7.0,
    spot_fallback: bool = False,
) -> pd.DataFrame:
    """The two v2 sleeves: Orderflow (unchanged rule) and rank-weighted Carry on perp prices.

    spot_fallback=True gives Carry a held coin's spot move on the day its perp
    data drops out (see carry_sleeve). The default is v2 as registered and
    first run in notebook 10.
    """
    of = orderflow_sleeve(taker_imbalance, returns, universe, cost_bps=cost_bps)
    ca = carry_sleeve(funding, perp_returns, universe, cost_bps=cost_bps, weighting="rank",
                      fallback_returns=returns if spot_fallback else None)
    return pd.concat([of, ca], axis=1)


def v2_book(sleeves: pd.DataFrame) -> pd.Series:
    """Equal weight of the v2 sleeves, rebalanced daily."""
    return sleeves.dropna().mean(axis=1).rename("v2")
