"""Pairs trading, used for the reversal baseline in notebook 01.

Every rebalance_days (63 by default) the engine picks pairs among names that
were in the universe for most of the trailing year: return correlation above
min_corr, a positive hedge ratio on log prices, and a spread half-life between
2 and 60 days. Each pair then trades its spread z-score with hysteresis.

Everything is point-in-time: selection only sees data up to the rebalance
date, the spread z-score uses a trailing mean and std shifted by one day, and
pair returns use the previous day's position.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _half_life(spread: pd.Series) -> float:
    """Mean-reversion speed via an AR(1) / Ornstein-Uhlenbeck fit."""
    s = spread.dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).loc[ds.index]
    if len(ds) < 30:
        return np.inf
    beta = np.polyfit(lag, ds, 1)[0]
    return -np.log(2) / beta if beta < 0 else np.inf


def select_pairs(
    logp: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    asof: pd.Timestamp,
    lookback_days: int = 365,
    top_k: int = 30,
    min_corr: float = 0.7,
    min_presence: float = 0.9,
    hl_bounds: tuple[float, float] = (2.0, 60.0),
):
    """Select up to top_k mean-reverting pairs using data up to asof.

    A pair qualifies if, over the trailing window, both coins are in the universe
    most of the time, their returns correlate above min_corr, the OLS hedge ratio
    is positive, and the spread's half-life is inside hl_bounds. Ranked by
    correlation.
    """
    win = slice(asof - pd.Timedelta(days=lookback_days), asof)
    presence = universe.loc[win].mean()
    cands = presence[presence > min_presence].index.tolist()
    if len(cands) < 5:
        return []

    corr = returns.loc[win, cands].corr()
    out = []
    for a, b in combinations(cands, 2):
        c = corr.loc[a, b]
        if not (c >= min_corr):
            continue
        df = pd.concat([logp.loc[win, a], logp.loc[win, b]], axis=1).dropna()
        if len(df) < 200:
            continue
        beta = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
        if beta <= 0:
            continue
        hl = _half_life(df.iloc[:, 0] - beta * df.iloc[:, 1])
        if hl_bounds[0] < hl < hl_bounds[1]:
            out.append((a, b, float(beta), float(c)))
    return sorted(out, key=lambda x: -x[3])[:top_k]


def _pair_stream(logp, returns, a, b, beta, idx, entry, exit, zwin):
    """Vectorised entry/exit z-score trading of one pair over ``idx``."""
    spread = logp[a] - beta * logp[b]
    m = spread.rolling(zwin, min_periods=zwin).mean().shift(1)
    sd = spread.rolling(zwin, min_periods=zwin).std().shift(1)
    z = (spread - m) / sd

    # +1 = long spread (long a / short b), -1 = short spread, 0 = flat. NaN holds
    # the prior state (hysteresis): enter at |z|>entry, exit only inside |z|<exit.
    raw = pd.Series(np.nan, index=spread.index)
    raw[z > entry] = -1.0
    raw[z < -entry] = 1.0
    raw[z.abs() < exit] = 0.0
    pos = raw.ffill().fillna(0.0)

    wa = pos / (1 + beta)            # dollar-neutral, gross ~1 per active pair
    wb = -pos * beta / (1 + beta)
    ha, hb = wa.shift(1), wb.shift(1)
    pr = ha * returns[a] + hb * returns[b]
    to = wa.diff().abs() + wb.diff().abs()
    short = ha.clip(upper=0).abs() + hb.clip(upper=0).abs()   # short notional held
    return (pr.reindex(idx).fillna(0.0),
            to.reindex(idx).fillna(0.0),
            short.reindex(idx).fillna(0.0))


def backtest_pairs(
    price: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    start: str = "2020-01-01",
    rebalance_days: int = 63,
    entry: float = 1.5,
    exit: float = 0.5,
    zwin: int = 30,
    cost_bps: float = 7.0,
    borrow_bps_annual: float = 0.0,
    selector=select_pairs,
    **select_kw,
):
    """Run the pairs strategy; returns (net_returns, turnover, info).

    selector is called on each rebalance date and must return (a, b, beta,
    score) tuples the way select_pairs does. borrow_bps_annual charges a daily
    carry on the short notional held (0 = off). info records the pairs chosen
    at each rebalance.
    """
    logp = np.log(price)
    dates = returns.loc[start:].index
    rebal = pd.date_range(dates[0], dates[-1], freq=f"{rebalance_days}D")

    pnl = pd.Series(0.0, index=dates)
    turn = pd.Series(0.0, index=dates)
    short = pd.Series(0.0, index=dates)
    chosen = {}

    for i, rb in enumerate(rebal):
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1] + pd.Timedelta(days=1)
        seg = dates[(dates >= rb) & (dates < end)]
        if len(seg) == 0:
            continue
        pairs = selector(logp, returns, universe, rb, **select_kw)
        chosen[rb] = [(a, b) for a, b, _, _ in pairs]
        if not pairs:
            continue
        prs, tos, sns = [], [], []
        for a, b, beta, _ in pairs:
            pr, to, sn = _pair_stream(logp, returns, a, b, beta, seg, entry, exit, zwin)
            prs.append(pr)
            tos.append(to)
            sns.append(sn)
        pnl.loc[seg] = pd.concat(prs, axis=1).mean(axis=1)        # equal-weight pairs
        turn.loc[seg] = pd.concat(tos, axis=1).mean(axis=1)
        short.loc[seg] = pd.concat(sns, axis=1).mean(axis=1)

    carry = short * (borrow_bps_annual / 1e4) / 365.0
    net = pnl - turn * (cost_bps / 1e4) - carry
    info = {
        "pairs_per_rebal": {k: len(v) for k, v in chosen.items()},
        "chosen": chosen,
        "short_notional": short.rename("short_notional"),  # for carry / capacity analysis
        "gross_pnl": pnl.rename("gross"),
    }
    return net.rename("reversal"), turn.rename("turnover"), info
