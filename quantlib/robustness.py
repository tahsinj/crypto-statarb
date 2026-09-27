"""Portfolio-level helpers that work on sleeve return series.

walk_forward_weights builds the combined book in notebook 06. capacity_curve,
regime_table and by_year are for follow-up checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

TRADING_DAYS = 365


def _mv_weights(train: pd.DataFrame, ridge: float = 1e-6) -> np.ndarray:
    """Max-Sharpe (tangency) weights, w proportional to inv(cov) @ mean, scaled
    so that sum(|w|) = 1.

    A small ridge on the diagonal keeps the solve stable when sleeves are
    close to collinear.
    """
    mu = train.mean().values
    cov = train.cov().values + ridge * np.eye(train.shape[1])
    try:
        w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w = np.ones(len(mu))
    denom = np.abs(w).sum()
    return w / denom if denom else np.ones(len(mu)) / len(mu)


def walk_forward_weights(
    sleeves: pd.DataFrame,
    train_days: int = 756,
    step_days: int = 63,
    min_train: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """Re-estimate the combination weights every ``step_days``, out of sample.

    Weights for each block are fitted on the ``train_days`` rows before it and
    applied to the next ``step_days`` rows only, so no block uses its own
    returns or anything later. The first block starts once ``train_days`` rows
    exist. ``min_train`` is a floor on the training length; because the loop
    starts at ``train_days`` it never binds.

    Returns (walk_forward_combined_returns, weight_path).
    """
    sleeves = sleeves.dropna()
    idx = sleeves.index
    weight_path = pd.DataFrame(index=idx, columns=sleeves.columns, dtype=float)
    combined = pd.Series(index=idx, dtype=float)

    i = train_days
    while i < len(idx):
        block = idx[i:i + step_days]
        train = sleeves.iloc[max(0, i - train_days):i]
        if len(train) >= min_train:
            w = _mv_weights(train)
        else:
            w = np.ones(sleeves.shape[1]) / sleeves.shape[1]
        wser = pd.Series(w, index=sleeves.columns)
        weight_path.loc[block] = w
        combined.loc[block] = (sleeves.loc[block] * wser).sum(axis=1)
        i += step_days

    return combined.dropna().rename("walk_forward"), weight_path.dropna(how="all")


def capacity_curve(
    net_returns: pd.Series,
    turnover: pd.Series,
    adv_usd: float,
    aum_grid,
    impact_coef_bps: float = 50.0,
) -> pd.DataFrame:
    """Sharpe / return vs deployed AUM under a square-root market-impact model.

    As AUM grows the daily traded notional (``turnover * AUM``) is a larger
    fraction ``participation`` of a representative traded-name ADV, and square-root
    impact charges ``impact_coef_bps * sqrt(participation)`` bps per unit traded on
    top of the base costs in ``net_returns``. impact_coef_bps ~ 50 corresponds to
    roughly a day's move at full participation, a standard Almgren-style scale.

    adv_usd is a representative daily dollar volume of the names actually traded.
    Returns a frame indexed by AUM with sharpe, ann_return and mean participation.
    """
    turnover = turnover.reindex(net_returns.index).fillna(0.0)
    rows = {}
    for aum in aum_grid:
        participation = (turnover * aum) / adv_usd
        impact_bps = impact_coef_bps * np.sqrt(participation.clip(lower=0))
        drag = turnover * impact_bps / 1e4          # return drag as a fraction of book
        net_at_aum = net_returns - drag
        rows[aum] = {
            "sharpe": metrics.sharpe(net_at_aum),
            "ann_return": metrics.ann_return(net_at_aum),
            "mean_participation": float(participation.mean()),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "AUM_usd"
    return out


def impact_drag(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    aum: float,
    y: float = 1.0,
    window: int = 30,
) -> pd.Series:
    """Daily market-impact cost, per coin, of running ``weights`` with ``aum`` dollars.

    Square-root law: trading Q dollars of a coin with daily volume V and daily
    volatility sigma moves its price by about y * sigma * sqrt(Q / V), and the
    trade pays that on Q. V is the trailing median of Binance volume and sigma
    the trailing standard deviation of daily returns, both up to the day
    before, so small, volatile coins cost the most. Returns the drag as a
    fraction of the book, to subtract from the sleeve's net returns (which
    already pay the fixed bps cost).
    """
    w = weights.reindex_like(returns).fillna(0.0)
    trade = (w - w.shift(1)).abs()
    adv = dollar_volume.reindex_like(returns).rolling(window, min_periods=window // 2).median().shift(1)
    sigma = returns.rolling(window, min_periods=window // 2).std().shift(1)
    impact = y * sigma * np.sqrt(trade * aum / adv.where(adv > 0))
    return (impact * trade).sum(axis=1).rename("impact")


def capacity_by_coin(
    net_returns: pd.Series,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    aum_grid,
    y: float = 1.0,
) -> pd.DataFrame:
    """Sharpe and return of a sleeve after per-coin impact, for each AUM in aum_grid."""
    rows = {}
    for aum in aum_grid:
        drag = impact_drag(weights, returns, dollar_volume, aum, y=y).reindex(net_returns.index).fillna(0.0)
        r = net_returns - drag
        rows[aum] = {"sharpe": metrics.sharpe(r), "ann_return": metrics.ann_return(r),
                     "impact_per_year": float(drag.mean() * 365)}
    out = pd.DataFrame(rows).T
    out.index.name = "AUM_usd"
    return out


def max_capacity(cap: pd.DataFrame, sharpe_floor_frac: float = 0.5) -> float:
    """Largest AUM whose Sharpe is still >= ``sharpe_floor_frac`` of the smallest-AUM Sharpe."""
    base = cap["sharpe"].iloc[0]
    ok = cap[cap["sharpe"] >= sharpe_floor_frac * base]
    return float(ok.index.max()) if len(ok) else float(cap.index.min())


def regime_table(returns: pd.Series, regime: pd.Series) -> pd.DataFrame:
    """Sharpe / ann return / vol / n by regime bucket (regime aligned to returns)."""
    df = pd.concat([returns.rename("r"), regime.rename("regime")], axis=1).dropna()
    rows = {}
    for label, grp in df.groupby("regime", observed=True):
        r = grp["r"]
        rows[label] = {
            "sharpe": metrics.sharpe(r),
            "ann_return": metrics.ann_return(r),
            "ann_vol": metrics.ann_vol(r),
            "n_days": len(r),
        }
    return pd.DataFrame(rows).T


def by_year(returns: pd.Series) -> pd.DataFrame:
    """Sharpe / ann return / max drawdown per calendar year."""
    r = returns.dropna()
    rows = {}
    for yr, grp in r.groupby(r.index.year):
        rows[yr] = {
            "sharpe": metrics.sharpe(grp),
            "ann_return": metrics.ann_return(grp),
            "max_drawdown": metrics.max_drawdown(grp),
            "n_days": len(grp),
        }
    return pd.DataFrame(rows).T
