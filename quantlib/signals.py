"""Signal construction helpers.

Every function takes and returns dates x symbols panels. A signal is a
cross-sectional score (higher = more attractive to be long); signal_to_weights
turns one into weights and is the only place dollar-neutrality and leverage
normalisation happen.

Nothing here shifts in time. The look-ahead-free contract is enforced once, in
backtest.run, which lags weights before applying returns. Signals are "as of
close t" using data through t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(price: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """L-day return over [t-lookback-skip, t-skip], skipping the most recent skip days.

    skip lets momentum exclude the short-horizon reversal window (the "12-1"
    construction) and lets reversal isolate the most recent window.
    """
    past = price.shift(skip)
    return past / past.shift(lookback) - 1.0


def cross_sectional_zscore(signal: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Z-score each row across the (masked) cross-section."""
    s = signal.where(mask) if mask is not None else signal.copy()
    mu = s.mean(axis=1)
    sd = s.std(axis=1)
    return s.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def cross_sectional_rank(signal: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank each row in [-0.5, 0.5], centred, NaNs preserved."""
    s = signal.where(mask) if mask is not None else signal.copy()
    r = s.rank(axis=1, pct=True)
    return r.sub(0.5)


def winsorize(signal: pd.DataFrame, limit: float = 0.02) -> pd.DataFrame:
    """Clip each row at its limit / 1-limit cross-sectional quantiles."""
    lo = signal.quantile(limit, axis=1)
    hi = signal.quantile(1 - limit, axis=1)
    return signal.clip(lower=lo, upper=hi, axis=0)


def ewm_vol(returns: pd.DataFrame, halflife: int = 30, min_periods: int = 20) -> pd.DataFrame:
    """Per-asset EWMA daily vol, shifted by 1 to stay point-in-time."""
    return returns.ewm(halflife=halflife, min_periods=min_periods).std().shift(1)


def realized_vol(returns: pd.Series, window: int = 30, shift: bool = True) -> pd.Series:
    """Rolling annualised realised vol of a return series, optionally shifted.

    shift=True lags by one day so the value at t only uses returns through t-1,
    which is what you want when using it as a point-in-time regime conditioner.
    """
    rv = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(365)
    return rv.shift(1) if shift else rv


def cross_sectional_dispersion(returns: pd.DataFrame, shift: bool = True) -> pd.Series:
    """Daily cross-sectional dispersion (std across coins), a macro-dislocation proxy."""
    disp = returns.std(axis=1)
    return disp.shift(1) if shift else disp


def seasonal_scale(returns: pd.Series, weekend: float = 1.0, weekday: float = 1.0) -> pd.Series:
    """Scale a strategy's daily returns by a weekday/weekend exposure multiplier.

    The day of week is known in advance, so multiplying each day's realised return
    by a fixed weekday/weekend factor is a look-ahead-free timing overlay (it is
    equivalent to sizing exposure up or down on those days). It does not charge
    for the trades that the resizing needs; see
    strategies.seasonal_momentum_sleeve(charge_resizing=True) for a version that
    does.
    """
    is_weekend = returns.index.dayofweek >= 5
    mult = pd.Series(np.where(is_weekend, weekend, weekday), index=returns.index)
    return (returns * mult).rename(returns.name)


def calendar_mask(index: pd.DatetimeIndex, bucket: str) -> pd.Series:
    """Boolean mask over ``index`` for a named calendar bucket.

    Calendar buckets are known in advance, so masking exposure by them is a
    look-ahead-free timing overlay (same argument as seasonal_scale).
    us_hours approximates the US cash session (14:00-21:00 UTC, weekdays).
    turn_of_month spans the last day of a month through the 3rd of the next.
    """
    if bucket == "weekday":
        vals = index.dayofweek < 5
    elif bucket == "weekend":
        vals = index.dayofweek >= 5
    elif bucket == "us_hours":
        vals = (index.hour >= 14) & (index.hour < 21) & (index.dayofweek < 5)
    elif bucket == "off_hours":
        vals = ~((index.hour >= 14) & (index.hour < 21) & (index.dayofweek < 5))
    elif bucket == "turn_of_month":
        vals = (index.day >= index.days_in_month) | (index.day <= 3)
    else:
        raise ValueError(
            f"unknown bucket {bucket!r}; valid: weekday, weekend, us_hours, "
            "off_hours, turn_of_month"
        )
    return pd.Series(vals, index=index)


def rolling_zscore(
    panel: pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
    shift: bool = True,
) -> pd.DataFrame:
    """Per-column time-series z-score against a trailing window.

    With shift=True (default) the rolling mean and std are lagged one period,
    so date t is scored against data through t-1 only. Signals should use this
    form; shift=False is for descriptive work.
    """
    mp = min_periods if min_periods is not None else max(10, window // 2)
    mean = panel.rolling(window, min_periods=mp).mean()
    std = panel.rolling(window, min_periods=mp).std()
    if shift:
        mean, std = mean.shift(1), std.shift(1)
    return (panel - mean) / std.where(std > 0)


def signal_to_weights(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    long_short: bool = True,
    gross_leverage: float = 1.0,
    inverse_vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert a cross-sectional signal into tradable weights.

    Restricts to the universe. If long_short, demeans each row for dollar
    neutrality and scales gross exposure to gross_leverage. If inverse_vol is
    given, scales the signal by 1/vol first so risk is equalised across names.
    """
    s = signal.where(universe)

    if inverse_vol is not None:
        s = s * (1.0 / inverse_vol.where(universe))

    if long_short:
        s = s.sub(s.mean(axis=1), axis=0)

    gross = s.abs().sum(axis=1).replace(0, np.nan)
    w = s.div(gross, axis=0) * gross_leverage
    return w.fillna(0.0)


def decile_weights(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    quantile: float = 0.2,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Long the top quantile, short the bottom, equal weight within each leg.

    Dollar-neutral, and less sensitive to outliers than linear z-score weighting.
    """
    s = signal.where(universe)
    lo = s.quantile(quantile, axis=1)
    hi = s.quantile(1 - quantile, axis=1)

    longs = s.ge(hi, axis=0) & universe
    shorts = s.le(lo, axis=0) & universe

    w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    nl = longs.sum(axis=1).replace(0, np.nan)
    ns = shorts.sum(axis=1).replace(0, np.nan)
    w = w.add(longs.div(nl, axis=0), fill_value=0.0)
    w = w.sub(shorts.div(ns, axis=0), fill_value=0.0)
    # Each leg sums to 1 -> gross = 2; rescale to the requested leverage.
    return (w * (gross_leverage / 2.0)).fillna(0.0)
