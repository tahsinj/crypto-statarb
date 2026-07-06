"""Panels and the point-in-time universe.

The raw frame built by quantlib.fetch is indexed by date and has a two-level
column index (symbol, field). Fields are 'price' (close), 'volume' (USD quote
volume) and, for Binance klines, 'taker' (taker-buy quote volume). Everything
downstream works with plain dates x symbols panels.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Stablecoins carry no momentum/reversal signal and would pollute cross-sectional
# ranks, so they are excluded from every universe.
STABLECOINS = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDD", "USDP", "GUSD",
    "FRAX", "LUSD", "USDE", "PYUSD", "EURT", "EURS", "VNST", "USTC", "CRVUSD",
    "USD0", "USDJ", "SUSD", "DOLA", "MIM", "USDX", "GHO",
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw frame written by fetch.save_raw (.pkl, or .pkl.zip)."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            inner = next(n for n in z.namelist() if n.endswith(".pkl"))
            return pd.read_pickle(io.BytesIO(z.read(inner)))
    return pd.read_pickle(path)


def to_panels(
    raw: pd.DataFrame,
    drop_stablecoins: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split the raw MultiIndex frame into price, dollar_volume and returns panels."""
    raw = raw.sort_index()
    price = raw.xs("price", axis=1, level=1).copy()
    volume = raw.xs("volume", axis=1, level=1).copy()

    if drop_stablecoins:
        keep = [c for c in price.columns if c not in STABLECOINS]
        price, volume = price[keep], volume[keep]

    price = price.sort_index(axis=1)
    volume = volume.reindex_like(price)

    fields = set(raw.columns.get_level_values(1))
    taker = None
    if "taker" in fields:
        taker = raw.xs("taker", axis=1, level=1).reindex_like(price)

    price = price.where(price > 0)  # non-positive prices are bad ticks
    returns = price.pct_change()

    out = {"price": price, "dollar_volume": volume, "returns": returns}
    if taker is not None:
        # Share of volume initiated by aggressive buyers; ~0.5 = balanced flow.
        out["taker_imbalance"] = (taker / volume).where(volume > 0)
    return out


def build_universe(
    panels: dict[str, pd.DataFrame],
    top_n: int | None = 100,
    min_adv_usd: float | None = None,
    adv_window: int = 30,
    min_history: int = 30,
) -> pd.DataFrame:
    """Build a point-in-time tradable universe (boolean dates x symbols).

    A coin is eligible on date t if, using only data through t, it has at least
    min_history trailing prices and its trailing adv_window median dollar-volume
    clears the liquidity bar (an absolute floor and/or top_n by ADV). Trailing
    medians (not point values) avoid selecting on the same-day volume spike that
    tends to accompany a return.
    """
    price = panels["price"]
    dvol = panels["dollar_volume"]

    # Shift by 1 so date t only sees volume through t-1: no look-ahead.
    adv = dvol.rolling(adv_window, min_periods=max(5, adv_window // 2)).median().shift(1)
    has_history = price.notna().rolling(min_history, min_periods=min_history).sum().shift(1) >= min_history

    eligible = adv.notna() & has_history
    if min_adv_usd is not None:
        eligible &= adv >= min_adv_usd

    if top_n is not None:
        ranks = adv.where(eligible).rank(axis=1, ascending=False, method="first")
        eligible &= ranks <= top_n

    return eligible.fillna(False)


def coverage_report(panels: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """Per-date diagnostics: coins with a price, and tradable-universe size."""
    price = panels["price"]
    return pd.DataFrame(
        {
            "coins_with_price": price.notna().sum(axis=1),
            "universe_size": universe.sum(axis=1),
        }
    )


def market_return(panels: dict[str, pd.DataFrame], universe: pd.DataFrame | None = None) -> pd.Series:
    """Equal-weight crypto-market daily return, used as a benchmark alongside BTC."""
    rets = panels["returns"]
    if universe is not None:
        rets = rets.where(universe.shift(1).fillna(False))
    return rets.mean(axis=1)
