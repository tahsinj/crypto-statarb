"""Keyless fetchers for the raw price-volume data.

Crypto bars are free to pull yourself (yfinance,
Kraken or Binance). This module uses Binance's public REST endpoints:

- ``fetch_binance``: spot klines, daily or hourly. Keeps the close, the USD
  quote volume and the taker-buy quote volume. Standard library only.
- ``fetch_funding``: USDT-margined perpetual funding rates, summed per UTC day.
- ``fetch_yahoo``: a yfinance fallback for daily bars. No taker volume, patchier
  volume data and heavy rate limiting, so it is not used for the research.

Each returns a date-indexed frame with a (symbol, field) column index.
Notebook 00 calls them through ``refresh``, which caches the result under
``data/raw/``::

    from quantlib import fetch
    raw = fetch.refresh("data/raw/binance_1d.pkl.zip",
                        lambda: fetch.fetch_binance(top_n=150, start="2018-01-01"))
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import STABLECOINS

BINANCE_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"
_INTERVAL_MS = {"1d": 86_400_000, "1h": 3_600_000}
# Leveraged tokens and wrapped/pegged assets are not real spot exposures and
# would pollute a momentum/reversal universe; drop them alongside stablecoins.
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
_WRAPPED = {"WBTC", "WETH", "WBETH", "BETH", "STETH", "WEETH"}


def _get_json(url: str, retries: int = 4, pause: float = 0.5):
    """GET a JSON endpoint with a browser UA and exponential backoff on 429/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) or e.code >= 500:
                time.sleep(pause * (2**attempt))
                continue
            raise
        except urllib.error.URLError:
            time.sleep(pause * (2**attempt))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts")


def _base_symbol(pair: str, quote: str) -> str:
    """'BTCUSDT' -> 'BTC' (the ticker the rest of the project uses)."""
    return pair[: -len(quote)] if pair.endswith(quote) else pair


def top_usdt_pairs(top_n: int = 150, quote: str = "USDT") -> list[str]:
    """The ``top_n`` most-traded spot ``*USDT`` pairs by 24h quote volume.

    Excludes stablecoins, wrapped assets, and leveraged tokens so the resulting
    universe is real, unlevered spot exposures.
    """
    tickers = _get_json(f"{BINANCE_BASE}/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith(quote):
            continue
        base = _base_symbol(sym, quote)
        if base in STABLECOINS or base in _WRAPPED:
            continue
        if any(base.endswith(sfx) for sfx in _LEVERAGED_SUFFIXES):
            continue
        rows.append((sym, float(t.get("quoteVolume", 0.0))))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [sym for sym, _ in rows[:top_n]]


def _klines(pair: str, start_ms: int, end_ms: int, interval: str = "1d") -> pd.DataFrame:
    """All bars for one pair in [start, end], following Binance's 1000-bar pages.

    Returns a Date-indexed frame with ``close``, ``quote_volume`` (USD) and
    ``taker_quote`` (taker-buy quote volume).
    """
    out = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{BINANCE_BASE}/api/v3/klines?symbol={pair}&interval={interval}"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        batch = _get_json(url)
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < 1000:
            break
        cursor = last_open + _INTERVAL_MS[interval]  # next bar after the last one returned
        time.sleep(0.15)  # stay well under Binance's weight limits
    if not out:
        return pd.DataFrame(columns=["close", "quote_volume", "taker_quote"])
    df = pd.DataFrame(
        out,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("Date")[["close", "quote_volume", "taker_quote"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


def fetch_binance(
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    top_n: int = 150,
    quote: str = "USDT",
    interval: str = "1d",
) -> pd.DataFrame:
    """Raw (symbol, field) frame from Binance spot klines.

    ``symbols`` may be bare tickers (``['BTC', 'ETH']``) or full pairs
    (``['BTCUSDT']``); if None, the top ``top_n`` pairs by 24h volume are used.
    Fields: ``price`` is the bar close, ``volume`` the quote volume in USD and
    ``taker`` the taker-buy quote volume.
    """
    if symbols is None:
        pairs = top_usdt_pairs(top_n=top_n, quote=quote)
    else:
        pairs = [s if s.endswith(quote) else f"{s}{quote}" for s in symbols]

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").normalize()
    end_ms = int(end_ts.timestamp() * 1000)

    frames = {}
    for pair in pairs:
        base = _base_symbol(pair, quote)
        try:
            bars = _klines(pair, start_ms, end_ms, interval)
        except Exception as e:  # noqa: BLE001 - skip a bad pair, keep the rest
            print(f"binance: skipping {pair}: {e}")
            continue
        if bars.empty:
            continue
        frames[(base, "price")] = bars["close"]
        frames[(base, "volume")] = bars["quote_volume"]
        frames[(base, "taker")] = bars["taker_quote"]

    if not frames:
        raise RuntimeError("binance fetch returned no data for any symbol")
    raw = pd.DataFrame(frames).sort_index()
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    raw.index.name = "Date"
    return raw.sort_index(axis=1)


def fetch_funding(
    symbols: list[str],
    start: str = "2019-09-01",
    end: str | None = None,
    quote: str = "USDT",
) -> pd.DataFrame:
    """Daily perp funding rates (sum of the day's 8h prints) per symbol.

    Positive = longs pay shorts. Symbols with no perp listing are skipped.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").normalize()
    end_ms = int(end_ts.timestamp() * 1000)

    out = {}
    for sym in symbols:
        pair = sym if sym.endswith(quote) else f"{sym}{quote}"
        base = _base_symbol(pair, quote)
        rows, cursor = [], start_ms
        try:
            while cursor < end_ms:
                url = (
                    f"{FAPI_BASE}/fapi/v1/fundingRate?symbol={pair}"
                    f"&startTime={cursor}&endTime={end_ms}&limit=1000"
                )
                batch = _get_json(url)
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < 1000:
                    break
                cursor = int(batch[-1]["fundingTime"]) + 1
                time.sleep(0.15)
        except Exception as e:  # noqa: BLE001 - no perp / delisted: skip, keep the rest
            print(f"funding: skipping {pair}: {e}")
            continue
        if not rows:
            continue
        s = pd.Series(
            [float(r["fundingRate"]) for r in rows],
            index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms"),
        )
        out[base] = s.resample("1D").sum(min_count=1)

    if not out:
        raise RuntimeError("funding fetch returned no data for any symbol")
    df = pd.DataFrame(out).sort_index()
    df.index.name = "Date"
    return df


def fetch_yahoo(
    symbols: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Fallback fetch via yfinance.

    ``symbols`` are bare tickers; each is queried as ``TICKER-USD``. ``price`` is
    the close and ``volume`` is Yahoo's reported USD volume. Yahoo throttles and
    its crypto volume is less reliable than Binance's, so this is a fallback.
    """
    import yfinance as yf  # local import: only needed for this path

    tickers = [f"{s}-USD" for s in symbols]
    data = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=True, progress=False, group_by="ticker",
    )
    if data is None or data.empty:
        raise RuntimeError("yahoo fetch returned no data (likely rate-limited)")

    frames = {}
    for s, tk in zip(symbols, tickers):
        try:
            sub = data[tk] if len(tickers) > 1 else data
        except KeyError:
            continue
        close = sub.get("Close")
        vol = sub.get("Volume")
        if close is None or close.dropna().empty:
            continue
        frames[(s, "price")] = close
        frames[(s, "volume")] = vol if vol is not None else np.nan
    if not frames:
        raise RuntimeError("yahoo fetch produced no usable series")
    raw = pd.DataFrame(frames).sort_index()
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"
    return raw.sort_index(axis=1)


def save_raw(raw: pd.DataFrame, path: str | Path) -> Path:
    """Write the raw panel to a (optionally zipped) pickle under data/raw/.

    A ``.pkl.zip`` path is written as a zip archive holding one ``.pkl`` member,
    matching ``data.load_raw``'s reader.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        inner = path.name[:-4] if path.name.endswith(".zip") else "raw.pkl"
        if not inner.endswith(".pkl"):
            inner += ".pkl"
        tmp = path.parent / inner
        raw.to_pickle(tmp)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(tmp, arcname=inner)
        tmp.unlink()
    else:
        raw.to_pickle(path)
    return path


def refresh(
    path: str | Path,
    fetch_fn,
    max_age_days: float = 1.0,
    force: bool = False,
) -> pd.DataFrame:
    """Cached fetch: reuse `path` if younger than max_age_days, else refetch.

    On fetch failure with an existing cache, warn loudly and serve the cache;
    with no cache, re-raise. Notebook 00 wraps every network call in this.
    """
    from .data import load_raw  # local import to avoid a cycle at module load

    path = Path(path)
    if path.exists() and not force:
        age_days = (time.time() - path.stat().st_mtime) / 86_400
        if age_days <= max_age_days:
            return load_raw(path)
    try:
        raw = fetch_fn()
    except Exception as e:  # noqa: BLE001
        if path.exists():
            print(f"WARNING: fetch failed ({e}); using stale cache {path.name}")
            return load_raw(path)
        raise
    save_raw(raw, path)
    return raw
