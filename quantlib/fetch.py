"""Keyless fetchers for the raw price-volume data.

Crypto bars are free to pull yourself (yfinance,
Kraken or Binance). This module uses Binance's public REST endpoints:

- ``fetch_binance``: spot klines, daily or hourly. Keeps the close, the USD
  quote volume and the taker-buy quote volume. Standard library only.
- ``fetch_funding``: USDT-margined perpetual funding rates, summed per UTC day.
- ``fetch_archive``: the same daily (or hourly) data from Binance's public data
  archive, which also holds pairs that have since been delisted. Notebook 09
  uses it to rebuild the universe without survivorship.
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

import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import STABLECOINS, WRAPPED

BINANCE_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"
_INTERVAL_MS = {"1d": 86_400_000, "1h": 3_600_000}
# Leveraged tokens and wrapped coins (WBTC and the like) are not real spot
# exposures and would pollute a momentum/reversal universe; top_usdt_pairs
# drops them alongside stablecoins. This is the filter the research coin list
# was built with: it checks the suffix alone, so it also dropped JUP and SYRUP,
# and it lets through the pegged assets in data.PEGGED (see data.py).
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
_WRAPPED = WRAPPED


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
            f"{BINANCE_BASE}/api/v3/klines?symbol={urllib.parse.quote(pair)}&interval={interval}"
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
                    f"{FAPI_BASE}/fapi/v1/fundingRate?symbol={urllib.parse.quote(pair)}"
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


# ---------------------------------------------------------------------------
# Binance's public data archive (data.binance.vision)
# ---------------------------------------------------------------------------
# The REST API only serves pairs that still trade, so a coin list taken from it
# leaves out everything delisted before the fetch. The archive keeps monthly
# files for delisted pairs as well, with the same kline columns as the API
# (taker volume included) and the full funding history of every perp.

ARCHIVE_URL = "https://data.binance.vision/"
ARCHIVE_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_ROOTS = {
    "spot": "data/spot/{freq}/klines/",
    "um": "data/futures/um/{freq}/klines/",
    "funding": "data/futures/um/{freq}/fundingRate/",
}
_KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                  "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def _get_bytes(url: str, retries: int = 5, pause: float = 0.5, missing_ok: bool = False) -> bytes | None:
    """GET raw bytes with backoff. With missing_ok, a missing file returns None."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) and missing_ok:
                return None
            if e.code in (418, 429) or e.code >= 500:
                time.sleep(pause * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(pause * (2**attempt))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts")


def archive_list(prefix: str) -> tuple[list[str], list[str]]:
    """Sub-folders and files directly under ``prefix`` in the archive."""
    folders, files, marker = [], [], ""
    while True:
        url = f"{ARCHIVE_LIST_URL}?delimiter=/&prefix={urllib.parse.quote(prefix)}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        xml = _get_bytes(url).decode()
        folders += [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", xml) if p != prefix]
        files += re.findall(r"<Key>([^<]+)</Key>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            return folders, files
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = nxt.group(1) if nxt else files[-1]


def archive_pairs(market: str = "spot", quote: str = "USDT") -> list[str]:
    """Every ``*USDT`` pair with monthly files in the archive, delisted ones included.

    ``market`` is 'spot', 'um' (USDT-margined perps) or 'funding'.
    """
    root = ARCHIVE_ROOTS[market].format(freq="monthly")
    folders, _ = archive_list(root)
    pairs = [f[len(root):].strip("/") for f in folders]
    return sorted(p for p in pairs if p.endswith(quote) and len(p) > len(quote))


def archive_months(market: str, pair: str, interval: str = "1d") -> dict[pd.Period, str]:
    """The monthly files the archive has for one pair, by month."""
    root = ARCHIVE_ROOTS[market].format(freq="monthly") + pair + "/"
    if market != "funding":
        root += f"{interval}/"
    _, files = archive_list(root)
    out = {}
    for f in files:
        m = re.search(r"-(\d{4}-\d{2})\.zip$", f)
        if m:
            out[pd.Period(m.group(1), "M")] = f
    return out


def archive_keys(
    market: str,
    monthly: dict[pd.Period, str],
    pair: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    latest_month: pd.Period,
    interval: str = "1d",
) -> list[str]:
    """Files covering [start, end] for one pair.

    Monthly files where they exist. A pair still trading in ``latest_month``
    (the newest month with monthly files) also gets daily files for the days
    after it, since the archive only writes a monthly file once the month is
    over. Funding has no daily files.
    """
    first, last = pd.Period(start, "M"), pd.Period(end, "M")
    keys = [k for p, k in sorted(monthly.items()) if first <= p <= last]
    if market == "funding" or not monthly or max(monthly) != latest_month or last <= latest_month:
        return keys
    daily_root = ARCHIVE_ROOTS[market].format(freq="daily") + f"{pair}/{interval}/"
    day0 = max(latest_month.end_time.normalize() + pd.Timedelta(days=1), start)
    return keys + [f"{daily_root}{pair}-{interval}-{d:%Y-%m-%d}.zip" for d in pd.date_range(day0, end)]


def archive_download(keys: list[str], dest: str | Path, workers: int = 16) -> list[Path]:
    """Download archive files not yet under ``dest``; return the local paths that exist.

    Files keep the archive's own paths below ``dest``. Missing files (a daily
    file not written yet, say) are skipped.
    """
    dest = Path(dest)

    def one(key: str) -> None:
        path = dest / key
        if path.exists():
            return
        blob = _get_bytes(ARCHIVE_URL + urllib.parse.quote(key), missing_ok=True)
        if blob is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(blob)
        part.replace(path)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, keys))
    return [dest / k for k in keys if (dest / k).exists()]


def _read_zip_csv(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0])


def read_archive_klines(paths: list[Path]) -> pd.DataFrame:
    """Daily bars from archive kline files, indexed by the bar's open date.

    Handles both layouts the archive uses: files with or without a header row,
    and open times in milliseconds or (spot files from 2025 on) microseconds.
    """
    frames = []
    for p in paths:
        blob = _read_zip_csv(p)
        if not blob.strip():
            continue
        has_header = not blob[:1].isdigit()
        df = pd.read_csv(io.BytesIO(blob), header=0 if has_header else None).iloc[:, :12]
        df.columns = _KLINE_COLUMNS
        frames.append(df)
    cols = ["open", "high", "low", "close", "quote_volume", "taker_quote"]
    if not frames:
        return pd.DataFrame(columns=cols, dtype=float)
    df = pd.concat(frames, ignore_index=True)
    t = df["open_time"].astype("int64").to_numpy()
    t = np.where(t > 10**14, t // 1000, t)
    df.index = pd.DatetimeIndex(pd.to_datetime(t, unit="ms"), name="Date")
    out = df[cols].astype(float)
    return out[~out.index.duplicated(keep="last")].sort_index()


def read_archive_funding(paths: list[Path]) -> pd.Series:
    """Funding payments from archive files, summed per UTC day (like fetch_funding)."""
    frames = [pd.read_csv(io.BytesIO(_read_zip_csv(p))) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    s = pd.Series(df["last_funding_rate"].astype(float).to_numpy(),
                  index=pd.to_datetime(df["calc_time"].astype("int64"), unit="ms"))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("1D").sum(min_count=1)


def fetch_archive(
    market: str,
    pairs: list[str],
    start: str,
    end: str,
    dest: str | Path,
    quote: str = "USDT",
    workers: int = 16,
    interval: str = "1d",
    ranges: dict[str, tuple[str, str]] | None = None,
    fields: tuple[str, ...] = ("price", "volume", "taker", "open", "high", "low"),
) -> pd.DataFrame:
    """Daily data for ``pairs`` from the archive, cached under ``dest``.

    For 'spot' and 'um' the result is a (symbol, field) frame like
    fetch_binance's, with the bar's open, high and low added to price, volume
    and taker. For 'funding' it is a dates x symbols frame of daily sums.
    Symbols keep the archive's base names, so the perp '1000SHIBUSDT' becomes
    '1000SHIB'. A rerun only downloads files that are new. ``interval='1h'``
    gives hourly bars; ``ranges`` can give some pairs their own (start, end);
    ``fields`` picks which kline fields to keep.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    span = {p: tuple(pd.Timestamp(x) for x in (ranges or {}).get(p, (start_ts, end_ts))) for p in pairs}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        monthly = dict(zip(pairs, ex.map(lambda p: archive_months(market, p, interval), pairs)))
    months = [max(m) for m in monthly.values() if m]
    if not months:
        raise RuntimeError(f"archive has no {market} files for these pairs")
    latest = max(months)
    keys = {p: archive_keys(market, monthly[p], p, span[p][0], span[p][1], latest, interval) for p in pairs}
    archive_download([k for ks in keys.values() for k in ks], dest, workers=workers)

    dest = Path(dest)
    out = {}
    for pair, ks in keys.items():
        paths = [dest / k for k in ks if (dest / k).exists()]
        if not paths:
            continue
        base = _base_symbol(pair, quote)
        if market == "funding":
            s = read_archive_funding(paths).loc[start_ts:end_ts]
            if s.notna().any():
                out[base] = s
            continue
        bars = read_archive_klines(paths).loc[span[pair][0]:span[pair][1] + pd.Timedelta(days=1) - pd.Timedelta(1)]
        if bars.empty:
            continue
        for field, col in [("price", "close"), ("volume", "quote_volume"), ("taker", "taker_quote"),
                           ("open", "open"), ("high", "high"), ("low", "low")]:
            if field in fields:
                out[(base, field)] = bars[col]
    if not out:
        raise RuntimeError(f"archive returned no {market} data")
    df = pd.DataFrame(out).sort_index()
    df.index.name = "Date"
    if market != "funding":
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.sort_index(axis=1)


def short_days(raw: pd.DataFrame, start: str, end: str, frac: float = 0.9, lookback: int = 7) -> pd.Series:
    """Bar counts on the days in [start, end] with bars for fewer than ``frac`` of the pairs of the best recent day.

    ``raw`` is a (symbol, field) frame from fetch_archive. Each day's count is
    compared with the most the archive had on any of the ``lookback`` days
    before it, so a day it has not published yet, or has published only in
    part, shows up here, while new listings and a batch of delistings do not.
    The first ``lookback`` days of the window only serve as that comparison.
    The archive has published a day late before (the 2026-09-26 perp bars).
    """
    n = raw.xs("price", axis=1, level=1).notna().sum(axis=1).reindex(pd.date_range(start, end), fill_value=0)
    best = n.rolling(lookback, min_periods=lookback).max().shift(1)
    return n[n < frac * best]


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
