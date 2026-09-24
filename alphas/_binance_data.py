"""Self-contained Binance data helper for the alphas/ directory.

No dependency on quantlib: it uses only the standard library and pandas, so
the alphas can run in the twsq venv without the research package installed.

Public API
----------
taker_imbalance_history(symbols, days, end=None) -> pd.DataFrame
    Daily taker-buy share (taker quote volume / total quote volume) for each
    symbol. Columns = symbols, index = UTC date. NaN where a symbol has no
    Binance listing.

funding_history(symbols, days, end=None) -> pd.DataFrame
    Daily sum of the 8h perp funding prints per symbol. Columns = symbols with
    a Binance perp, index = UTC date. Symbols without a perp are left out.

rows_before(frame, ts) -> pd.DataFrame
    Rows dated strictly before the calendar day of ts. A daily row is only
    complete once its UTC day has closed, so an alpha rebalancing at the start
    of day D may use rows up to D-1.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pandas as pd

BINANCE_BASE = "https://api.binance.com"
FAPI_BASE = "https://fapi.binance.com"
_INTERVAL_MS = 86_400_000  # 1 day in ms


def _get_json(url: str, retries: int = 4, pause: float = 0.5):
    """GET a JSON endpoint with exponential backoff on 429/5xx."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429) or e.code >= 500:
                time.sleep(pause * (2 ** attempt))
                continue
            raise
        except urllib.error.URLError:
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts")


def _window_ms(days: int, end: str | None) -> tuple[int, int]:
    """(start_ms, end_ms) for the `days` calendar days ending at `end` (UTC)."""
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now("UTC").normalize()
    end_ms = int(end_ts.timestamp() * 1000)
    return end_ms - days * _INTERVAL_MS, end_ms


def rows_before(frame: pd.DataFrame, ts) -> pd.DataFrame:
    """Rows dated strictly before the calendar day of ts (see module docstring)."""
    day = pd.Timestamp(ts).normalize()
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return frame[frame.index < day]


def _klines_one(pair: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginate /api/v3/klines for one USDT pair; return Date-indexed frame with
    close, quote_volume, taker_quote."""
    out = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{BINANCE_BASE}/api/v3/klines?symbol={pair}&interval=1d"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        batch = _get_json(url)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + _INTERVAL_MS
        time.sleep(0.15)
    if not out:
        return pd.DataFrame(columns=["close", "quote_volume", "taker_quote"])
    df = pd.DataFrame(
        out,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore",
        ],
    )
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    df = df.set_index("Date")[["close", "quote_volume", "taker_quote"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


def taker_imbalance_history(symbols: list[str], days: int = 30, end: str | None = None) -> pd.DataFrame:
    """Daily taker-buy share for each symbol over the `days` calendar days to `end`.

    taker_share = taker_quote_volume / total_quote_volume. Values lie between
    0 and 1, with 0.5 meaning balanced flow. `end` defaults to today (UTC).
    """
    start_ms, end_ms = _window_ms(days, end)

    series = {}
    for sym in symbols:
        pair = sym if sym.endswith("USDT") else f"{sym}USDT"
        try:
            df = _klines_one(pair, start_ms, end_ms)
            if df.empty or "taker_quote" not in df or "quote_volume" not in df:
                continue
            total = df["quote_volume"].replace(0, float("nan"))
            series[sym] = (df["taker_quote"] / total).rename(sym)
        except Exception as e:  # noqa: BLE001
            print(f"_binance_data: taker_imbalance skipping {sym}: {e}")
    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series).sort_index()
    out.index.name = "Date"
    return out


def funding_history(symbols: list[str], days: int = 30, end: str | None = None) -> pd.DataFrame:
    """Daily sum of 8-hourly perp funding rates over the `days` calendar days to `end`.

    Positive = longs paid shorts. Symbols without a Binance perp are left out
    (no column). `end` defaults to today (UTC).
    """
    start_ms, end_ms = _window_ms(days, end)

    series = {}
    for sym in symbols:
        pair = sym if sym.endswith("USDT") else f"{sym}USDT"
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
        except Exception as e:  # noqa: BLE001
            print(f"_binance_data: funding skipping {sym}: {e}")
            continue
        if not rows:
            continue
        s = pd.Series(
            [float(r["fundingRate"]) for r in rows],
            index=pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms"),
        )
        series[sym] = s.resample("1D").sum(min_count=1)

    if not series:
        return pd.DataFrame()
    out = pd.DataFrame(series).sort_index()
    out.index.name = "Date"
    return out
