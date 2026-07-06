"""Fetch-layer schema tests. No network: fetch._get_json is monkeypatched manually."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import data, fetch  # noqa: E402

DAY_MS = 86_400_000
HOUR_MS = 3_600_000


def _kline(open_ms, close_px, quote_vol=1e6, taker_quote=4e5):
    # 12 fields, matching Binance /api/v3/klines
    return [open_ms, "1", "2", "0.5", str(close_px), "10", open_ms + DAY_MS - 1,
            str(quote_vol), 100, "5", str(taker_quote), "0"]


def _patch(fake):
    orig = fetch._get_json
    fetch._get_json = fake
    return orig


def test_taker_field_in_raw_and_panels():
    rows = [_kline(i * DAY_MS, i, quote_vol=1e6, taker_quote=6e5) for i in range(400, 410)]
    orig = _patch(lambda url, **kw: rows)
    try:
        raw = fetch.fetch_binance(symbols=["BTC"], start="1971-01-01", end="1971-03-01")
    finally:
        fetch._get_json = orig
    assert set(raw.columns.get_level_values(1)) == {"price", "volume", "taker"}
    assert raw[("BTC", "price")].iloc[-1] == 409.0
    assert raw[("BTC", "taker")].iloc[0] == 6e5

    panels = data.to_panels(raw)
    assert "taker_imbalance" in panels
    assert np.allclose(panels["taker_imbalance"]["BTC"].dropna(), 0.6)


def test_panels_without_taker_field():
    idx = pd.date_range("2020-01-01", periods=5)
    cols = pd.MultiIndex.from_product([["BTC"], ["price", "volume"]])
    raw = pd.DataFrame(np.ones((5, 2)), index=idx, columns=cols)
    panels = data.to_panels(raw)
    assert "taker_imbalance" not in panels
    assert set(panels) == {"price", "dollar_volume", "returns"}


def test_hourly_pagination_step():
    calls = []

    def fake(url, **kw):
        calls.append(url)
        # parse startTime from the url
        start = int(url.split("startTime=")[1].split("&")[0])
        if len(calls) == 1:
            return [_kline(start + i * HOUR_MS, 100 + i) for i in range(1000)]
        return [_kline(start + i * HOUR_MS, 200 + i) for i in range(10)]

    orig = _patch(fake)
    try:
        bars = fetch._klines("BTCUSDT", 0, 2000 * HOUR_MS, interval="1h")
    finally:
        fetch._get_json = orig
    assert len(bars) == 1010
    # second page must start exactly one HOUR after the last bar of page one
    second_start = int(calls[1].split("startTime=")[1].split("&")[0])
    assert second_start == 999 * HOUR_MS + HOUR_MS


def test_fetch_binance_interval_kwarg():
    rows = [_kline(i * HOUR_MS, 100) for i in range(48)]
    orig = _patch(lambda url, **kw: rows)
    try:
        raw = fetch.fetch_binance(symbols=["ETH"], start="1970-01-01", end="1970-01-10", interval="1h")
    finally:
        fetch._get_json = orig
    assert len(raw) == 48


def test_fetch_funding_daily_agg():
    day0 = 400 * DAY_MS

    def fake(url, **kw):
        if "fundingRate" not in url:
            raise AssertionError(f"unexpected url {url}")
        if "NOPERPUSDT" in url:
            import urllib.error
            raise urllib.error.HTTPError(url, 400, "no perp", None, None)
        # 3 funding events on day0 (00:00, 08:00, 16:00), 1 on day0+1
        return [
            {"symbol": "BTCUSDT", "fundingTime": day0, "fundingRate": "0.0001"},
            {"symbol": "BTCUSDT", "fundingTime": day0 + 8 * HOUR_MS, "fundingRate": "0.0002"},
            {"symbol": "BTCUSDT", "fundingTime": day0 + 16 * HOUR_MS, "fundingRate": "0.0003"},
            {"symbol": "BTCUSDT", "fundingTime": day0 + DAY_MS, "fundingRate": "-0.0001"},
        ]

    orig = _patch(fake)
    try:
        f = fetch.fetch_funding(["BTC", "NOPERP"], start="1971-01-01", end="1971-03-01")
    finally:
        fetch._get_json = orig
    assert list(f.columns) == ["BTC"]  # NOPERP skipped, not fatal
    assert np.isclose(f["BTC"].iloc[0], 0.0006)   # 8h rates summed per day
    assert np.isclose(f["BTC"].iloc[1], -0.0001)


def _tiny_raw():
    idx = pd.date_range("2024-01-01", periods=3)
    cols = pd.MultiIndex.from_product([["BTC"], ["price", "volume"]])
    return pd.DataFrame(np.ones((3, 2)), index=idx, columns=cols)


def test_refresh_cache_and_fallback():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test_1d.pkl.zip"
        calls = {"n": 0}

        def fetcher():
            calls["n"] += 1
            return _tiny_raw()

        r1 = fetch.refresh(path, fetcher)               # cold: fetches
        assert calls["n"] == 1 and len(r1) == 3
        r2 = fetch.refresh(path, fetcher)               # warm: cache hit
        assert calls["n"] == 1
        fetch.refresh(path, fetcher, force=True)        # force: fetches
        assert calls["n"] == 2

        def broken():
            raise RuntimeError("api down")

        r4 = fetch.refresh(path, broken, force=True)    # falls back to stale cache
        assert len(r4) == 3

        os.remove(path)
        try:
            fetch.refresh(path, broken)
            raise AssertionError("should have raised with no cache")
        except RuntimeError:
            pass


def test_load_raw_single_file():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pkl.zip"
        fetch.save_raw(_tiny_raw(), p)
        loaded = data.load_raw(p)
        assert loaded.shape == (3, 2)


def test_utc_epoch_windows():
    captured = []

    def fake(url, **kw):
        captured.append(url)
        return []  # empty batch ends pagination immediately

    orig = _patch(fake)
    try:
        try:
            fetch.fetch_binance(symbols=["BTC"], start="2020-01-01", end="2020-01-02")
        except RuntimeError:
            pass  # "no data" is expected with an empty fake
        try:
            fetch.fetch_funding(["BTC"], start="2020-01-01", end="2020-01-02")
        except RuntimeError:
            pass
    finally:
        fetch._get_json = orig
    for url in captured:
        start_ms = int(url.split("startTime=")[1].split("&")[0])
        assert start_ms == 1577836800000, f"start not UTC-pinned in {url}"


if __name__ == "__main__":
    for fn in [test_taker_field_in_raw_and_panels, test_panels_without_taker_field, test_hourly_pagination_step, test_fetch_binance_interval_kwarg, test_fetch_funding_daily_agg, test_refresh_cache_and_fallback, test_load_raw_single_file, test_utc_epoch_windows]:
        fn()
        print(f"ok {fn.__name__}")
