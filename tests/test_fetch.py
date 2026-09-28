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


def _zip_csv(path, lines):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(Path(path).name.replace(".zip", ".csv"), "\n".join(lines) + "\n")
    return Path(path)


def _archive_row(open_ms, close_px, unit=1):
    # archive kline rows: same 12 fields as the API, open time in ms or us
    return (f"{open_ms * unit},1,2,0.5,{close_px},10,{(open_ms + DAY_MS - 1) * unit},"
            f"1000000,100,5,400000,0")


def test_read_archive_klines_both_layouts():
    """Old files: no header, ms. Spot files from 2025: microseconds. Perp files: a header."""
    d0 = 18_000 * DAY_MS                      # 2019-04-14
    with tempfile.TemporaryDirectory() as td:
        a = _zip_csv(Path(td) / "a.zip", [_archive_row(d0, 10), _archive_row(d0 + DAY_MS, 11)])
        b = _zip_csv(Path(td) / "b.zip", [_archive_row(d0 + 2 * DAY_MS, 12, unit=1000)])
        c = _zip_csv(Path(td) / "c.zip", [",".join(fetch._KLINE_COLUMNS),
                                            _archive_row(d0 + 3 * DAY_MS, 13)])
        bars = fetch.read_archive_klines([a, b, c])
    assert list(bars.index) == list(pd.date_range("2019-04-14", periods=4))
    assert bars["close"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert bars["taker_quote"].eq(4e5).all() and bars["quote_volume"].eq(1e6).all()


def test_read_archive_funding_daily_sum():
    d0 = 18_000 * DAY_MS
    rows = ["calc_time,funding_interval_hours,last_funding_rate",
            f"{d0},8,0.0001", f"{d0 + 8 * HOUR_MS},8,0.0002", f"{d0 + 16 * HOUR_MS},8,0.0003",
            f"{d0 + DAY_MS},8,-0.0001"]
    with tempfile.TemporaryDirectory() as td:
        f = fetch.read_archive_funding([_zip_csv(Path(td) / "f.zip", rows)])
    assert np.isclose(f.iloc[0], 0.0006) and np.isclose(f.iloc[1], -0.0001)


def test_archive_keys_daily_files_only_for_live_pairs():
    months = {pd.Period(m, "M"): f"x/BTCUSDT-1d-{m}.zip" for m in ["2026-06", "2026-07", "2026-08"]}
    latest = pd.Period("2026-08", "M")
    keys = fetch.archive_keys("spot", months, "BTCUSDT", pd.Timestamp("2026-07-01"),
                              pd.Timestamp("2026-09-03"), latest)
    assert keys[:2] == ["x/BTCUSDT-1d-2026-07.zip", "x/BTCUSDT-1d-2026-08.zip"]
    assert keys[2:] == [f"data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2026-09-0{d}.zip" for d in (1, 2, 3)]
    hourly = fetch.archive_keys("spot", months, "BTCUSDT", pd.Timestamp("2026-08-01"),
                                pd.Timestamp("2026-09-01"), latest, interval="1h")
    assert hourly[-1] == "data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-09-01.zip"
    dead = {pd.Period("2022-05", "M"): "x/LUNAUSDT-1d-2022-05.zip"}
    assert fetch.archive_keys("spot", dead, "LUNAUSDT", pd.Timestamp("2022-01-01"),
                              pd.Timestamp("2026-09-03"), latest) == ["x/LUNAUSDT-1d-2022-05.zip"]


def test_archive_list_follows_pages():
    pages = [
        b"<ListBucketResult><Prefix>data/spot/</Prefix><IsTruncated>true</IsTruncated>"
        b"<NextMarker>data/spot/B/</NextMarker><CommonPrefixes><Prefix>data/spot/A/</Prefix>"
        b"</CommonPrefixes><CommonPrefixes><Prefix>data/spot/B/</Prefix></CommonPrefixes></ListBucketResult>",
        b"<ListBucketResult><Prefix>data/spot/</Prefix><IsTruncated>false</IsTruncated>"
        b"<CommonPrefixes><Prefix>data/spot/C/</Prefix></CommonPrefixes></ListBucketResult>",
    ]
    urls = []

    def fake(url, **kw):
        urls.append(url)
        return pages[len(urls) - 1]

    orig = fetch._get_bytes
    fetch._get_bytes = fake
    try:
        folders, files = fetch.archive_list("data/spot/")
    finally:
        fetch._get_bytes = orig
    assert folders == ["data/spot/A/", "data/spot/B/", "data/spot/C/"] and files == []
    assert "marker=data/spot/B/" in urls[1]


def test_archive_download_quotes_non_ascii_names():
    """Binance lists pairs with Chinese names; their paths must be percent-encoded."""
    urls = []

    def fake(url, **kw):
        urls.append(url)
        return None                                   # as if the file were missing

    orig = fetch._get_bytes
    fetch._get_bytes = fake
    try:
        with tempfile.TemporaryDirectory() as td:
            got = fetch.archive_download(["data/spot/monthly/klines/币安人生USDT/1d/x.zip"], td, workers=1)
    finally:
        fetch._get_bytes = orig
    assert got == [] and urls[0].isascii() and "%E5%B8%81" in urls[0]


def test_archive_panels_join_contracts_without_a_fake_return():
    idx = pd.date_range("2022-05-01", periods=6)
    def frame(values):
        cols = pd.MultiIndex.from_product([list(values), ["price", "volume", "taker", "open", "high", "low"]])
        df = pd.DataFrame(index=idx, columns=cols, dtype=float)
        for coin, px in values.items():
            for f in ["price", "open", "high", "low"]:
                df[(coin, f)] = px
            df[(coin, "volume")], df[(coin, "taker")] = 1e6, 5e5
        return df
    spot = frame({"LUNA": [80.0, 60.0, 0.1, np.nan, 9.0, 8.0]})
    perp = frame({"LUNA": [80.0, 60.0, 0.1, np.nan, np.nan, np.nan],
                  "LUNA2": [np.nan, np.nan, np.nan, np.nan, 9.0, 7.2]})
    funding = pd.DataFrame({"LUNA": [0.0, -0.01, -0.02, np.nan, np.nan, np.nan],
                            "LUNA2": [np.nan] * 4 + [0.001, 0.001]}, index=idx)
    p = data.archive_panels(spot, perp, funding, {"LUNA": "LUNA", "LUNA2": "LUNA"})
    r = p["perp_returns"]["LUNA"]
    assert np.isnan(r.iloc[4]) and np.isclose(r.iloc[5], -0.2)   # no return across the switch
    assert np.isclose(r.iloc[1], -0.25) and p["funding"]["LUNA"].notna().sum() == 5
    assert np.isnan(p["returns"]["LUNA"].iloc[4])                # the spot gap is not bridged either


def test_perp_data_dropped_when_it_stops_tracking_spot():
    idx = pd.date_range("2024-03-25", periods=6)
    cols = pd.MultiIndex.from_product([["STRAX", "SHIB"], ["price", "volume", "taker", "open", "high", "low"]])
    spot = pd.DataFrame(1.0, index=idx, columns=cols)
    spot[("SHIB", "price")] = 2e-5
    perp_cols = pd.MultiIndex.from_product([["STRAX", "1000SHIB"], ["price", "volume", "taker", "open", "high", "low"]])
    perp = pd.DataFrame(1.0, index=idx, columns=perp_cols)
    perp[("STRAX", "price")] = [1.0, 1.02, 0.98, 9.0, 9.1, 9.2]   # a migration: 9x the spot price
    perp[("1000SHIB", "price")] = 2e-2                            # 1000 coins per contract
    perp.loc[idx[-1], ("1000SHIB", "volume")] = 0.0               # a delisted perp's frozen bar
    funding = pd.DataFrame(1e-4, index=idx, columns=["STRAX", "1000SHIB"])
    p = data.archive_panels(spot, perp, funding, {"STRAX": "STRAX", "1000SHIB": "SHIB"})
    assert p["funding"]["STRAX"].notna().tolist() == [True, True, True, False, False, False]
    assert p["perp_returns"]["STRAX"].iloc[3:].isna().all()
    assert p["funding"]["SHIB"].notna().tolist() == [True] * 5 + [False]
    assert data.contract_size("1000SHIB", "SHIB") == 1000 and data.contract_size("1MBABYDOGE", "BABYDOGE") == 1e6
    assert data.contract_size("1000SATS", "1000SATS") == 1 and data.contract_size("LUNA2", "LUNA") == 1


def test_short_days_flags_days_not_yet_published():
    idx = pd.date_range("2026-10-01", periods=12)
    cols = pd.MultiIndex.from_product([[f"C{i}" for i in range(50)], ["price", "volume"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw.loc[idx[:4], [(f"C{i}", "price") for i in range(40, 50)]] = np.nan   # ten coins listed on 10-05
    raw.loc[idx[8], [(f"C{i}", "price") for i in range(3)]] = np.nan          # 10-09: three delisted, a normal day
    raw.loc[idx[-1]] = np.nan                               # 10-12: rows written, no bars yet
    short = fetch.short_days(raw, "2026-10-01", "2026-10-13")  # 10-13: not in the data at all
    assert list(short.index) == list(pd.to_datetime(["2026-10-12", "2026-10-13"])) and (short == 0).all()


def test_leveraged_tokens_and_perp_names():
    bases = {"BTC", "ETH", "JUP", "SYRUP", "BTCUP", "ETHDOWN", "BULL", "SHIB", "SATS", "1000SATS"}
    flagged = {b for b in bases if data.is_leveraged_token(b, bases)}
    assert flagged == {"BTCUP", "ETHDOWN", "BULL"}
    m = data.perp_to_spot(["BTC", "1000SHIB", "1000SATS", "1MBABYDOGE", "BTCDOM"], sorted(bases) + ["BABYDOGE"])
    assert m == {"BTC": "BTC", "1000SHIB": "SHIB", "1000SATS": "1000SATS", "1MBABYDOGE": "BABYDOGE"}


if __name__ == "__main__":
    for fn in [test_taker_field_in_raw_and_panels, test_panels_without_taker_field, test_hourly_pagination_step, test_fetch_binance_interval_kwarg, test_fetch_funding_daily_agg, test_refresh_cache_and_fallback, test_load_raw_single_file, test_utc_epoch_windows,
               test_read_archive_klines_both_layouts, test_read_archive_funding_daily_sum,
               test_archive_keys_daily_files_only_for_live_pairs, test_archive_list_follows_pages,
               test_archive_download_quotes_non_ascii_names, test_archive_panels_join_contracts_without_a_fake_return,
               test_perp_data_dropped_when_it_stops_tracking_spot, test_short_days_flags_days_not_yet_published,
               test_leveraged_tokens_and_perp_names]:
        fn()
        print(f"ok {fn.__name__}")
