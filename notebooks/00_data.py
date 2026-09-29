# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 00: Data from Binance
#
# Fetches the three raw datasets from Binance's public API (cached under
# `data/raw/`) and writes the processed panels that the later notebooks read.
#
# Windows used throughout: development 2020-01 to 2024-07, gate 2024-08 to
# 2025-06, lockbox 2025-07 to the end of the data. Notebook 06 opens the
# lockbox; notebooks 07 and 11 re-score the frozen book on it afterwards and
# say so, and notebook 08 reruns it as a check before the forward window.
#
# The cached data in this project was fetched on 2026-07-06 at about 15:20 UTC,
# so the last day in each panel (2026-07-06) is a partial day. `END` pins the
# sample to that date; without it a rerun would quietly extend the data and
# change every window that runs "to the end". The coin list is pinned too, in
# `data.RESEARCH_COINS`: the top 150 pairs by volume that day, of which 149
# returned data; 11 of those had in fact stopped trading before then, the
# earliest (POLY) in 2022. The cache never expires, so a rerun reads it, and
# only `FORCE = True` downloads again. A new download asks for the same coins
# but cannot match the cache exactly: its last day would be complete rather
# than partial, a pair Binance has since removed may no longer be served, and
# the API refuses US IP addresses.

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import data, fetch

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

FORCE = False       # set True to download again; otherwise the cache is always used
END = "2026-07-06"  # last bar kept (UTC); the fetch stops here
KEEP = float("inf")  # cache age limit in days: never refetch on its own

# ## Daily panel (the research coin list, 2018 to END)

raw_1d = fetch.refresh(
    RAW / "binance_1d.pkl.zip",
    lambda: fetch.fetch_binance(symbols=list(data.RESEARCH_COINS), start="2018-01-01", end=END),
    max_age_days=KEEP, force=FORCE,
)
assert sorted(set(raw_1d.columns.get_level_values(0))) == sorted(data.RESEARCH_COINS), \
    "the daily data is not the research coin list"
panels = data.to_panels(raw_1d)
print(raw_1d.shape, raw_1d.index.min().date(), "->", raw_1d.index.max().date())

# ## Point-in-time universe: top 100 by trailing 30-day median ADV, $1M floor
# The volume is Binance-only, so the floor is lower than an all-exchange
# figure would need to be.

universe = data.build_universe(panels, top_n=100, min_adv_usd=1_000_000)
print("median universe size:", int(universe.loc["2020-01-01":].sum(axis=1).median()))

# ## Hourly panel (top 60 by ADV at fetch time, 2020 to END)
# Only the names the intraday work could realistically trade.

adv_now = panels["dollar_volume"].rolling(30).median().iloc[-1]
top60 = adv_now.dropna().sort_values(ascending=False).head(60).index.tolist()

raw_1h = fetch.refresh(
    RAW / "binance_1h.pkl.zip",
    lambda: fetch.fetch_binance(symbols=top60, start="2020-01-01", end=END, interval="1h"),
    max_age_days=KEEP, force=FORCE,
)
panels_1h = data.to_panels(raw_1h)
print(raw_1h.shape)

# ## Funding rates (the same 60 names, where a perp exists; 2019-09 to END)

funding = fetch.refresh(
    RAW / "funding.pkl.zip",
    lambda: fetch.fetch_funding(top60, start="2019-09-01", end=END),
    max_age_days=KEEP, force=FORCE,
)
print("funding coverage:", funding.shape)

# ## Check: daily closes against hourly closes
# Both panels come from the same Binance trades, so each daily close has to
# equal the close of that day's 23:00 hourly bar. A mismatch would point to a
# paging or timestamp bug in the fetch.

hourly_close = panels_1h["price"]
last_hour = hourly_close[hourly_close.index.hour == 23]
last_hour.index = last_hour.index.normalize()
common = [s for s in last_hour.columns if s in panels["price"].columns]
daily = panels["price"].reindex(index=last_hour.index, columns=common)
gap = ((daily - last_hour[common]).abs() / daily).to_numpy().ravel()
gap = gap[~np.isnan(gap)]
print(f"{gap.size:,} day/symbol closes compared; largest relative gap {gap.max():.1e}")
assert gap.max() < 1e-9, "daily and hourly closes disagree"

# ## Benchmarks: BTC and the equal-weight tradable market

benchmarks = pd.DataFrame({
    "BTC": panels["returns"]["BTC"],
    "MKT": data.market_return(panels, universe),
})

# ## Save the processed panels

panels["price"].to_parquet(PROC / "price.parquet")
panels["returns"].to_parquet(PROC / "returns.parquet")
panels["dollar_volume"].to_parquet(PROC / "dollar_volume.parquet")
panels["taker_imbalance"].to_parquet(PROC / "taker_imbalance.parquet")
universe.to_parquet(PROC / "universe.parquet")
benchmarks.to_parquet(PROC / "benchmarks.parquet")
funding.to_parquet(PROC / "funding.parquet")
panels_1h["price"].to_parquet(PROC / "price_1h.parquet")
panels_1h["dollar_volume"].to_parquet(PROC / "volume_1h.parquet")
panels_1h["taker_imbalance"].to_parquet(PROC / "taker_1h.parquet")
print("saved 10 panels to data/processed/")
