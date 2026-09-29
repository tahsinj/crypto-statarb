# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 09: Every Binance pair, delisted coins included
#
# The research coin list is the 150 USDT pairs with the most 24-hour volume
# on the day of the fetch, 2026-07-06. Most coins that had been delisted by
# then, and those that had shrunk out of the top 150, are not in it, so the
# universe held mostly coins that were still big in mid-2026.
# Binance's public data archive (data.binance.vision) keeps daily files for
# delisted pairs as well. This notebook downloads every USDT pair and perp
# from it, checks the result against the research data where the two
# overlap, and writes the panels notebooks 10 to 12 use. It computes no
# strategy results.
#
# Three mistakes in the research coin list show up along the way. The filter
# for leveraged tokens dropped any name ending in UP, which took out JUP and
# SYRUP. Six pegged assets got through: the euro, two gold tokens (PAXG,
# XAUT) and three newer dollar tokens (RLUSD, USD1, U); EUR and PAXG sat in
# the universe for most of the sample. And five tokenized US stocks, which
# Binance began listing in June 2026, were in the list (CRCLB, MSTRB, MUB,
# SNDKB, SPCXB). They were too new to enter the research universe, but not
# the forward test's. Here the leveraged-token check needs the rest of the
# name to be a listed coin (BTCUP is BTC plus UP), and the pegged assets and
# tokenized stocks are listed in `data.PEGGED` and `data.TOKENIZED_STOCKS`.
# The list also held 11 coins that had stopped trading before the fetch, from
# POLY (2022-10) to TON (2026-06-30).

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import data, fetch

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
ARCHIVE = RAW / "archive"          # the archive's own file paths, cached
OUT = PROC / "all_pairs"
OUT.mkdir(parents=True, exist_ok=True)

END = "2026-09-26"                 # the forward test's last day
RESEARCH_END = "2026-07-05"        # last full day of the research data
# -

# ## Which pairs

# +
def listing(name: str, fn) -> pd.Series:
    """An archive listing, cached like the data, so a rerun needs no network."""
    return fetch.refresh(RAW / f"archive_listing_{name}.pkl.zip", lambda: pd.Series(fn()), max_age_days=float("inf"))


spot_pairs = listing("spot", lambda: fetch.archive_pairs("spot")).tolist()
bases = [p[:-4] for p in spot_pairs]
keep = set(data.tradable_bases(bases))
spot_keep = [p for p in spot_pairs if p[:-4] in keep]
print(f"{len(spot_pairs)} USDT pairs in the archive, {len(spot_keep)} kept")
print("left out:", sorted(set(bases) - keep))

research_raw = data.load_raw(RAW / "binance_1d.pkl.zip")
research_funding = data.load_raw(RAW / "funding.pkl.zip")
research_coins = sorted(set(research_raw.columns.get_level_values(0)))
print("research coins left out here:", sorted(set(research_coins) - keep))
print("research coins the archive does not have:", sorted(set(research_coins) - set(bases)))

# a new tokenized stock would show up as a B-suffixed listing from mid-2026
first_month = listing("spot_b_first_month", lambda: {p[:-4]: str(min(fetch.archive_months("spot", p)))
                                                     for p in spot_pairs if p[:-4].endswith("B")})
unknown = sorted(b for b, m in first_month.items()
                 if pd.Period(m, "M") >= pd.Period("2026-06", "M") and b not in data.TOKENIZED_STOCKS)
print("B-suffixed pairs listed since June 2026 and not in data.TOKENIZED_STOCKS:", unknown)
# -

# ## Download
#
# About a hundred thousand small files, cached under `data/raw/archive/`, so
# a rerun only fetches what is new. Spot bars start in 2018 like the
# research data; perps and funding start in September 2019, when Binance
# launched them. The archive writes funding a month at a time, so the days
# after the last monthly file come from the API.

# +
raw_spot = fetch.refresh(RAW / "archive_spot_1d.pkl.zip",
                         lambda: fetch.fetch_archive("spot", spot_keep, "2018-01-01", END, ARCHIVE, workers=24),
                         max_age_days=float("inf"))
raw_spot = raw_spot.loc[:, raw_spot.columns.get_level_values(0).isin(keep)]
# the research list's own non-crypto coins, only to rebuild that list on archive data
research_extra = fetch.refresh(RAW / "archive_research_extra_1d.pkl.zip",
                               lambda: fetch.fetch_archive("spot", [c + "USDT" for c in sorted(set(research_coins) - keep)],
                                                           "2018-01-01", END, ARCHIVE, workers=24),
                               max_age_days=float("inf"))

perp_pairs = listing("um", lambda: fetch.archive_pairs("um")).tolist()
perp_names = data.perp_to_spot([p[:-4] for p in perp_pairs], sorted(keep))
perp_keep = sorted(p + "USDT" for p in perp_names)
raw_um = fetch.refresh(RAW / "archive_um_1d.pkl.zip",
                       lambda: fetch.fetch_archive("um", perp_keep, "2019-09-01", END, ARCHIVE, workers=24),
                       max_age_days=float("inf"))
# the monthly perp files of about 50 contracts miss a few days (the last three
# of February 2022, the first two of April 2022); the daily files exist
um_holes = fetch.refresh(RAW / "archive_um_1d_holes.pkl.zip",
                         lambda: fetch.fill_short_days("um", raw_um, "2020-01-01", END, ARCHIVE, workers=24),
                         max_age_days=float("inf"))
print("perp bars taken from daily files:",
      {d.date(): int(n) for d, n in um_holes.xs("price", axis=1, level=1).notna().sum(axis=1).items()})
raw_um = raw_um.combine_first(um_holes)

fund_pairs = set(listing("funding", lambda: fetch.archive_pairs("funding")))
fund_keep = [p for p in perp_keep if p in fund_pairs]
funding_archive = fetch.refresh(RAW / "archive_funding.pkl.zip",
                                lambda: fetch.fetch_archive("funding", fund_keep, "2019-09-01", END, ARCHIVE,
                                                            workers=24),
                                max_age_days=float("inf"))
last_file_day = funding_archive.index.max()
live = sorted(c for c in funding_archive.columns
              if funding_archive[c].loc[last_file_day - pd.Timedelta(days=3):].notna().any())
recent_start = (last_file_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
funding_recent = fetch.refresh(RAW / "archive_funding_recent.pkl.zip",
                               lambda: fetch.fetch_funding(live, start=recent_start, end="2026-09-27"),
                               max_age_days=float("inf")).loc[recent_start:END]
# the archive's funding files start in 2020-01; the API has the months before
early = sorted(c for c in funding_archive.columns if funding_archive[c].first_valid_index() <= pd.Timestamp("2020-01-02"))
funding_early = fetch.refresh(RAW / "archive_funding_2019.pkl.zip",
                              lambda: fetch.fetch_funding(early, start="2019-09-01", end="2020-01-01"),
                              max_age_days=float("inf")).loc[:"2019-12-31"]
funding_perp = pd.concat([funding_early, funding_archive, funding_recent]).sort_index()
funding_perp = funding_perp[~funding_perp.index.duplicated(keep="last")]

# a few contracts traded before their first archive funding file (ICP's
# files start in 2022-09, its perp in 2021-05); the API fills those days
um_first = raw_um.xs("price", axis=1, level=1).apply(lambda s: s.first_valid_index())
fund_first = funding_perp.apply(lambda s: s.first_valid_index())
late = {c: (um_first[c], fund_first[c]) for c in funding_perp.columns
        if c in um_first.index and pd.notna(um_first[c]) and fund_first[c] - um_first[c] >= pd.Timedelta(days=2)}
funding_gaps = fetch.refresh(
    RAW / "archive_funding_gaps.pkl.zip",
    lambda: pd.concat({c: fetch.fetch_funding([c], start=f"{a:%Y-%m-%d}", end=f"{b:%Y-%m-%d}")[c]
                       for c, (a, b) in late.items()}, axis=1),
    max_age_days=float("inf"))
funding_perp = funding_perp.combine_first(funding_gaps.loc[:, funding_gaps.columns.isin(funding_perp.columns)])
print("funding filled from the API before the first archive file:",
      {c: f"{a:%Y-%m} to {b:%Y-%m}" for c, (a, b) in late.items()})

# the research list also fetched funding for two gold-token perps (PAXG,
# XAUT); only the like-for-like rebuild of that list uses them
extra_perps = sorted(c for c in research_funding.columns if f"{c}USDT" not in fund_keep)
funding_extra = pd.concat([
    fetch.refresh(RAW / "archive_research_extra_funding.pkl.zip",
                  lambda: fetch.fetch_archive("funding", [c + "USDT" for c in extra_perps], "2019-09-01", END,
                                              ARCHIVE, workers=8),
                  max_age_days=float("inf")),
    fetch.refresh(RAW / "archive_research_extra_funding_recent.pkl.zip",
                  lambda: fetch.fetch_funding(extra_perps, start=recent_start, end="2026-09-27"),
                  max_age_days=float("inf")).loc[recent_start:END],
]).sort_index()
research_list_funding = pd.concat([funding_perp, funding_extra], axis=1)[list(research_funding.columns)]

print("spot:", raw_spot.shape, raw_spot.index.min().date(), "->", raw_spot.index.max().date())
print("perps:", raw_um.shape, raw_um.index.min().date(), "->", raw_um.index.max().date())
print(f"funding: {funding_perp.shape}, archive files to {last_file_day.date()}, API after that for {len(live)} live "
      f"perps and for 2019 for {len(early)}")
# no day may be short of bars across the market (a file missing, or published late)
for name, raw, first in [("spot", raw_spot, "2018-01-01"), ("perps", raw_um, "2020-01-01")]:
    short = fetch.short_days(raw, first, END)
    print(f"{name}: days short of bars: {[d.date() for d in short.index]}")
    assert short.empty
# -

# ## Check against the research data
#
# Past daily bars do not change, so for the coins in both sets every close,
# volume and taker volume up to 2026-07-05 should be identical, and so should
# the funding of the 55 perps the research used. The forward test's API
# download (notebook 08) is checked the same way for July to September.


# +
def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    b = b.reindex(index=a.index, columns=a.columns)
    both = a.notna() & b.notna()
    same = np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12)
    diff = both.to_numpy() & ~same
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(b.to_numpy() / a.to_numpy() - 1)[diff]
    return {"compared": int(both.to_numpy().sum()), "differ": int(diff.sum()),
            "median gap when they differ": float(np.median(rel)) if diff.any() else 0.0,
            "only in the first": int((a.notna() & b.isna()).to_numpy().sum()),
            "only in the archive": int((a.isna() & b.notna()).to_numpy().sum())}


archive_coins = set(raw_spot.columns.get_level_values(0))
common = [c for c in research_coins if c in archive_coins]
forward_raw = data.load_raw(RAW / "forward_1d.pkl.zip")
fwd_common = [c for c in sorted(set(forward_raw.columns.get_level_values(0))) if c in archive_coins]
checks = {}
for field in ["price", "volume", "taker"]:
    checks[f"research {field}"] = compare(research_raw.xs(field, axis=1, level=1).loc[:RESEARCH_END, common],
                                          raw_spot.xs(field, axis=1, level=1))
    checks[f"forward {field}"] = compare(forward_raw.xs(field, axis=1, level=1).loc["2026-07-06":END, fwd_common],
                                         raw_spot.xs(field, axis=1, level=1))
checks["research funding"] = compare(research_funding.loc[:RESEARCH_END], research_list_funding)
checks = pd.DataFrame(checks).T
checks.astype({c: int for c in ["compared", "differ", "only in the first", "only in the archive"]})
# -

# Prices and funding match exactly. A few hundred volumes differ, all on a
# handful of days between 2018 and 2021, by a few hundredths of a percent at
# the median. The four funding values only the research has are CHZ's first
# days, in December 2020, which the archive does not have.

# ## Perp names and prices
#
# Perps are mapped to the spot coin they track: '1000SHIB' is SHIB, since
# the contract holds a thousand coins. The mapping is checked by the
# correlation of daily perp and spot returns, which should be close to 1.
# Before any cleaning, some contracts are far from it. The eight lowest, whose
# perp/spot price ratio wanders away from the contract size:

# +
um_price = raw_um.xs("price", axis=1, level=1)
spot_price = raw_spot.xs("price", axis=1, level=1)
by_spot = {}
for perp, coin in perp_names.items():
    if perp in um_price.columns:
        by_spot.setdefault(coin, []).append(perp)
print("spot coins with more than one perp:", {k: v for k, v in by_spot.items() if len(v) > 1})


def perp_spot_corr(perp_ret: pd.DataFrame) -> pd.Series:
    out = {}
    for coin in perp_ret.columns:
        both = pd.concat([perp_ret[coin], spot_price[coin].pct_change()], axis=1, sort=True).dropna()
        if len(both) > 30:
            out[coin] = both.corr().iloc[0, 1]
    return pd.Series(out).sort_values()


raw_perp_ret = pd.DataFrame({coin: um_price[perps[0]].pct_change() for coin, perps in by_spot.items()})
weak = perp_spot_corr(raw_perp_ret).head(8)
ratio = {c: (um_price[by_spot[c][0]] / (data.contract_size(by_spot[c][0], c) * spot_price[c])).dropna() for c in weak.index}
pd.DataFrame({"correlation": weak.round(3),
              "price ratio by year": {c: r.groupby(r.index.year).median().round(2).to_dict() for c, r in ratio.items()}})
# -

# Most of this comes from the archive itself. After a perp is delisted
# the archive keeps writing a bar for it every day, with the last price
# frozen and zero volume, so the ratio drifts as spot keeps moving:

stale = (raw_um.xs("volume", axis=1, level=1) == 0) & um_price.notna()
print(f"frozen zero-volume perp bars: {int(stale.to_numpy().sum()):,} in {int(stale.any().sum())} contracts")

# `data.archive_panels` keeps a perp's data, funding included, only on days
# when it traded and its price is within 20% of the spot price times the
# contract size, and a perp return only when the day before passed too. DEXE's
# perp closed 10.5% below spot at the worst of its July crash, so it passes,
# but a few real dislocations go past 20%: OMG's perp traded 27% below spot on
# 2021-11-11, and LUNA's and FTT's crashes went past it too. The rule decides
# which perps a strategy can pick. A position already held earns what its
# contract did, so the panels ending in `_traded` keep each contract's own
# bars and funding on every day it traded, however far from spot.

# ## Gaps inside a coin's history
#
# A few coins stop trading for a while and come back: suspensions, token
# migrations, and one ticker that changed hands. The Terra coin LUNA trades
# until 2022-05-13, when it collapsed to almost nothing, and the relaunched
# LUNA takes over the ticker from 2022-05-31. Its new perp is LUNA2, mapped
# back to LUNA. Returns are never computed across a gap, and a coin needs 30
# straight days of prices before it can enter the universe, so a gap cannot
# leak into a backtest as a fake move.

# +
spot_px = raw_spot.xs("price", axis=1, level=1)
gaps = {}
for c in spot_px.columns:
    s = spot_px[c]
    inside = s.loc[s.first_valid_index():s.last_valid_index()]
    if inside.isna().any():
        run = inside.isna().astype(int).groupby(inside.notna().cumsum()).sum()
        if run.max() >= 5:
            last_trade = inside.index[inside.notna().cumsum() == run.idxmax()][0]
            start = last_trade + pd.Timedelta(days=1)
            gaps[c] = {"longest gap (days)": int(run.max()), "gap starts": start.date()}
pd.DataFrame(gaps).T.sort_values("longest gap (days)", ascending=False)
# -

# ## Panels and the universe
#
# Built with notebook 00's rules: the 100 coins with the highest 30-day
# median dollar volume up to the day before, a $1M floor and 30 days of
# history.

# +
panels = data.archive_panels(raw_spot, raw_um, funding_perp, perp_names)
pv = panels["perp_valid"]
spot_on = pd.DataFrame({perp: spot_price[perp_names[perp]].reindex(pv.index) for perp in pv.columns}).notna()
traded = raw_um.xs("volume", axis=1, level=1).reindex(pv.index)[pv.columns] > 0
mismatch = (~pv & traded & spot_on).sum()
print("days a live perp traded more than 20% away from spot, by contract:",
      mismatch[mismatch > 0].sort_values(ascending=False).head(10).to_dict(),
      f"({int((mismatch > 0).sum())} contracts, {int(mismatch.sum())} days)")
print(f"after it, median perp/spot return correlation {perp_spot_corr(panels['perp_returns']).median():.3f}, "
      f"lowest {perp_spot_corr(panels['perp_returns']).head(3).round(3).to_dict()}")
universe = data.build_universe(panels, top_n=100, min_adv_usd=1_000_000)
bench = pd.DataFrame({"BTC": panels["returns"]["BTC"], "MKT": data.market_return(panels, universe)})
print("median universe size since 2020:", int(universe.loc["2020-01-01":].sum(axis=1).median()))

# a pegged coin would show up as an almost flat price; none should be left
vol90 = panels["returns"].rolling(90, min_periods=60).std()
flat = (vol90 < 0.01) & universe
print("universe coins with under 1% daily volatility over 90 days:", sorted(flat.columns[flat.any()]))
# TRX (Tron) is a real coin that has simply been calm; it stays
# -

# ## What the research coin list missed
#
# The research list is the top 150 by volume in July 2026, so it holds mostly
# coins that were still big then. Coins that were in the top 100 years
# earlier and faded or were delisted are missing from its history.

# +
DATA_END = "2026-07-06"
outside = ~universe.columns.isin(research_coins)
coin_days = universe.loc["2020-01-01":DATA_END]
share = coin_days.loc[:, outside].sum(axis=1) / coin_days.sum(axis=1)
print(f"share of universe coin-days on coins outside the research list, 2020-01 to 2026-07: "
      f"{coin_days.loc[:, outside].to_numpy().sum() / coin_days.to_numpy().sum():.0%}")
print(share.groupby(share.index.year).mean().round(2).to_string())
missed = coin_days.loc[:, outside].sum().sort_values(ascending=False)
still_listed = spot_px.loc["2026-09-20":].notna().any()
pd.DataFrame({"universe days": missed.head(20),
              "still trading on Binance": still_listed.reindex(missed.head(20).index)})
# -

# ## Save
#
# Everything later notebooks need, under `data/processed/all_pairs/`.

names = ["price", "returns", "dollar_volume", "taker_imbalance", "high", "low", "perp_price", "perp_high",
         "perp_low", "perp_returns", "perp_dollar_volume", "funding", "perp_price_traded", "perp_high_traded",
         "perp_low_traded", "perp_returns_traded", "funding_traded"]
for name in names:
    panels[name].to_parquet(OUT / f"{name}.parquet")
universe.to_parquet(OUT / "universe.parquet")
bench.to_parquet(OUT / "benchmarks.parquet")
print("saved", len(names) + 2, "panels to", OUT.relative_to(ROOT))

# The research coin list on archive data, including its non-crypto coins, and
# the 55 perps it fetched funding for: notebook 11 reruns the frozen book on
# it as a like-for-like check, and notebook 12 extends the frozen book with it.
research_list = pd.concat([raw_spot.loc[:, raw_spot.columns.get_level_values(0).isin(research_coins)],
                           research_extra], axis=1).sort_index(axis=1)
rl = data.to_panels(research_list)
(OUT / "research_list").mkdir(exist_ok=True)
for name in ["price", "returns", "dollar_volume", "taker_imbalance"]:
    rl[name].to_parquet(OUT / "research_list" / f"{name}.parquet")
research_list_funding.to_parquet(OUT / "research_list" / "funding.parquet")
print(f"research list: {rl['price'].shape[1]} coins, funding for {research_funding.shape[1]} perps")

# ## Hourly bars for the fast-reversal check
#
# Notebook 05's hourly test used 60 coins picked by volume at fetch time.
# Here every coin that was in the daily top-100 universe at any point from
# 2020-06 to the end of the gate gets hourly bars, from the month before it
# first entered to the day it last left. Only closes are kept.

# +
HOURLY_START, HOURLY_END = "2020-06-01", "2025-06-30"
member = universe.loc["2020-06-01":HOURLY_END]
ranges = {}
for coin in member.columns[member.any()]:
    days = member.index[member[coin]]
    ranges[f"{coin}USDT"] = ((days.min() - pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                             days.max().strftime("%Y-%m-%d"))
raw_1h = fetch.refresh(RAW / "archive_spot_1h.pkl.zip",
                       lambda: fetch.fetch_archive("spot", sorted(ranges), "2020-05-01", HOURLY_END, ARCHIVE,
                                                   workers=24, interval="1h", ranges=ranges, fields=("price",)),
                       max_age_days=float("inf"))
price_1h = raw_1h.xs("price", axis=1, level=1).sort_index().loc[:HOURLY_END]
print(f"hourly closes: {price_1h.shape[1]} coins, {price_1h.index.min()} to {price_1h.index.max()}")

# the research hourly panel's coins should match it where the two overlap
research_1h = pd.read_parquet(PROC / "price_1h.parquet")
both = [c for c in research_1h.columns if c in price_1h.columns]
a = research_1h.loc[HOURLY_START:HOURLY_END, both]
b = price_1h.reindex(index=a.index, columns=both)
m = a.notna() & b.notna()
print(f"hourly closes compared with the research panel: {int(m.to_numpy().sum()):,}, "
      f"differ: {int((m & ~pd.DataFrame(np.isclose(a, b, rtol=1e-9), index=a.index, columns=both)).to_numpy().sum())}")
price_1h.to_parquet(OUT / "price_1h.parquet")
# -

# The 91 hourly closes that differ from the research panel are off by a
# quarter of a percent at the median. They fall on 19 days: 16 with one to
# five closes each, 15 of them in December 2021, and three with more
# (2020-12-21, 2021-04-23 and 2022-04-13). Too few to matter.
