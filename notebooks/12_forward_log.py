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

# # 12: Forward log
#
# The running out-of-sample record, extended once a month. To update it, set
# END to the last complete UTC day, run the notebook and commit it. The
# commit dates then show each month's numbers were written down before the
# next month's data existed.
#
# Nothing here is tuned. Three records are kept:
#
# - The frozen book (v1), from 2026-07-07, the first day the research data
#   did not have: its coin list, sleeves, walk-forward weights and costs.
#   Notebook 08 ran its first 82 days.
# - v2 (notebook 10), from 2026-09-28, the first full day after it was
#   registered. It runs with `spot_fallback=True`: Carry keeps a held coin's
#   spot move on the days its perp data is dropped, a measurement fix made
#   after v2's first run and before its test began (notebook 11).
# - v2's two sleeves on their own, since notebook 10 found that its
#   Orderflow sleeve fails the gate on the every-pair universe.
#
# History up to 2026-09-26 comes from notebook 09. Later days come from the
# Binance archive the same way, so a coin delisted during a month stays in
# the record until its last trade. Funding for the current month, which the
# archive has not written yet, comes from the API.

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import data, fetch, metrics, robustness, strategies

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
ARCHIVE, ALL = RAW / "archive", PROC / "all_pairs"

END = "2026-09-26"                 # set to the last complete UTC day before each run
BASE_END = "2026-09-26"            # notebook 09's data ends here
V1_START, V2_START = "2026-07-07", strategies.V2_START
assert pd.Timestamp(END) < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize(), "END must be a finished day"
# -

# ## Data
#
# Notebook 09's panels, plus any days after 2026-09-26. The new days are
# fetched with a week of overlap, and the overlap has to match.


# +
def panels_from(name_dir: Path, names: list[str]) -> dict:
    return {n: pd.read_parquet(name_dir / f"{n}.parquet") for n in names}


SPOT = ["price", "returns", "dollar_volume", "taker_imbalance", "high", "low"]
PERP = ["perp_price", "perp_high", "perp_low", "perp_returns", "perp_dollar_volume", "funding"]
allp = panels_from(ALL, SPOT + PERP)
rl = panels_from(ALL / "research_list", ["price", "returns", "dollar_volume", "taker_imbalance", "funding"])

if pd.Timestamp(END) > pd.Timestamp(BASE_END):
    start = (pd.Timestamp(BASE_END) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    spot_pairs = fetch.archive_pairs("spot")
    keep = set(data.tradable_bases([p[:-4] for p in spot_pairs]))
    research_coins = list(rl["price"].columns)
    spot_keep = [p for p in spot_pairs if p[:-4] in keep or p[:-4] in research_coins]
    new_spot = fetch.refresh(RAW / f"log_spot_{END}.pkl.zip",
                             lambda: fetch.fetch_archive("spot", spot_keep, start, END, ARCHIVE, workers=24),
                             max_age_days=float("inf"))
    perp_names = data.perp_to_spot([p[:-4] for p in fetch.archive_pairs("um")], sorted(keep))
    new_um = fetch.refresh(RAW / f"log_um_{END}.pkl.zip",
                           lambda: fetch.fetch_archive("um", sorted(p + "USDT" for p in perp_names), start, END,
                                                       ARCHIVE, workers=24),
                           max_age_days=float("inf"))
    fund_names = sorted(set(perp_names) | set(rl["funding"].columns))

    def archive_funding() -> pd.DataFrame:
        try:
            return fetch.fetch_archive("funding", [p + "USDT" for p in fund_names], start, END, ARCHIVE, workers=24)
        except RuntimeError:          # no monthly funding file covers these days yet
            return pd.DataFrame()

    fund_arch = fetch.refresh(RAW / f"log_funding_{END}.pkl.zip", archive_funding, max_age_days=float("inf"))
    api_start = (fund_arch.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if len(fund_arch) else start
    fund_api = fetch.refresh(RAW / f"log_funding_api_{END}.pkl.zip",
                             lambda: fetch.fetch_funding(fund_names, start=api_start,
                                                         end=(pd.Timestamp(END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
                             max_age_days=float("inf")).loc[api_start:END]
    new_funding = pd.concat([fund_arch, fund_api]).sort_index()

    new_all = data.archive_panels(new_spot.loc[:, new_spot.columns.get_level_values(0).isin(keep)],
                                  new_um, new_funding, perp_names)
    new_rl = data.to_panels(new_spot.loc[:, new_spot.columns.get_level_values(0).isin(research_coins)])
    new_rl["funding"] = new_funding.reindex(columns=rl["funding"].columns)
    overlap = slice(start, BASE_END)
    for name, old, new in [("price", allp["price"], new_all["price"]), ("funding", allp["funding"], new_all["funding"])]:
        a, b = old.loc[overlap], new.reindex(columns=old.columns).loc[overlap]
        both = a.notna() & b.notna()
        assert np.allclose(a[both].fillna(0), b[both].fillna(0), rtol=1e-9, atol=1e-12), f"{name} overlap differs"

    def join(old: dict, new: dict) -> dict:
        """Old days, then new ones; the overlap also fills days the archive published late."""
        after = pd.Timestamp(BASE_END) + pd.Timedelta(days=1)
        return {k: pd.concat([old[k], new[k].loc[after:]]).sort_index().combine_first(new[k])
                for k in old if k in new}

    allp, rl = join(allp, new_all), join(rl, new_rl)
    first = {p[:-4]: min(fetch.archive_months("spot", p)) for p in spot_pairs if p[:-4].endswith("B")}
    unknown = sorted(b for b, m in first.items() if m >= pd.Period("2026-06", "M") and b not in data.TOKENIZED_STOCKS)
    print("B-suffixed listings not in data.TOKENIZED_STOCKS (check whether they are stocks):", unknown)

print("data ends", allp["price"].index.max().date())
# -

# ## The frozen book (v1)
#
# The research coin list and its 55 perps, the three frozen sleeves and
# notebook 06's combination rules, on archive data.

# +
uni_v1 = data.build_universe(rl, top_n=100, min_adv_usd=1_000_000)
v1 = pd.DataFrame({
    "seasonality": strategies.seasonal_momentum_sleeve(rl["price"], rl["returns"], uni_v1),
    "orderflow": strategies.orderflow_sleeve(rl["taker_imbalance"], rl["returns"], uni_v1),
    "carry": strategies.carry_sleeve(rl["funding"].reindex(columns=uni_v1.columns), rl["returns"], uni_v1),
}).dropna()
v1_wf, _ = robustness.walk_forward_weights(v1, train_days=756, step_days=63, min_train=252)
v1_ew = v1.mean(axis=1)

# notebook 08 ran the first 82 days on the API data; the archive rebuild should agree
nb08 = pd.read_parquet(PROC / "forward_book.parquet").loc[V1_START:"2026-09-26"]
agree = pd.concat([v1_wf.loc[V1_START:"2026-09-26"], nb08["walk_forward"]], axis=1).dropna()
print(f"v1 walk-forward vs notebook 08 over its 82 days: largest daily gap {agree.diff(axis=1).iloc[:, 1].abs().max():.2e}, "
      f"total return {(1 + agree.iloc[:, 0]).prod() - 1:+.4f} against {(1 + agree.iloc[:, 1]).prod() - 1:+.4f}")
# -

# ## v2

uni_v2 = data.build_universe(allp, top_n=100, min_adv_usd=1_000_000)
v2 = strategies.v2_sleeves(allp["taker_imbalance"], allp["returns"],
                           allp["perp_returns"].reindex(columns=uni_v2.columns),
                           allp["funding"].reindex(columns=uni_v2.columns), uni_v2, spot_fallback=True)
v2_book = strategies.v2_book(v2)
btc = allp["returns"]["BTC"]

# ## The record

# +
records = {"v1, walk-forward": (v1_wf, V1_START), "v1, equal weight": (v1_ew, V1_START),
           "v2 book": (v2_book, V2_START), "v2 orderflow": (v2["orderflow"], V2_START),
           "v2 carry": (v2["carry"], V2_START)}


def record(r: pd.Series, start: str) -> dict:
    r = r.loc[start:END].dropna()
    if len(r) < 2:
        return {"from": start, "days": len(r)}
    b = btc.reindex(r.index)
    return {"from": start, "days": len(r), "total_return": (1 + r).prod() - 1, "ann_vol": metrics.ann_vol(r),
            "sharpe": metrics.sharpe(r), "t_stat": metrics.sharpe(r) * np.sqrt(len(r) / 365),
            "max_dd": metrics.max_drawdown(r), "beta_btc": np.cov(r, b)[0, 1] / b.var(),
            "btc_total_return": (1 + b).prod() - 1}


log = pd.DataFrame({k: record(s, start) for k, (s, start) in records.items()}).T
log
# -

# Month by month (total return in each calendar month):

daily = pd.DataFrame({k: s.loc[start:END] for k, (s, start) in records.items()} | {"BTC": btc.loc[V1_START:END]})
monthly = (1 + daily).resample("ME").prod(min_count=1) - 1
monthly.index = monthly.index.strftime("%Y-%m")
monthly.round(4)

# Save the daily record (not committed; the executed notebook is the record).

daily.to_parquet(PROC / "forward_log.parquet")
print("saved forward_log.parquet through", END)
