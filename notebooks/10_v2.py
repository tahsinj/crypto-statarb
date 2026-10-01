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

# # 10: Version 2, registered before it was tested
#
# This notebook was written on 2026-09-27 and committed without outputs
# before it was first run, so the rules below were fixed before any v2
# number existed. The commit after that one adds the outputs.
#
# ## What changes from the frozen book
#
# Each change answers a problem found before the lockbox or in the data
# audit. None was picked by comparing results.
#
# 1. Seasonality is dropped. Charged for the trades that resize it, it fails
#    notebook 02's own rule on dev and gate (notebook 07).
# 2. Carry weights coins by rank instead of z-score. With z-scores one coin
#    with extreme funding can take a whole side of the sleeve, as SOL did
#    during the FTX collapse in November 2022, on dev. With ranks and 40 or
#    more coins, no coin gets more than about 5% of the sleeve.
# 3. Carry is measured on perp prices. The trade holds perps, and in a crash
#    a perp can trade well below spot.
# 4. The universe comes from every Binance pair, delisted ones included, with
#    pegged assets and tokenized stocks taken out and JUP and SYRUP put back,
#    and perp data kept only on days the contract traded close to spot
#    (notebook 09).
# 5. The two sleeves get equal weights, rebalanced daily. Equal weight beat
#    the walk-forward weights on dev (0.46 against 0.40 over the same dates)
#    and on the gate (2.90 against 2.52).
#
# Orderflow keeps its rule, and costs stay at 7 bps per dollar traded. The
# code is `strategies.v2_sleeves` and `strategies.v2_book`.
#
# ## How it will be judged
#
# - Here, on dev and gate only, as a check that v2 behaves sensibly on the
#   windows the research used. The data is cut at the end of the gate before
#   anything is computed. v2 is never run on the lockbox or on the July to
#   September forward window, since both have been seen.
# - The test is every day from 2026-09-28, the first full day after this
#   commit. Notebook 12 adds each new month, with the frozen book alongside.
#   The first reading is after 12 months, on 2027-09-28. At a Sharpe near 1.5
#   it takes about two years to reach a t-stat of 2, so the record should
#   keep running after that.
# - The registry gets three rows: the two sleeves and the book.

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import metrics, strategies, trials

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
ALL = PROC / "all_pairs"
REGISTRY = PROC / "trial_registry.csv"

GATE_END = "2025-06-30"
DEV = slice("2020-01-01", "2024-07-31")
GATE = slice("2024-08-01", GATE_END)
# -

# Every panel is cut at the end of the gate before any signal is computed.


# +
def load(name):
    return pd.read_parquet(ALL / f"{name}.parquet").loc[:GATE_END]


returns, taker, universe = load("returns"), load("taker_imbalance"), load("universe")
perp_returns = load("perp_returns").reindex(columns=universe.columns)
funding = load("funding").reindex(columns=universe.columns)
bench = load("benchmarks")
assert max(x.index.max() for x in [returns, taker, universe, perp_returns, funding]) <= pd.Timestamp(GATE_END)
print("data ends", returns.index.max().date())
# -

# ## v2 on dev and gate

# +
sleeves = strategies.v2_sleeves(taker, returns, perp_returns, funding, universe)
book = strategies.v2_book(sleeves)
series = {"v2 book": book, "orderflow": sleeves["orderflow"], "carry (rank, perp prices)": sleeves["carry"]}


def stats(r):
    r = r.dropna()
    ab = metrics.alpha_beta(r, bench["BTC"], bench["MKT"], names=["BTC", "MKT"])
    return {"sharpe": metrics.sharpe(r), "ann_return": metrics.ann_return(r), "ann_vol": metrics.ann_vol(r),
            "max_dd": metrics.max_drawdown(r), "beta_btc": ab["beta_BTC"], "alpha": ab["alpha_ann"],
            "alpha_t": ab["alpha_tstat"]}


table = pd.DataFrame({(name, w): stats(s.loc[sl]) for name, s in series.items()
                      for w, sl in [("dev", DEV), ("gate", GATE)]}).T
table.round(3)
# -

# For reference, the frozen book and its sleeves on the same windows, from
# the registry (research universe, spot prices, z-score carry):

reg = trials.load_registry(REGISTRY)
ref = reg[reg["family"].isin(["orderflow", "carry", "portfolio"])]
ref = ref[ref["config"].isin(['{"dir": "follow", "probe": "P3", "smooth": 10}', '{"probe": "P1", "smooth": 7}',
                              '{"min_train": 252, "rule": "walk_forward_mv", "step_days": 63, "train_days": 756}',
                              '{"rule": "equal_weight"}'])]
ref[["family", "config", "dev_sharpe", "gate_sharpe"]].round(3)

# ## Does rank weighting do what it is meant to?
#
# The largest single-coin weight in Carry each day, as a share of the
# sleeve's gross, under both rules on the same data:

# +
top = {w: strategies.carry_weights(funding, universe, weighting=w).abs().max(axis=1)
       for w in ["zscore", "rank"]}
fuv_size = (universe & funding.notna()).sum(axis=1)
conc = pd.DataFrame({(w, name): {"median": t.loc[sl].median(), "max": t.loc[sl].max(),
                                 "days above 0.25": int((t.loc[sl] > 0.25).sum())}
                     for w, t in top.items() for name, sl in [("dev", DEV), ("gate", GATE)]}).T
print(f"coins Carry can trade, median over dev: {int(fuv_size.loc[DEV].median())}, gate: {int(fuv_size.loc[GATE].median())}")
conc.round(3)
# -

# ## Log the three rows

# +
note = "v2, registered 2026-09-27 before testing"
rows = [
    ({"sleeve": "orderflow", "smooth": 10, "universe": "all_pairs"}, sleeves["orderflow"]),
    ({"sleeve": "carry", "smooth": 7, "weighting": "rank", "prices": "perp", "universe": "all_pairs"},
     sleeves["carry"]),
    ({"rule": "equal_weight", "sleeves": ["carry", "orderflow"]}, book),
]
for cfg, s in rows:
    added = trials.log_trial(REGISTRY, "v2", cfg, metrics.sharpe(s.loc[DEV].dropna()),
                             metrics.sharpe(s.loc[GATE].dropna()), note=note)
    print("logged" if added else "already in the registry", cfg)
reg = trials.load_registry(REGISTRY)
print(f"registry: {len(reg)} rows, {int((reg['family'] == 'v2').sum())} of them v2")
# -

# Save for the report.

pd.concat([sleeves, book], axis=1).to_parquet(PROC / "v2_dev_gate.parquet")

# ## After the first run
#
# Added with the outputs. v2 has a Sharpe of 1.39 on dev and 1.12 on the
# gate, and almost all of it comes from Carry: rank-weighted, on perp prices
# and across every pair, Carry has 1.76 on dev and 2.41 on the gate, at
# about half the frozen Carry's volatility (15% against 31%). Rank weighting
# does what it was meant to: the largest single-coin weight has a median of
# about 2% of the sleeve, and it only went above a quarter in the first
# weeks of January 2020, when a handful of coins had funding data.
#
# Orderflow is a different story. On every pair it has a Sharpe of 0.44 on
# dev and -0.78 on the gate, against 0.89 and 2.12 on the research coin
# list, so by notebook 03's own rule it would not have been kept. It stays
# in v2, because v2's rules were fixed before this run. Notebook 12 reports
# each sleeve as well as the book, so Carry can be followed on its own, and
# notebook 11 shows what the wider universe does to the frozen book.
#
# **Added later, the registered text above left as it was.** "None was picked
# by comparing results" means none was picked on v2's own results: change 5
# was chosen on the frozen book's dev and gate results, and rank weights also
# answer the DEXE case of the forward window (notebook 08). "Never run on the
# lockbox or the forward window" holds for every result: notebook 12 uses
# those months only as history for v2's first positions and drops v2's P&L on
# them before recording anything. Two fixes came after this run, both measured
# in notebook 11. A Carry position held into a day the rule in change 4 drops
# its perp's data now earns what its contract did: its own move and funding,
# or nothing once it has stopped trading. The rule still decides which perps
# Carry can pick. And two holes in the archive's 2022 files are filled. Both
# were made on 2026-09-29 (UTC; an earlier version of the first, on
# 2026-09-28, used the coin's spot move), before any test result was computed.
# The test start then moved from 2026-09-28 to 2026-10-01, after the
# repository was first pushed to GitHub (2026-09-29, 15:17 UTC), and to
# 2026-10-02 after the history was rewritten for wording on 2026-10-01 and
# force-pushed, so that every test day comes after a public record of these
# rules and of both fixes; the first reading moves to 2027-10-02. This
# notebook is not meant to be rerun: on today's data it would write different
# numbers over its saved file, which notebook 11 checks against the registry.
