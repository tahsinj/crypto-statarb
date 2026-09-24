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

# # 04: Funding carry and crowding
#
# The idea, written down before running anything: perp funding is a price paid
# for positioning. Leveraged longs pay it, so persistently high funding is
# (a) carry that the short side collects and (b) a sign of crowding, since
# crowded longs tend to unwind. Three probes:
#   P1  carry: short high-funding names and long low-funding names on k-day
#       smoothed funding, with the funding payments added to the price P&L.
#   P2  crowding reversal: the plain 5-day reversal, traded only in names
#       with an extreme funding z-score (crowded books should unwind harder).
#   P3  funding momentum: the 7-day change in funding as a cross-sectional
#       signal, in both directions (does rising funding keep rising, or
#       mark a top?), at two smoothings.
# All probes rebalance daily, are dollar-neutral and pay 7 bps, on the
# universe restricted to names with funding data. Everything stops at
# 2025-06-30.

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, metrics, signals, trials

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
REGISTRY = PROC / "trial_registry.csv"

GATE_END = "2025-06-30"
DEV = slice("2020-01-01", "2024-07-31")
GATE = slice("2024-08-01", GATE_END)

price = pd.read_parquet(PROC / "price.parquet").loc[:GATE_END]
returns = pd.read_parquet(PROC / "returns.parquet").loc[:GATE_END]
universe = pd.read_parquet(PROC / "universe.parquet").loc[:GATE_END]
funding = pd.read_parquet(PROC / "funding.parquet").loc[:GATE_END]

# Tradable set: the point-in-time universe, limited to names with funding data.
funding = funding.reindex(columns=universe.columns)
fuv = universe & funding.notna()
print("tradable breadth by year (universe ∩ funding):")
print(fuv.sum(axis=1).groupby(fuv.index.year).mean().round(0))


def run_carry(sig, config, note=""):
    """L/S backtest at 7 bps with explicit funding P&L; logs, returns net."""
    w = signals.signal_to_weights(sig, fuv, long_short=True)
    res = backtest.run(w, returns, cost_bps=7)
    held = w.shift(1)
    fund_pnl = -(held * funding).sum(axis=1)  # longs pay positive funding
    net = (res.net_returns + fund_pnl).loc["2020-01-01":]
    d, g = metrics.sharpe(net.loc[DEV]), metrics.sharpe(net.loc[GATE])
    trials.log_trial(REGISTRY, "carry", config, d, g, note=note)
    return net, d, g


# ## P1: carry (short high funding, long low funding)

p1 = {}
for k in [1, 7, 30]:
    smooth = funding.rolling(k).mean() if k > 1 else funding
    sig = -signals.cross_sectional_zscore(smooth, fuv)
    net, d, g = run_carry(sig, {"probe": "P1", "smooth": k},
                          note="carry: short high funding")
    p1[k] = (net, d, g)
pd.DataFrame({k: v[1:] for k, v in p1.items()}, index=["dev", "gate"]).T.round(2)

# ## P2: reversal only in crowded (extreme-funding) names
# Same construction as the plain 5-day reversal (logged in the orderflow
# family as its baseline, so not logged again here), masked to crowded names.

fz = signals.cross_sectional_zscore(funding.rolling(7).mean(), fuv)
p2 = {}
for thresh in [1.0, 1.5]:
    tr = signals.trailing_return(price, 5)
    sig = (-signals.cross_sectional_zscore(tr, fuv)).where(fz.abs() > thresh)
    net, d, g = run_carry(sig, {"probe": "P2", "lookback": 5, "fz_thresh": thresh},
                          note="reversal in crowded names only")
    p2[thresh] = (d, g)
pd.DataFrame(p2, index=["dev", "gate"]).T.round(2)

# ## P3: funding momentum (7-day change in funding, both signs)

p3 = {}
for k in [7, 30]:
    base_f = funding.rolling(k).mean()
    chg = base_f - base_f.shift(7)
    for sign, lbl in [(1, "follow"), (-1, "fade")]:
        sig = sign * signals.cross_sectional_zscore(chg, fuv)
        net, d, g = run_carry(sig, {"probe": "P3", "smooth": k, "dir": lbl},
                              note="funding change as XS signal")
        p3[(k, lbl)] = (d, g)
pd.DataFrame(p3, index=["dev", "gate"]).T.round(2)

# ## Gate check
reg = trials.load_registry(REGISTRY)
ca = reg[reg["family"] == "carry"].copy()
ca[["dev_sharpe", "gate_sharpe"]] = ca[["dev_sharpe", "gate_sharpe"]].astype(float)
ca[["config", "dev_sharpe", "gate_sharpe", "note"]]

# ## Conclusion
#
# **Breadth.** Names in both the universe and the funding panel average about
# 13 in 2020, 22 in 2021, 23 in 2022, 29 in 2023, 36 in 2024 and 44 in 2025.
# Early dev results rest on a thin cross-section; the gate window is the more
# reliable of the two. The funding panel only covers the 60 names with the
# most volume at fetch time (July 2026), so it leans toward coins that
# survived and grew; the point-in-time universe decides which of them can
# trade on a given day, but coins that faded earlier are simply not there.
#
# **P1.** All three smoothings are positive on both windows:
#   k=1:  dev +0.05, gate +1.75 (neighbour k=7 positive on both)
#   k=7:  dev +1.01, gate +2.19 (neighbours k=1 and k=30 positive on both)
#   k=30: dev +0.29, gate +0.72 (neighbour k=7 positive on both)
# All three pass. The result peaks at k=7: a week of smoothing removes the
# day-to-day noise in funding without lagging the funding regime too much.
# The short leg collects funding directly, on top of the price P&L.
#
# **P2.** Restricting the 5-day reversal to crowded names does not help:
#   thresh=1.0: dev -0.64, gate -1.17
#   thresh=1.5: dev -0.54, gate -1.43
# Plain reversal already loses at 7 bps (notebook 03 baselines) and the mask
# does not change that. No P2 survivor.
#
# **P3.** Following rising funding loses badly:
#   k=7 follow:  dev -1.11, gate -3.17
#   k=30 follow: dev -1.02, gate -2.06
# Fading it is positive on both windows:
#   k=7 fade:  dev +0.43, gate +2.50 (neighbour k=30 fade positive on both)
#   k=30 fade: dev +0.59, gate +1.59 (neighbour k=7 fade positive on both)
# Both fade configs pass. When funding rises quickly, long positioning is
# building, and fading it has paid. As in notebook 03, follow and fade are
# mirror images of one signal rather than two separate findings.
#
# **Decision: P1 with k=7** (dev +1.01, gate +2.19). It is the most direct
# carry trade and has the best dev Sharpe of the survivors. P3 fade k=7 has a
# higher gate Sharpe (+2.50) but a much weaker dev (+0.43), and it is a
# crowding trade rather than carry. It might work better as a filter on the
# carry sleeve, which was not tested here. Choosing P1 keeps the sleeve what
# its name says it is.
#
# Notebook 07 splits the sleeve's P&L into the funding and price legs:
# funding is about 63% of it on dev and 39% on the gate. On the lockbox the
# sleeve scored +0.88 (notebook 06).
#
# `sleeve_carry.parquet` is written below (column "carry").

# Recompute the selected config (P1, k=7) and save the sleeve.
smooth_final = funding.rolling(7).mean()
sig_final = -signals.cross_sectional_zscore(smooth_final, fuv)
w_final = signals.signal_to_weights(sig_final, fuv, long_short=True)
res_final = backtest.run(w_final, returns, cost_bps=7)
held_final = w_final.shift(1)
fund_pnl_final = -(held_final * funding).sum(axis=1)
best = (res_final.net_returns + fund_pnl_final).loc["2020-01-01":]
print(f"Survivor P1 k=7: dev={metrics.sharpe(best.loc[DEV]):+.3f}  gate={metrics.sharpe(best.loc[GATE]):+.3f}")
best.rename("carry").to_frame().to_parquet(PROC / "sleeve_carry.parquet")
print("sleeve_carry saved:", best.shape, "index max:", best.index.max().date())
