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

# # 05: Fast reversal (hourly)
#
# The idea, written down before running anything: daily cross-sectional
# reversal has a gross edge that costs eat (the notebook 03 baselines). If the
# edge comes from providing liquidity, it should be bigger and faster at
# intraday horizons, and so should the turnover. The rule, fixed in advance:
# a config is only worth tuning if its gross annualised return is at least 3x
# its annualised cost drag. The expectation going in was that costs would
# kill this family; a negative result is still worth recording.
#
# Configs: lookback of 4, 12, 24 or 48 hours, rebalanced every 1 or 6 hours,
# dollar-neutral, 7 bps per rebalance, hourly bars from 2020-06 (the first
# months of the hourly panel have too few names).
#
# Data caveat: the 60-name hourly panel was picked by ADV at fetch time, which
# tilts the early years toward coins that survived. To limit this, a name is
# only tradable in an hour if it was in the point-in-time daily top-100
# universe on that date, so coins that were small or dead at the time cannot
# enter early books. Everything stops at 2025-06-30.

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
DEV = slice("2020-06-01", "2024-07-31")   # hourly panel too thin before mid-2020
GATE = slice("2024-08-01", GATE_END)
PPY = 24 * 365

price = pd.read_parquet(PROC / "price_1h.parquet").loc["2020-06-01":GATE_END]
returns = price.pct_change(fill_method=None)
uni_d = pd.read_parquet(PROC / "universe.parquet").loc[:GATE_END]

# Point-in-time tradability by the hour: a name is tradable in hour t if it
# was in the daily top-100 universe on t's date (carried forward within the day).
uni_h = uni_d.reindex(columns=price.columns).reindex(price.index, method="ffill").fillna(False)
print("hourly tradable breadth by year:")
print(uni_h.sum(axis=1).groupby(uni_h.index.year).mean().round(0))


def run_fast(lookback, rebal, note=""):
    """XS reversal at `lookback` hours, rebalanced every `rebal` hours, 7 bps.

    Logs net dev/gate to the registry and returns (net, gross, avg_turnover).
    """
    sig = -signals.cross_sectional_zscore(signals.trailing_return(price, lookback), uni_h)
    w = signals.signal_to_weights(sig, uni_h, long_short=True)
    if rebal > 1:  # hold weights between rebalances (stale in-between bars)
        mask = pd.Series(np.arange(len(w)) % rebal == 0, index=w.index)
        w = w.where(mask).ffill()
    res = backtest.run(w, returns, cost_bps=7)
    net, gross = res.net_returns, res.gross_returns
    to = res.turnover.mean()
    d = metrics.sharpe(net.loc[DEV], periods_per_year=PPY)
    g = metrics.sharpe(net.loc[GATE], periods_per_year=PPY)
    trials.log_trial(REGISTRY, "fastrev", {"lookback_h": lookback, "rebal_h": rebal},
                     d, g, note=note)
    return net, gross, to, d, g


# ## The grid, with the 3x gross-to-cost screen for each config

rows = {}
for lb in [4, 12, 24, 48]:
    for rb in [1, 6]:
        net, gross, to, d, g = run_fast(lb, rb, note="hourly XS reversal")
        gross_dev = gross.loc[DEV]
        gross_ann = gross_dev.mean() * PPY
        drag_ann = to * 7 / 1e4 * PPY          # per-bar cost drag, annualized
        rows[(lb, rb)] = {
            "gross_sharpe_dev": metrics.sharpe(gross_dev, periods_per_year=PPY),
            "gross_ann": gross_ann,
            "cost_drag_ann": drag_ann,
            "gross_over_drag": gross_ann / drag_ann if drag_ann > 0 else np.nan,
            "net_dev": d, "net_gate": g,
        }
grid = pd.DataFrame(rows).T.round(2)
grid["passes_screen"] = (grid["gross_sharpe_dev"] > 0) & (grid["gross_over_drag"] >= 3)
print(grid.to_string())

# ## Gate check
reg = trials.load_registry(REGISTRY)
fr = reg[reg["family"] == "fastrev"].copy()
fr[["dev_sharpe", "gate_sharpe"]] = fr[["dev_sharpe", "gate_sharpe"]].astype(float)
print(fr[["config", "dev_sharpe", "gate_sharpe"]])

# ## Conclusion
#
# Survivor rule: pass the screen, positive net Sharpe on dev and gate, and the
# adjacent lookback at the same rebalance frequency positive on both. A
# survivor would be summed to daily returns and saved as sleeve_fastrev.
#
# **Breadth.** Tradable hourly names per year (the daily universe carried into
# each hour): about 18 in 2020, 28 in 2021, 28 in 2022, 32 in 2023, 40 in 2024
# and 47 in 2025. Because the hourly panel was picked by ADV at fetch time,
# 2020-2021 lean toward survivors. The daily-universe mask keeps dead and
# small coins out of the books, but early breadth is still thin.
#
# **Grid** (dev window 2020-06-01 to 2024-07-31, 8,760 hours a year; the
# cost column takes its turnover over dev and gate together, which moves it
# by well under 1%: 461% against 463% on dev alone for the 4-hour config):
#
#   lb=4,  rb=1: gross Sharpe +4.83, gross +272%/yr, drag 461%/yr, ratio 0.59, fails
#   lb=4,  rb=6: gross Sharpe +0.86, gross +44%/yr,  drag 146%/yr, ratio 0.30, fails
#   lb=12, rb=1: gross Sharpe +1.80, gross +103%/yr, drag 267%/yr, ratio 0.39, fails
#   lb=12, rb=6: gross Sharpe -0.91, gross -50%/yr,  drag 103%/yr, ratio -0.48, fails
#   lb=24, rb=1: gross Sharpe +0.63, gross +37%/yr,  drag 187%/yr, ratio 0.20, fails
#   lb=24, rb=6: gross Sharpe -0.84, gross -48%/yr,  drag 73%/yr,  ratio -0.66, fails
#   lb=48, rb=1: gross Sharpe +0.01, gross +1%/yr,   drag 130%/yr, ratio 0.01, fails
#   lb=48, rb=6: gross Sharpe -1.00, gross -57%/yr,  drag 51%/yr,  ratio -1.12, fails
#
# All eight configs fail the 3x screen. Net dev Sharpe is negative for every
# config (-1.90 to -3.41) and the gate is worse (-2.28 to -7.79). No survivor.
#
# **Reading.** The 4-hour lookback does have a real gross reversal signal
# (gross Sharpe +4.83), so the liquidity-provision idea has something to it.
# But rebalancing every hour at 7 bps costs 461% a year, about 1.7 times the
# 272% gross return, and a ratio of 0.59 is far from 3. Rebalancing every six
# hours cuts the drag by about two thirds (461% to 146%), but the gross return
# falls faster (272% to 44%, down 84%), so the ratio gets worse (0.30).
# Longer lookbacks shrink both the signal and the drag without getting out of
# the problem: at these frequencies even limit-order costs are too high.
#
# **Survivorship, again.** Survivorship bias does not explain the loss. In the
# gate window (2024-08 to 2025-06), where breadth is about 44 names and the
# panel is closest to representative, net Sharpe is -2.28 to -7.79, worse
# than on dev.
#
# **No survivor, so sleeve_fastrev.parquet is not written.** The family is
# rejected on costs.
