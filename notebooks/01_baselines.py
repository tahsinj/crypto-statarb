# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 01: Baseline strategies
#
# Two strategies, run first as baselines that every later
# idea has to beat: time-series momentum (sign of the 30-day return,
# vol-targeted to 15%) and correlation pairs. Their settings were fixed before
# the gate window, so the gate shows how they hold up on data they were not
# chosen on. They come from the project's first version (on CoinGecko data,
# before this protocol), which picked them from small grids on 2020-2022 data
# and had also run both baselines as twsq alphas over most of the lockbox
# (README). Everything here stops at 2025-06-30; this protocol first reads the
# lockbox in notebook 06.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import metrics, strategies, trials

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
REGISTRY = PROC / "trial_registry.csv"

GATE_END = "2025-06-30"          # nothing below reads past this date
DEV = slice("2020-01-01", "2024-07-31")
GATE = slice("2024-08-01", GATE_END)

price = pd.read_parquet(PROC / "price.parquet").loc[:GATE_END]
returns = pd.read_parquet(PROC / "returns.parquet").loc[:GATE_END]
universe = pd.read_parquet(PROC / "universe.parquet").loc[:GATE_END]

# ## The two baselines

mom = strategies.momentum_sleeve(price, returns, universe, lookback=30,
                                 target_vol=0.15, cost_bps=20).loc["2020-01-01":]
rev = strategies.reversal_sleeve(price, returns, universe, cost_bps=7).loc["2020-01-01":]

trials.log_trial(REGISTRY, "incumbent_momentum", {"lookback": 30, "target_vol": 0.15},
                 metrics.sharpe(mom.loc[DEV]), metrics.sharpe(mom.loc[GATE]),
                 note="baseline, parameters from the first version")
trials.log_trial(REGISTRY, "incumbent_pairs", {"entry": 2.0, "zwin": 30, "top_k": 20},
                 metrics.sharpe(rev.loc[DEV]), metrics.sharpe(rev.loc[GATE]),
                 note="baseline, parameters from the first version")
# (the registry keeps the note these two rows were first logged with, on 2026-07-06)

# ## Per-window table

table = pd.DataFrame({
    ("momentum", "dev"): metrics.summary(mom.loc[DEV]),
    ("momentum", "gate"): metrics.summary(mom.loc[GATE]),
    ("pairs", "dev"): metrics.summary(rev.loc[DEV]),
    ("pairs", "gate"): metrics.summary(rev.loc[GATE]),
})
table.round(3)

# ## Rolling 180-day Sharpe

fig, ax = plt.subplots(figsize=(10, 4))
for s, lbl in [(mom, "momentum"), (rev, "pairs")]:
    roll = s.rolling(180).apply(lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan)
    roll.plot(ax=ax, label=lbl)
ax.axvline(pd.Timestamp("2024-08-01"), ls="--", c="k", lw=1)
ax.annotate("gate starts", xy=(pd.Timestamp("2024-08-01"), ax.get_ylim()[1] * 0.9))
ax.axhline(0, c="grey", lw=0.5)
ax.legend(); ax.set_title("Rolling 180-day Sharpe of the baseline sleeves")
plt.tight_layout()

# ## Save for the later notebooks and the report

mom.rename("momentum").to_frame().to_parquet(PROC / "sleeve_baseline_momentum.parquet")
rev.rename("reversal").to_frame().to_parquet(PROC / "sleeve_baseline_reversal.parquet")
print("dev momentum sharpe:", round(metrics.sharpe(mom.loc[DEV]), 2),
      "| gate:", round(metrics.sharpe(mom.loc[GATE]), 2))
