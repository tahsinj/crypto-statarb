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

# # 02: Seasonality and time of day
#
# The idea, written down before running anything: crypto trades around the
# clock, but who is trading changes with the clock. Institutions are more
# active on weekdays and in US hours, retail at night and at weekends. If the
# institutional flow carries more information, weekday and US-hours moves
# should trend more, and the retail-heavy buckets should revert or earn a
# different premium. Turn-of-month rebalancing is a related flow with a fixed
# schedule.
#
# Probes (every config is logged to the trial registry). A survivor has to
# beat the untilted momentum sleeve, and zero, on both dev and gate, and a
# neighbouring parameter value has to be positive on both windows as well.
#   P1  weekday/weekend exposure tilts on the momentum sleeve (daily, 20 bps)
#   P2  long-flat equal-weight market by calendar bucket (daily, 20 bps)
#   P3  US hours vs off hours on the hourly panel (7 bps): descriptive first,
#       with a timing test only if the gap is material
#   P4  turn-of-month tilt on the momentum sleeve (daily, 20 bps)
#
# Everything stops at 2025-06-30; the lockbox is only read in notebook 06.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, metrics, signals, strategies, trials

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
REGISTRY = PROC / "trial_registry.csv"

GATE_END = "2025-06-30"
DEV = slice("2020-01-01", "2024-07-31")
GATE = slice("2024-08-01", GATE_END)
HOURLY_PPY = 24 * 365

price = pd.read_parquet(PROC / "price.parquet").loc[:GATE_END]
returns = pd.read_parquet(PROC / "returns.parquet").loc[:GATE_END]
universe = pd.read_parquet(PROC / "universe.parquet").loc[:GATE_END]
mkt = pd.read_parquet(PROC / "benchmarks.parquet").loc[:GATE_END, "MKT"]
price_1h = pd.read_parquet(PROC / "price_1h.parquet").loc[:GATE_END]


def log(config, ret, note="", ppy=365):
    """Log one probe config; returns (dev_sharpe, gate_sharpe)."""
    d = metrics.sharpe(ret.loc[DEV], periods_per_year=ppy)
    g = metrics.sharpe(ret.loc[GATE], periods_per_year=ppy)
    trials.log_trial(REGISTRY, "seasonality", config, d, g, note=note)
    return d, g


# ## P1: weekday/weekend tilts on the momentum sleeve
# The sleeve is the notebook 01 baseline (lookback 30, vol target 15%,
# 20 bps). A tilt multiplies the day's exposure. The day of the week is known
# in advance, so the overlay has no look-ahead.

mom = strategies.momentum_sleeve(price, returns, universe,
                                 lookback=30, target_vol=0.15, cost_bps=20)
p1 = {}
for we, wd in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 0.5), (1.0, 0.0)]:
    r = signals.seasonal_scale(mom, weekend=we, weekday=wd)
    d, g = log({"probe": "P1", "weekend": we, "weekday": wd}, r,
               note="momentum sleeve calendar tilt")
    p1[(we, wd)] = (d, g)
pd.DataFrame(p1, index=["dev", "gate"]).T.round(2)

# ## P2: long-flat market timing by bucket
# Hold the equal-weight market only inside a bucket, paying 20 bps on each
# entry and each exit.

p2 = {}
for bucket in ["weekday", "weekend", "turn_of_month"]:
    m = signals.calendar_mask(mkt.index, bucket).astype(float)
    held = m.shift(1).fillna(0.0)               # decide at close t-1, hold day t
    turno = held.diff().abs().fillna(0.0)
    r = (mkt * held - turno * 20 / 1e4).rename(bucket)
    d, g = log({"probe": "P2", "bucket": bucket}, r, note="long-flat EW market")
    p2[bucket] = (d, g)
buyhold_dev = metrics.sharpe(mkt.loc[DEV])
buyhold_gate = metrics.sharpe(mkt.loc[GATE])
print(f"buy-and-hold market: dev {buyhold_dev:+.2f}  gate {buyhold_gate:+.2f}")
pd.DataFrame(p2, index=["dev", "gate"]).T.round(2)

# ## P3: US hours vs off hours (hourly panel, descriptive first)

rets_1h = price_1h.pct_change()
ew_1h = rets_1h.mean(axis=1)
split = {}
for bucket in ["us_hours", "off_hours"]:
    m = signals.calendar_mask(ew_1h.index, bucket)
    r = ew_1h[m]
    split[bucket] = {
        "ann_ret": r.mean() * HOURLY_PPY,
        "sharpe_dev": metrics.sharpe(r.loc[DEV], periods_per_year=HOURLY_PPY),
        "sharpe_gate": metrics.sharpe(r.loc[GATE], periods_per_year=HOURLY_PPY),
        "share_of_hours": m.mean(),
    }
split_df = pd.DataFrame(split).T.round(3)
split_df

# A timing test runs only if the gap is material (more than 5% a year in
# annualised return): hold the equal-weight market in the stronger bucket
# only, at 7 bps per switch (limit orders).

gap = abs(split["us_hours"]["ann_ret"] - split["off_hours"]["ann_ret"])
if gap > 0.05:
    rich = max(split, key=lambda b: split[b]["ann_ret"])
    m = signals.calendar_mask(ew_1h.index, rich).astype(float)
    held = m.shift(1).fillna(0.0)
    turno = held.diff().abs().fillna(0.0)
    r = (ew_1h * held - turno * 7 / 1e4).rename(f"hold_{rich}")
    d, g = log({"probe": "P3", "bucket": rich, "cost_bps": 7}, r,
               note="hourly long-flat timing", ppy=HOURLY_PPY)
    print(f"P3 hold-{rich}: dev {d:+.2f}  gate {g:+.2f}")
else:
    trials.log_trial(REGISTRY, "seasonality", {"probe": "P3", "bucket": "none"},
                     np.nan, np.nan, note=f"split gap {gap:.3f} < 0.05, not tested")
    print(f"P3 skipped: annualised bucket gap only {gap:.1%}")

# ## P4: turn-of-month tilt on momentum (a neighbour of P2's turn-of-month bucket)

for mult in [1.5, 2.0]:
    m = signals.calendar_mask(mom.index, "turn_of_month")
    r = (mom * np.where(m, mult, 1.0)).rename("tom_tilt")
    log({"probe": "P4", "tom_mult": mult}, r, note="turn-of-month tilt on momentum")

# ## Gate check and survivor selection
# A survivor must beat the untilted momentum sleeve (and zero) on both dev
# and gate, and a neighbouring parameter value must also be positive on both,
# so that no result hangs on one lucky number.

reg = trials.load_registry(REGISTRY)
season = reg[reg["family"] == "seasonality"].copy()
season[["dev_sharpe", "gate_sharpe"]] = season[["dev_sharpe", "gate_sharpe"]].astype(float)
base_dev, base_gate = p1[(1.0, 1.0)]
print(f"baseline (untilted momentum): dev {base_dev:+.2f}  gate {base_gate:+.2f}")
season[["config", "dev_sharpe", "gate_sharpe", "note"]]

survivors = season[(season["dev_sharpe"] > max(0, base_dev)) &
                   (season["gate_sharpe"] > max(0, base_gate))]
print(f"{len(survivors)} config(s) beat baseline on both windows")

# At most one sleeve is saved: the survivor with the best gate Sharpe whose
# neighbouring config is also positive on both windows. The neighbour check
# is read off the table above and written out in the conclusion.

# ## Conclusion
#
# One P1 config survives: weekday exposure halved (weekend=1.0, weekday=0.5).
#
# **P1.** The untilted sleeve (weekend=1.0, weekday=1.0) is the notebook 01
# baseline: dev +1.65, gate +0.43. Cutting weekday exposure raises the gate
# Sharpe at each step (0.43 at full weekday size, 1.24 at half, 2.54 at zero).
# Adding weekend exposure raises dev (+1.19, +1.47, +1.65, +1.75 for weekend
# 0, 0.5, 1.0, 1.5) and the gate moves the same way. Two configs beat the
# baseline on both windows:
# - weekend=1.5, weekday=1.0: dev +1.75, gate +0.90
# - weekend=1.0, weekday=0.5: dev +1.79, gate +1.24 (selected)
#
# The selected config's neighbours on the weekday axis are positive on both
# windows: weekday=1.0 (dev +1.65, gate +0.43) and weekday=0.0 (dev +1.34,
# gate +2.54). The gate Sharpe moves the same way along the whole weekday
# axis, so a single lucky parameter is unlikely, although a regime effect in
# an 11-month gate cannot be ruled out.
#
# **P2.** Buy-and-hold of the equal-weight market: dev +1.17, gate +0.02.
# Weekday-only (dev +0.79, gate +0.64) and turn-of-month (dev +0.98,
# gate -1.78) trail the momentum baseline on dev, and weekend-only is flat on
# dev and negative on the gate. Timing the market by calendar bucket does not
# work at 20 bps. No P2 survivor.
#
# **P3.** Nearly all of the equal-weight market's return came outside US
# hours: annualising each bucket's average hourly return gives 115% a year
# off hours against 4% in US hours, far above the 5% trigger. Holding the
# market only off hours, at 7 bps per switch, scores dev +0.59 and gate -0.35,
# so it fails the gate. The split is striking, but it does not survive as a
# timing rule after costs. No P3 survivor.
#
# **P4.** tom_mult=1.5: dev 1.648 (baseline 1.650), gate 0.573. tom_mult=2.0:
# dev 1.619, gate 0.670. Both improve the gate but fall short of the baseline
# on dev. No P4 survivor.
#
# **Decision.** P1 (weekend=1.0, weekday=0.5): dev +1.79, gate +1.24, with
# positive neighbours on both sides of the weekday axis. One reading is that
# weekday price action carries more noise from institutional rebalancing, so
# trend signals work better at weekends; this test cannot separate that story
# from others. `sleeve_seasonality.parquet` is this overlay applied to the
# baseline momentum sleeve (lookback 30, vol target 15%, 20 bps).
#
# **Caveat.** The gate window is only 11 months (Aug 2024 to Jun 2025). Part
# of the gate improvement could be a regime shift in that period rather than
# a lasting effect. Notebook 06 checks it on the lockbox.

best = signals.seasonal_scale(mom, weekend=1.0, weekday=0.5)
best.rename("seasonality").to_frame().to_parquet(PROC / "sleeve_seasonality.parquet")
print("sleeve_seasonality saved:", best.shape, "index max:", best.index.max().date())
