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

# # 03: Order flow
#
# The idea, written down before running anything: reversal should be
# strongest where a price move carries no information, which is the
# uninformed-flow argument. Binance klines include the
# taker-buy share of volume (about 0.5 when aggressive buyers and sellers
# balance), so flow can be measured directly. Three ways to use it:
#   P1  reversal on quiet names only: fade moves in names whose volume
#       z-score is below its trailing norm, and skip busy (likely informed)
#       moves.
#   P2  reversal where flow disagreed with the move: fade a rise that came
#       with a buy share at or below 0.5 (and the reverse), and skip moves
#       the flow confirmed.
#   P3  the imbalance itself as a cross-sectional signal, smoothed over k
#       days, in both directions (follow the crowd or fade it).
# All probes rebalance daily, are dollar-neutral, pay limit costs (7 bps) and
# trade the point-in-time top-100 universe. Everything stops at 2025-06-30.

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
dvol = pd.read_parquet(PROC / "dollar_volume.parquet").loc[:GATE_END]
imb = pd.read_parquet(PROC / "taker_imbalance.parquet").loc[:GATE_END]


def run_ls(sig, config, note=""):
    """Dollar-neutral L/S backtest at 7 bps; logs and returns the net series."""
    w = signals.signal_to_weights(sig, universe, long_short=True)
    net = backtest.run(w, returns, cost_bps=7).net_returns.loc["2020-01-01":]
    d, g = metrics.sharpe(net.loc[DEV]), metrics.sharpe(net.loc[GATE])
    trials.log_trial(REGISTRY, "orderflow", config, d, g, note=note)
    return net, d, g


# ## P1: reversal conditioned on low activity
# Baselines first: plain cross-sectional reversal at three lookbacks.

base = {}
for lb in [3, 5, 10]:
    sig = -signals.cross_sectional_zscore(signals.trailing_return(price, lb), universe)
    net, d, g = run_ls(sig, {"probe": "P1", "lookback": lb, "cond": "none"},
                       note="unconditioned reversal baseline")
    base[lb] = (net, d, g)
pd.DataFrame({lb: v[1:] for lb, v in base.items()}, index=["dev", "gate"]).T.round(2)

# Conditioned: trade the same signal only in names whose volume z-score (day
# t's volume against a mean and std through t-1) is below the threshold.

volz = signals.rolling_zscore(dvol, window=60)
p1 = {}
for lb in [3, 5, 10]:
    for thresh in [0.0, -0.5]:
        sig = -signals.cross_sectional_zscore(signals.trailing_return(price, lb), universe)
        sig = sig.where(volz < thresh)
        net, d, g = run_ls(sig, {"probe": "P1", "lookback": lb, "cond": "low_vol",
                                 "thresh": thresh},
                           note="reversal in quiet names only")
        p1[(lb, thresh)] = (d, g)
pd.DataFrame(p1, index=["dev", "gate"]).T.round(2)

# ## P2: reversal conditioned on unconfirmed moves
# Fade a trailing move only where taker flow disagreed with it: the sign of
# the lb-day return differs from the sign of (mean imbalance - 0.5) over the
# same window. Both are known at the close of t and traded from t+1 through
# the lag in backtest.run, so there is no look-ahead.

imb_c = imb - 0.5
p2 = {}
for lb in [3, 5]:
    tr = signals.trailing_return(price, lb)
    flow = imb_c.rolling(lb).mean()
    unconfirmed = np.sign(tr) != np.sign(flow)
    sig = (-signals.cross_sectional_zscore(tr, universe)).where(unconfirmed)
    net, d, g = run_ls(sig, {"probe": "P2", "lookback": lb, "cond": "unconfirmed"},
                       note="fade flow-unconfirmed moves")
    p2[lb] = (d, g)
pd.DataFrame(p2, index=["dev", "gate"]).T.round(2)

# ## P3: the imbalance level as a signal (both signs)

p3 = {}
for k in [1, 5, 10]:
    for sign, lbl in [(1, "follow"), (-1, "fade")]:
        raw = imb_c.rolling(k).mean() if k > 1 else imb_c
        sig = sign * signals.cross_sectional_zscore(raw, universe)
        net, d, g = run_ls(sig, {"probe": "P3", "smooth": k, "dir": lbl},
                           note="taker imbalance as XS signal")
        p3[(k, lbl)] = (d, g)
pd.DataFrame(p3, index=["dev", "gate"]).T.round(2)

# ## Gate check and survivor selection
# P1 and P2 must beat zero and their unconditioned baseline (same lookback)
# on both dev and gate. P3 must beat zero on both windows, and the adjacent
# smoothing in the same direction must also be positive on both.

reg = trials.load_registry(REGISTRY)
of = reg[reg["family"] == "orderflow"].copy()
of[["dev_sharpe", "gate_sharpe"]] = of[["dev_sharpe", "gate_sharpe"]].astype(float)
of[["config", "dev_sharpe", "gate_sharpe", "note"]]

# ## Conclusion
#
# One P3 config survives: follow the taker imbalance, 10-day smoothing.
#
# **P1.** Plain cross-sectional reversal loses money after 7 bps (lb=3:
# dev -0.99, gate +0.12; lb=5: dev -1.36, gate +0.14; lb=10: dev -1.41,
# gate -0.69). All three baselines are negative on dev. Restricting the trade
# to quiet names (volume z-score below 0 or below -0.5) cannot rescue a signal
# that is already negative: all six conditioned configs are negative on both
# windows. No P1 survivor.
#
# **P2.** The unconfirmed-move mask keeps about 40% of the cross-section but
# does not change the result: lb=3 dev -0.22, gate -0.34; lb=5 dev -0.86,
# gate -0.11. Both are negative on dev. With no reversal edge left at 7 bps,
# filtering on flow disagreement has nothing to work with. No P2 survivor.
#
# **P3.** Fading the imbalance loses at every smoothing (gate -3.21, -3.60
# and -3.60 for k=1, 5 and 10). Following it:
#   k=1:  dev -0.86, gate -2.77 (fails)
#   k=5:  dev +0.50, gate +1.45 (passes; neighbour k=10 positive)
#   k=10: dev +0.89, gate +2.12 (passes; neighbour k=5 positive)
#
# k=5 and k=10 both pass. k=10 is selected for the best gate Sharpe, and the
# results improve steadily with smoothing (k=1 fails, k=5 passes, k=10 is
# stronger). The fade rows are not separate evidence: fading holds the same
# weights with the sign flipped, so it earns roughly minus the follow
# signal's gross return and pays the same costs.
#
# One reading: coins with sustained taker buying over ten days have demand
# that is not fully in the price yet, so the signal behaves like flow
# momentum.
#
# **Caveat.** Inside the point-in-time universe the signal is almost always
# available (over 99.99% of universe-days; the missing values are names
# before their listing). The bigger issue is breadth. The universe held about
# 16-20 names in early 2020 (23 on average that year), about 65 in 2022, 90 in
# 2024 and 99 in 2025. A 100-name dollar-neutral book diversifies much better
# than a 20-name one, so part of the gap between gate (+2.12) and dev (+0.89)
# may be mechanical rather than a change of regime. Notebook 06 checks the
# sleeve on the lockbox, where it scored +1.95.
#
# **Survivor: P3 follow, k=10.** `sleeve_orderflow.parquet` is written below.

imb_c_final = imb - 0.5
raw_final = imb_c_final.rolling(10).mean()
sig_final = signals.cross_sectional_zscore(raw_final, universe)
w_final = signals.signal_to_weights(sig_final, universe, long_short=True)
best = backtest.run(w_final, returns, cost_bps=7).net_returns.loc["2020-01-01":]
print(f"Survivor P3 follow k=10: dev={metrics.sharpe(best.loc[DEV]):+.3f}  gate={metrics.sharpe(best.loc[GATE]):+.3f}")
best.rename("orderflow").to_frame().to_parquet(PROC / "sleeve_orderflow.parquet")
print("sleeve_orderflow saved:", best.shape, "index max:", best.index.max().date())
