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

# # 07: Post-hoc checks
#
# Added in September 2026, after notebook 06 had opened the lockbox. Nothing
# here changes a sleeve, a parameter or the book, and none of it is a new
# trial, so the registry is not touched. Four questions came up when the
# backtests were audited:
#
# 1. The seasonality overlay halves the book on Mondays and restores it on
#    Saturdays. `signals.seasonal_scale` scales the returns but never charges
#    for those trades. What does the sleeve earn with every trade charged, and
#    what does that do to the frozen book, lockbox included?
# 2. Orderflow and carry pay the limit-order rate (7 bps) on every rebalance.
#    How much is left at 14 and 20 bps? (dev and gate only)
# 3. How much of the carry sleeve's P&L comes from funding and how much from
#    price? (dev and gate only)
# 4. How did every strategy do over the whole sample, 2020-01 to 2026-07? The
#    dev/gate/lockbox split is how the book was chosen and tested; this is
#    the plain full-period view next to it, including the two baselines,
#    which notebook 01 only ran up to the end of the gate.

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, metrics, robustness, signals, strategies

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"

DEV = slice("2020-01-01", "2024-07-31")
GATE = slice("2024-08-01", "2025-06-30")
LOCKBOX = slice("2025-07-01", "2026-07-06")
FULL = slice("2020-01-01", "2026-07-06")
WINDOWS = {"dev": DEV, "gate": GATE, "lockbox": LOCKBOX, "full": FULL}

price = pd.read_parquet(PROC / "price.parquet")
returns = pd.read_parquet(PROC / "returns.parquet")
universe = pd.read_parquet(PROC / "universe.parquet")
taker = pd.read_parquet(PROC / "taker_imbalance.parquet")
funding = pd.read_parquet(PROC / "funding.parquet").reindex(columns=universe.columns)
bench = pd.read_parquet(PROC / "benchmarks.parquet")
asrun = pd.read_parquet(PROC / "combined_portfolio.parquet")
# -

# ## 1. Seasonality with every trade charged
#
# First rebuild the as-run book from the frozen sleeves and check it matches
# `combined_portfolio.parquet` exactly, so that the only difference in the
# re-score below is the extra cost.

# +
seas = strategies.seasonal_momentum_sleeve(price, returns, universe)
orderflow = strategies.orderflow_sleeve(taker, returns, universe)
carry = strategies.carry_sleeve(funding, returns, universe)


def combine(seasonality):
    """Walk-forward and equal-weight books with notebook 06's settings."""
    sleeves = pd.concat([seasonality, orderflow, carry], axis=1, sort=True).dropna()
    wf, _ = robustness.walk_forward_weights(sleeves, train_days=756, step_days=63, min_train=252)
    return wf, sleeves.mean(axis=1)


wf_asrun, ew_asrun = combine(seas)
for name, mine, saved in [("walk_forward", wf_asrun, asrun["walk_forward"]),
                          ("equal_weight", ew_asrun, asrun["equal_weight"])]:
    both = pd.concat([mine, saved], axis=1, sort=True).dropna()
    same = np.array_equal(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())
    print(f"{name}: {len(both)} days, identical to notebook 06: {same}")
    assert same
# -

# Now the charged versions. `charge_resizing=True` rebuilds the positions
# actually held each day (momentum weights x vol scale x day multiplier) and
# pays 20 bps on every change. The untilted sleeve is charged the same way,
# because notebook 02's rule compared the overlay with it.

# +
seas_charged = strategies.seasonal_momentum_sleeve(price, returns, universe, charge_resizing=True)
untilted = strategies.seasonal_momentum_sleeve(price, returns, universe, weekend=1.0, weekday=1.0)
untilted_charged = strategies.seasonal_momentum_sleeve(price, returns, universe, weekend=1.0,
                                                       weekday=1.0, charge_resizing=True)


def sharpe_row(s, windows):
    return {w: metrics.sharpe(s.loc[WINDOWS[w]].dropna()) for w in windows}


sleeve_table = pd.DataFrame({
    "seasonality, as run": sharpe_row(seas, ["dev", "gate", "lockbox"]),
    "seasonality, all trades charged": sharpe_row(seas_charged, ["dev", "gate", "lockbox"]),
    "untilted momentum, as run": sharpe_row(untilted, ["dev", "gate"]),
    "untilted momentum, all trades charged": sharpe_row(untilted_charged, ["dev", "gate"]),
}).T
sleeve_table.round(3)
# -

# The overlay's extra trading, as a yearly cost on dev:

extra = (seas.loc[DEV] - seas_charged.loc[DEV]).mean() * 365
print(f"extra cost of resizing on dev: {extra:.2%} a year")

# The same frozen book with the charged seasonality sleeve. Weights,
# windows and the walk-forward settings are unchanged.

# +
wf_charged, ew_charged = combine(seas_charged)


def book_stats(r):
    ab = metrics.alpha_beta(r, bench["BTC"], bench["MKT"], names=["BTC", "MKT"])
    return {
        "sharpe": metrics.sharpe(r),
        "ann_return": metrics.ann_return(r),
        "ann_vol": metrics.ann_vol(r),
        "max_dd": metrics.max_drawdown(r),
        "beta_btc": ab["beta_BTC"],
        "alpha_ann": ab["alpha_ann"],
        "alpha_t": ab["alpha_tstat"],
    }


rows = {}
for label, wf, ew in [("as run", wf_asrun, ew_asrun), ("seasonality charged", wf_charged, ew_charged)]:
    for w in ["dev", "gate", "lockbox", "full"]:
        rows[(label, "walk-forward", w)] = book_stats(wf.loc[WINDOWS[w]].dropna())
        rows[(label, "equal weight", w)] = book_stats(ew.loc[WINDOWS[w]].dropna())
book_table = pd.DataFrame(rows).T
book_table.round(3)
# -

# Significance of the re-scored walk-forward book, with the same 48-trial
# count and bootstrap settings as notebook 06.

# +
sig_rows = {}
for label, wf in [("as run", wf_asrun), ("seasonality charged", wf_charged)]:
    lk = wf.loc[LOCKBOX].dropna()
    full = wf.loc[FULL].dropna()
    lo, hi = metrics.bootstrap_sharpe_ci(lk, n_boot=2000, method="block", seed=42)
    sig_rows[label] = {
        "lockbox sharpe": metrics.sharpe(lk),
        "lockbox 95% CI low": lo,
        "lockbox 95% CI high": hi,
        "deflated sharpe, full": metrics.deflated_sharpe(full, n_trials=48),
        "deflated sharpe, lockbox": metrics.deflated_sharpe(lk, n_trials=48),
    }
pd.DataFrame(sig_rows).T.round(3)
# -

# ## 2. Orderflow and carry at higher costs (dev and gate only)

# +
upto_gate = slice(None, "2025-06-30")
stress = {}
for bps in [7, 14, 20]:
    of = strategies.orderflow_sleeve(taker.loc[upto_gate], returns.loc[upto_gate],
                                     universe.loc[upto_gate], cost_bps=bps)
    ca = strategies.carry_sleeve(funding.loc[upto_gate], returns.loc[upto_gate],
                                 universe.loc[upto_gate], cost_bps=bps)
    for name, s in [("orderflow", of), ("carry", ca)]:
        stress[(name, bps)] = sharpe_row(s.loc["2020-01-01":], ["dev", "gate"])
cost_table = pd.DataFrame(stress).T
cost_table.index.names = ["sleeve", "cost_bps"]
cost_table.round(3)
# -

# ## 3. Carry: funding leg vs price leg (dev and gate only)

# +
fuv = universe & funding.notna()
w = signals.signal_to_weights(-signals.cross_sectional_zscore(funding.rolling(7).mean(), fuv),
                              fuv, long_short=True)
price_leg = backtest.run(w, returns, cost_bps=7).net_returns
funding_leg = -(w.shift(1) * funding).sum(axis=1)
split = {}
for w_name in ["dev", "gate"]:
    sl = WINDOWS[w_name]
    p, f = price_leg.loc[sl], funding_leg.loc[sl]
    split[w_name] = {
        "price leg, mean %/yr": p.mean() * 365,
        "funding leg, mean %/yr": f.mean() * 365,
        "funding share of total": f.mean() / (p.mean() + f.mean()),
        "price leg sharpe": metrics.sharpe(p),
        "funding leg sharpe": metrics.sharpe(f),
    }
carry_table = pd.DataFrame(split).T
carry_table.round(3)
# -

# ## 4. Every strategy over the whole sample
#
# The baselines run with notebook 01's settings on the full panels. Up to
# 2025-06-30 they have to match notebook 01's saved output, so the only new
# information is the year after the gate.

# +
base_mom = strategies.momentum_sleeve(price, returns, universe, lookback=30,
                                      target_vol=0.15, cost_bps=20).loc["2020-01-01":]
base_rev = strategies.reversal_sleeve(price, returns, universe, cost_bps=7).loc["2020-01-01":]
for name, s, col in [("sleeve_baseline_momentum", base_mom, "momentum"),
                     ("sleeve_baseline_reversal", base_rev, "reversal")]:
    saved = pd.read_parquet(PROC / f"{name}.parquet")[col]
    both = pd.concat([s.loc[:"2025-06-30"], saved], axis=1, sort=True).dropna()
    same = np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-12)
    print(f"{name}: {len(both)} days, matches notebook 01 up to 2025-06-30: {same}")
    assert same

series = {
    "baseline momentum": base_mom, "baseline pairs": base_rev,
    "seasonality": seas, "orderflow": orderflow, "carry": carry,
    "book, walk-forward": asrun["walk_forward"], "book, equal weight": asrun["equal_weight"],
    "BTC": bench["BTC"], "equal-weight market": bench["MKT"],
}
full_rows = {}
for name, s in series.items():
    r = s.loc[FULL].dropna()
    row = {"from": r.index.min().date(), "ann_return": metrics.ann_return(r),
           "ann_vol": metrics.ann_vol(r), "sharpe": metrics.sharpe(r),
           "max_dd": metrics.max_drawdown(r)}
    if name not in ("BTC", "equal-weight market"):
        ab = metrics.alpha_beta(r, bench["BTC"], bench["MKT"], names=["BTC", "MKT"])
        row.update(beta_btc=ab["beta_BTC"], beta_mkt=ab["beta_MKT"],
                   alpha=ab["alpha_ann"], alpha_t=ab["alpha_tstat"])
    full_rows[name] = row
full_table = pd.DataFrame(full_rows).T
full_table
# -

# ## Save for the report

pd.concat([seas.rename("seasonality"), orderflow.rename("orderflow"), carry.rename("carry")],
          axis=1).to_parquet(PROC / "sleeves_full.parquet")
pd.DataFrame({
    "seasonality_charged": seas_charged,
    "walk_forward_charged": wf_charged,
    "equal_weight_charged": ew_charged,
}).to_parquet(PROC / "posthoc_book.parquet")
cost_table.reset_index().to_csv(PROC / "posthoc_cost_stress.csv", index=False)
carry_table.rename_axis("window").reset_index().to_csv(PROC / "posthoc_carry_split.csv", index=False)
pd.DataFrame({"momentum": base_mom, "reversal": base_rev}).to_parquet(PROC / "baselines_full.parquet")
print("saved sleeves_full, posthoc_book, posthoc_cost_stress, posthoc_carry_split, baselines_full")

# ## Conclusion
#
# **Seasonality.** Charging every resizing trade costs the sleeve about 2.7%
# a year on dev. Its Sharpe falls from 1.79 to 1.51 on dev, from 1.24 to 0.99
# on the gate and from -0.56 to -0.89 on the lockbox. The untilted momentum
# sleeve, charged the same way, scores 1.64 on dev and 0.42 on the gate. The
# overlay still wins on the gate but loses on dev, so under notebook 02's
# rule (beat the untilted sleeve on both windows) it would not have been
# selected. The error was in the cost accounting; the signal itself is fine.
#
# **The book.** On the lockbox the effect is small and goes both ways:
# walk-forward moves from 1.458 to 1.490 and equal weight from 1.233 to
# 1.158. Walk-forward gains because a weaker seasonality sleeve gets less
# weight in the training windows, and seasonality lost money on the lockbox.
# Over the full sample the walk-forward Sharpe falls from 1.02 to 0.70 and
# its deflated Sharpe from 0.49 to 0.23. On the lockbox the deflated Sharpe
# is 0.21, against 0.20 as run, and beta to BTC stays close to zero. The
# conclusion of notebook 06 stands: a positive, low-beta result that is not
# statistically proven.
#
# **Costs.** At 20 bps instead of 7, orderflow falls to 0.37 on dev and 0.75
# on the gate, and carry to 0.70 and 1.90. Both stay positive, but orderflow
# depends heavily on getting limit-order fills. Carry is less sensitive: it
# turns over about a fifth of its book a day, and most of its dev P&L is
# funding.
#
# **Carry P&L.** On dev, funding brings in 19.3% a year out of the sleeve's
# 30.7% (arithmetic means), 63% of the total. On the gate the price leg did
# more, 41.2% against 26.9%. The funding leg on its own has a very high
# Sharpe (4.8 on dev) because the payments are small and steady; the risk
# sits in the price leg.
#
# **Whole sample.** From 2020 to July 2026 the equal-weight book returned
# 25.6% a year at 13.4% volatility (Sharpe 1.76, worst drawdown -10.1%). The
# walk-forward book, which starts in 2021-10, returned 8.1% at 7.9% (Sharpe
# 1.02). Neither has much market exposure: every beta is within 0.03 of zero.
# BTC returned 39.3% a year over the same period, but at 61% volatility and
# with a -77% drawdown (Sharpe 0.86). Baseline momentum kept a Sharpe of 1.17
# over the whole period and the pairs baseline lost money (-0.31). These
# figures mix the windows the strategies were chosen on with the windows they
# were tested on, so they summarise the history; the lockbox in notebook 06 is
# the clean out-of-sample test.
