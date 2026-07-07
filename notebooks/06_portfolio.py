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

# # 06: Portfolio and lockbox
#
# ## Commitments, written before any result
#
# This notebook opens the lockbox (2025-07-01 to 2026-07-06) once. The three
# sleeves (seasonality: weekday-halved momentum; orderflow: taker follow,
# k=10; carry: short high funding, k=7) were selected on the dev window
# (2020-01-01 to 2024-07-31) and checked on the gate (2024-08-01 to
# 2025-06-30). The parity check below confirms that the library functions
# reproduce the saved research parquets (to 1e-12) over 2020-01-01 to
# 2025-06-30 before any lockbox data is used.
#
# **The lockbox numbers are reported as they come out.** No construction,
# weighting rule or lookback may change after this notebook runs. A code bug
# (an exception, a parity failure, NaN output) may be fixed and the run
# repeated; a modelling change may not.
#
# Combination: walk-forward mean-variance (train_days=756, step_days=63,
# min_train=252), the library defaults with no tuning, reported next to an
# equal-weight baseline. Both rules are logged to the registry so the trial
# count behind the deflated Sharpe stays complete.

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

matplotlib.use("Agg")  # non-interactive; prevents display errors in nbconvert

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import metrics, robustness, signals, strategies, trials

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
REGISTRY = PROC / "trial_registry.csv"

# Windows
DEV     = slice("2020-01-01", "2024-07-31")
GATE    = slice("2024-08-01", "2025-06-30")
LOCKBOX = slice("2025-07-01", None)        # opens here, once
FULL    = slice("2020-01-01", None)

# +
# Full panels, lockbox included (no .loc[:GATE_END] here). The parity check
# below runs on data up to 2025-06-30 before anything later is used.
price    = pd.read_parquet(PROC / "price.parquet")
returns  = pd.read_parquet(PROC / "returns.parquet")
universe = pd.read_parquet(PROC / "universe.parquet")
funding  = pd.read_parquet(PROC / "funding.parquet")
taker    = pd.read_parquet(PROC / "taker_imbalance.parquet")

# Funding needs the universe's columns, as in notebook 04.
funding = funding.reindex(columns=universe.columns)

print("Panel extents (full, including lockbox):")
for name, df in [("price", price), ("returns", returns), ("universe", universe),
                 ("funding", funding), ("taker", taker)]:
    print(f"  {name}: {df.index.min().date()} → {df.index.max().date()}  shape={df.shape}")

# +
# The three sleeves on the full panels, using the frozen defaults.
seasonality = strategies.seasonal_momentum_sleeve(price, returns, universe)
orderflow   = strategies.orderflow_sleeve(taker, returns, universe)
carry       = strategies.carry_sleeve(funding, returns, universe)

print("Sleeve extents (recomputed on full panels):")
for s in [seasonality, orderflow, carry]:
    r = s.dropna()
    print(f"  {s.name}: {r.index.min().date()} → {r.index.max().date()}  n={len(r)}")

# +
# Parity check, before any lockbox data is read: each recomputed sleeve, cut
# to 2020-01-01:2025-06-30, must match its research parquet to within 1e-12.
# A failure means the frozen function no longer matches the research, and the
# run stops.
def _parity(recomputed: pd.Series, parquet_path: Path, col: str) -> None:
    saved = pd.read_parquet(parquet_path)[col]
    # Compare on the window the parquet covers
    r = recomputed.loc["2020-01-01":"2025-06-30"]
    s = saved.loc["2020-01-01":"2025-06-30"]
    check = pd.concat([r.rename("recomp"), s.rename("saved")], axis=1).dropna()
    if len(check) == 0:
        raise AssertionError(f"Parity check for {col}: no overlapping non-NaN rows")
    ok = np.allclose(check["recomp"].values, check["saved"].values, atol=1e-12)
    print(f"  {col}: {len(check)} rows compared, allclose={ok}")
    assert ok, (
        f"PARITY FAILURE for {col}: max abs diff = "
        f"{(check['recomp'] - check['saved']).abs().max():.2e}"
    )

print("Parity checks:")
_parity(seasonality, PROC / "sleeve_seasonality.parquet", "seasonality")
_parity(orderflow,   PROC / "sleeve_orderflow.parquet",   "orderflow")
_parity(carry,       PROC / "sleeve_carry.parquet",       "carry")
print("All parity checks PASSED.")
# -

# ## Which sleeves go in (decided before the lockbox)
#
# These choices were made on the dev and gate windows before this notebook
# ran.
#
# **In: seasonality** (weekday-halved momentum, weekend=1.0, weekday=0.5).
# It replaces the baseline momentum sleeve: the same trend book with a
# calendar overlay that beat it on both dev and gate. Holding both would
# double up the same position.
#
# **In: orderflow** (taker follow, smooth=10, 7 bps). Positive on dev and
# gate, with both neighbouring k values positive. The idea is that informed
# flow predicts short-term direction.
#
# **In: carry** (short high funding, k=7, 7 bps). Dev +1.01, gate +2.19, and
# both neighbours (k=1, k=30) positive. Funding is carry the short side gets
# paid, and it doubles as a crowding signal.
#
# **Out: baseline pairs.** A dev Sharpe of -0.40 fails the dev > 0 rule.
#
# **Out: fast reversal.** Every config is negative on both windows; costs eat
# the gross edge (notebook 05).

# +
# Sleeve correlations over the dev window.
sleeves_full = pd.concat([seasonality, orderflow, carry], axis=1).dropna()
print(f"Sleeves combined shape (full, dropna): {sleeves_full.shape}")
print(f"  Common index: {sleeves_full.index.min().date()} → {sleeves_full.index.max().date()}")

dev_corr = sleeves_full.loc[DEV].corr()
print("\nSleeve correlation matrix (DEV window 2020-01-01 → 2024-07-31):")
print(dev_corr.round(3))
# -

# n_search is the registry size before the portfolio rows go in: the 48
# research trials are the search space, while the two portfolio rows are
# combination rules rather than searched configs.
n_search = trials.n_trials(REGISTRY)
print(f"Registry rows before portfolio logging: {n_search}")
assert n_search == 48, f"Expected 48 research trial rows, got {n_search}"

# Drop portfolio rows left by an earlier run of this notebook, so a rerun does
# not count them twice. This happens before the two rows are logged below.
_reg = trials.load_registry(REGISTRY)
_pre_portfolio = _reg[_reg["family"] != "portfolio"]
if len(_pre_portfolio) < len(_reg):
    _trimmed = len(_reg) - len(_pre_portfolio)
    print(f"Trimming {_trimmed} prior portfolio row(s) from registry.")
    _pre_portfolio.to_csv(REGISTRY, index=False)
else:
    print("No prior portfolio rows found; registry clean.")

# +
# Combine the sleeves two ways: walk-forward mean-variance (library defaults,
# no tuning) and a plain equal-weight average.
wf, weight_path = robustness.walk_forward_weights(
    sleeves_full,
    train_days=756,
    step_days=63,
    min_train=252,
)
wf.name = "walk_forward"

ew = sleeves_full.mean(axis=1).rename("equal_weight")

print(f"Walk-forward: {wf.dropna().index.min().date()} → {wf.dropna().index.max().date()}  "
      f"n={len(wf.dropna())}")
print(f"Equal-weight: {ew.dropna().index.min().date()} → {ew.dropna().index.max().date()}  "
      f"n={len(ew.dropna())}")

# +
# Log both combination rules (family "portfolio"). Each Sharpe uses whatever
# part of the window the series covers.
wf_dev_sharpe  = metrics.sharpe(wf.loc[DEV].dropna())
wf_gate_sharpe = metrics.sharpe(wf.loc[GATE].dropna())
ew_dev_sharpe  = metrics.sharpe(ew.loc[DEV].dropna())
ew_gate_sharpe = metrics.sharpe(ew.loc[GATE].dropna())

trials.log_trial(
    REGISTRY, "portfolio",
    {"rule": "walk_forward_mv", "train_days": 756, "step_days": 63, "min_train": 252},
    dev_sharpe=wf_dev_sharpe,
    gate_sharpe=wf_gate_sharpe,
    note="walk-forward mean-variance combination of 3 sleeves",
)
trials.log_trial(
    REGISTRY, "portfolio",
    {"rule": "equal_weight"},
    dev_sharpe=ew_dev_sharpe,
    gate_sharpe=ew_gate_sharpe,
    note="equal-weight naive baseline (3 sleeves)",
)

print(f"Logged walk_forward_mv: dev={wf_dev_sharpe:.3f}  gate={wf_gate_sharpe:.3f}")
print(f"Logged equal_weight:    dev={ew_dev_sharpe:.3f}  gate={ew_gate_sharpe:.3f}")
n_after = trials.n_trials(REGISTRY)
assert n_after == 50, f"Expected 50 registry rows after portfolio logging, got {n_after}"
print(f"Registry total: {n_after} rows (48 research + 2 portfolio). PASS.")

# +
# Performance by window (metrics.summary) for both books and each sleeve.
windows = {
    "DEV":     DEV,
    "GATE":    GATE,
    "LOCKBOX": LOCKBOX,
    "FULL":    FULL,
}

series_map = {
    "walk_forward":  wf,
    "equal_weight":  ew,
    "seasonality":   seasonality,
    "orderflow":     orderflow,
    "carry":         carry,
}

rows = []
for win_name, win_slice in windows.items():
    for ser_name, ser in series_map.items():
        r = ser.loc[win_slice].dropna()
        if len(r) < 10:
            continue
        s = metrics.summary(r, n_trials=n_search, name=f"{ser_name}|{win_name}")
        s.name = (win_name, ser_name)
        rows.append(s)

table = pd.DataFrame(rows).T
table.columns = pd.MultiIndex.from_tuples(table.columns)
print("\n=== Performance table ===")
print(table.round(3).to_string())
# -

# HAC t-stat and block-bootstrap 95% CI for the walk-forward book.
print("\n=== Walk-forward significance ===")
for win_name, win_slice in windows.items():
    r = wf.loc[win_slice].dropna()
    if len(r) < 30:
        print(f"  {win_name}: insufficient data (n={len(r)})")
        continue
    hac_t = metrics.sharpe_tstat(r, hac=True)
    lo, hi = metrics.bootstrap_sharpe_ci(r, n_boot=2000, method="block", seed=42)
    sr = metrics.sharpe(r)
    print(f"  {win_name}: Sharpe={sr:.3f}  HAC t={hac_t:.3f}  "
          f"95% CI=[{lo:.3f}, {hi:.3f}]  n={len(r)}")

# Deflated Sharpe of the walk-forward book, full sample and lockbox only,
# with n_trials = n_search = 48 (every trial before the portfolio step).
print("\n=== Deflated Sharpe (walk-forward, n_trials=48) ===")
for label, r in [("FULL",    wf.loc[FULL].dropna()),
                 ("LOCKBOX", wf.loc[LOCKBOX].dropna())]:
    if len(r) < 10:
        print(f"  {label}: insufficient data")
        continue
    dsr = metrics.deflated_sharpe(r, n_trials=n_search)
    sr  = metrics.sharpe(r)
    print(f"  {label}: Sharpe={sr:.3f}  DeflatedSharpe={dsr:.4f}  "
          f"(threshold: 0.95; n_trials={n_search}; n={len(r)})")

# +
# Alpha and beta of the walk-forward book against BTC and the equal-weight
# market, on the full sample and on the lockbox alone.
benchmarks = pd.read_parquet(PROC / "benchmarks.parquet")
btc = benchmarks["BTC"]
mkt = benchmarks["MKT"]

print("\n=== Alpha / Beta ===")
for label, r_slice in [("FULL", FULL), ("LOCKBOX", LOCKBOX)]:
    r = wf.loc[r_slice].dropna()
    ab = metrics.alpha_beta(r, btc, mkt, names=["BTC", "MKT"])
    if not ab:
        print(f"  {label}: no overlap with benchmarks")
        continue
    print(f"  {label} (n={len(pd.concat([r, btc, mkt], axis=1).dropna())}):")
    print(f"    alpha_ann={ab['alpha_ann']:.4f}  alpha_tstat={ab['alpha_tstat']:.3f}")
    print(f"    beta_BTC={ab['beta_BTC']:.3f}  beta_MKT={ab['beta_MKT']:.3f}")
    print(f"    R²={ab['r_squared']:.4f}  IR={ab['resid_info_ratio']:.3f}")
# -

# Weight path of the walk-forward book.
fig, ax = plt.subplots(figsize=(12, 4))
for col in weight_path.columns:
    ax.plot(weight_path.index, weight_path[col], label=col, linewidth=0.8)
ax.axvline(pd.Timestamp("2025-07-01"), color="red", linestyle="--",
           linewidth=1.2, label="Lockbox open")
ax.set_title("Walk-forward sleeve weights (max-Sharpe)")
ax.set_ylabel("Weight")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "06_weight_path.png", dpi=120)
plt.close(fig)
print("Weight-path plot saved.")

# +
# Equity curves: walk-forward, equal weight and BTC.
fig, ax = plt.subplots(figsize=(12, 5))

start = "2020-01-01"
for ser, label, color in [
    (wf, "Walk-forward portfolio", "steelblue"),
    (ew, "Equal-weight portfolio", "darkorange"),
]:
    r = ser.loc[start:].dropna()
    eq = (1 + r).cumprod()
    ax.plot(eq.index, eq, label=label, color=color, linewidth=1.2)

# BTC normalised to 1 at start
btc_r = btc.loc[start:].dropna()
btc_eq = (1 + btc_r).cumprod()
ax.plot(btc_eq.index, btc_eq, label="BTC (normalised)", color="gray",
        linewidth=0.8, linestyle="--")

ax.axvline(pd.Timestamp("2024-08-01"), color="navy", linestyle=":",
           linewidth=1.0, label="Gate start")
ax.axvline(pd.Timestamp("2025-07-01"), color="red", linestyle="--",
           linewidth=1.2, label="Lockbox open")
ax.set_yscale("log")
ax.set_title("Equity curves (log scale), 2020-01-01 onward")
ax.set_ylabel("Cumulative return (1 = start)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / "reports" / "06_equity_curves.png", dpi=120)
plt.close(fig)
print("Equity curves plot saved.")
# -

# Save combined_portfolio.parquet (columns: walk_forward, equal_weight).
combined = pd.concat([wf, ew], axis=1)
combined.index.name = "Date"
combined.to_parquet(PROC / "combined_portfolio.parquet")
print(f"Saved combined_portfolio.parquet: shape={combined.shape}  "
      f"idx={combined.index.min().date()} → {combined.index.max().date()}")
print(combined.describe().round(4))

# The last rows of the registry and the count per family.
print("\n=== Registry summary (final 10 rows) ===")
reg_final = trials.load_registry(REGISTRY)
print(reg_final.tail(10).to_string(index=False))
print(f"\nFamily counts:\n{reg_final['family'].value_counts().to_string()}")
print(f"\nTotal rows: {len(reg_final)}")

# ## Conclusion: the lockbox result
#
# **Lockbox (2025-07-01 to 2026-07-06, 371 days, opened once).** Walk-forward
# Sharpe **+1.458**, equal weight **+1.233**. Beta to BTC -0.020. Annualised
# alpha 9.48% with a t-stat of 1.25, which is not significant on one year of
# data. The block-bootstrap 95% CI, [-0.42, +3.44], includes zero.
#
# **Deflated Sharpe against the 48 research trials: 0.490 on the full sample
# and 0.205 on the lockbox.** Both are below the usual 0.95 bar, so the book
# does not pass the significance test. Out-of-sample performance is positive
# with almost no market exposure, but at this sample size and trial count it
# cannot be told apart from selection luck.
#
# **Sleeves on the lockbox.** Orderflow (+1.95) and carry (+0.88) held up.
# Seasonality came in at -0.57: its gate outperformance did not last, which is
# the regime risk flagged in notebook 02.
#
# **Equal weight vs walk-forward.** Equal weight beats walk-forward on dev,
# gate and the full sample (full: 1.76 vs 1.02) and loses only on the lockbox.
# With three nearly uncorrelated sleeves, re-estimating mean-variance weights
# adds little over a plain average.
#
# **Summary.** A diversified, low-beta book with a positive out-of-sample
# result that is not statistically proven. The trial registry and the
# deflated Sharpe are what keep that claim in proportion.
