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

# # 11: Realism checks on the frozen book
#
# Added on 2026-09-27, after the lockbox, the forward test and the v2
# registration. Each step makes one assumption of the backtest more
# realistic and reruns the same frozen book: same sleeves, parameters, costs
# and walk-forward settings. Like notebook 07, this re-scores the lockbox
# year with corrected inputs; it changes nothing in the book and chooses
# nothing. v2's rules were committed before this notebook was first run, so
# none of it feeds into them. No trials are logged.
#
# The steps build on each other:
#
# 0. As run: the research data (notebooks 06 and 08).
# 0b. The same coin list rebuilt from the archive: a like-for-like check that
#     the data source alone changes nothing.
# 1. The coin list's non-crypto assets taken out: six pegged assets and,
#    from July 2026, five tokenized stocks.
# 2. Every Binance pair, delisted coins included (JUP and SYRUP too), with
#    funding for every perp instead of the 55 the research fetched.
# 3. Carry measured on perp prices instead of spot.
# 4. Orderflow and Carry trading with limit orders that fill only when the
#    next day's price trades through them.
#
# Then three checks that stand on their own: where Orderflow's edge sits by
# coin size and how much money it could take, the pairs baseline with its
# rebalance trades charged, and the frozen book's whole out-of-sample record
# so far.

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, data, metrics, pairs, robustness, strategies

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PROC = ROOT / "data" / "processed"
ALL = PROC / "all_pairs"

WINDOWS = {"dev": slice("2020-01-01", "2024-07-31"), "gate": slice("2024-08-01", "2025-06-30"),
           "lockbox": slice("2025-07-01", "2026-07-06"), "forward": slice("2026-07-07", "2026-09-26")}
SLEEVES = ["seasonality", "orderflow", "carry"]
BOOT = dict(n_boot=2000, method="block", seed=42)
# -

# ## The data

# +
A = {n: pd.read_parquet(ALL / f"{n}.parquet") for n in
     ["price", "returns", "dollar_volume", "taker_imbalance", "high", "low", "universe",
      "perp_price", "perp_high", "perp_low", "perp_returns", "perp_dollar_volume", "funding", "benchmarks"]}
RL = {n: pd.read_parquet(ALL / "research_list" / f"{n}.parquet")
      for n in ["price", "returns", "dollar_volume", "taker_imbalance", "funding"]}
non_crypto = sorted(c for c in RL["price"].columns if c in data.PEGGED | data.TOKENIZED_STOCKS)
print(f"research list: {RL['price'].shape[1]} coins, non-crypto among them: {non_crypto}")
print(f"all pairs: {A['price'].shape[1]} coins; universe coin-days {int(A['universe'].to_numpy().sum()):,}")


def book(sleeves: pd.DataFrame) -> pd.DataFrame:
    """Both frozen combination rules on a set of sleeves."""
    s = sleeves[SLEEVES].dropna()
    wf, _ = robustness.walk_forward_weights(s, train_days=756, step_days=63, min_train=252)
    return pd.DataFrame({"walk_forward": wf, "equal_weight": s.mean(axis=1)})


def by_window(s: pd.Series) -> dict:
    return {w: metrics.sharpe(s.loc[sl].dropna()) for w, sl in WINDOWS.items()}
# -

# ## Step 0: as run
#
# The research sleeves and books up to 2026-07-06, then the forward test's
# (notebook 08) from 2026-07-07.

# +
research = pd.read_parquet(PROC / "sleeves_full.parquet")
fwd = pd.read_parquet(PROC / "forward_book.parquet").loc["2026-07-07":]
cp = pd.read_parquet(PROC / "combined_portfolio.parquet")
step = {0: pd.concat([pd.concat([research[SLEEVES], cp[["walk_forward", "equal_weight"]]], axis=1),
                      fwd[SLEEVES + ["walk_forward", "equal_weight"]]]).sort_index()}
# -

# ## Step 0b: the same coin list, archive data


# +
def frozen_sleeves(p: dict, universe: pd.DataFrame, funding: pd.DataFrame, carry_returns=None) -> pd.DataFrame:
    """The three frozen sleeves on a set of panels."""
    return pd.DataFrame({
        "seasonality": strategies.seasonal_momentum_sleeve(p["price"], p["returns"], universe),
        "orderflow": strategies.orderflow_sleeve(p["taker_imbalance"], p["returns"], universe),
        "carry": strategies.carry_sleeve(funding.reindex(columns=universe.columns),
                                         p["returns"] if carry_returns is None else carry_returns, universe),
    })


uni0b = data.build_universe(RL, top_n=100, min_adv_usd=1_000_000)
s0b = frozen_sleeves(RL, uni0b, RL["funding"])
step["0b"] = pd.concat([s0b, book(s0b)], axis=1)
# -

# ## Step 1: non-crypto assets out
#
# The research coin list without EUR, PAXG, XAUT, RLUSD, USD1, U and the
# five tokenized stocks, universe rebuilt with the same rule, Carry on the
# same perps minus the two gold tokens.

# +
crypto = [c for c in RL["price"].columns if c not in non_crypto]
p1 = {k: RL[k][crypto] for k in ["price", "returns", "dollar_volume", "taker_imbalance"]}
uni1 = data.build_universe(p1, top_n=100, min_adv_usd=1_000_000)
s1 = frozen_sleeves(p1, uni1, RL["funding"][[c for c in RL["funding"].columns if c in crypto]])
step[1] = pd.concat([s1, book(s1)], axis=1)
stock_days = uni0b.reindex(columns=[c for c in non_crypto if c in data.TOKENIZED_STOCKS]).loc["2026-07-07":].sum()
print("tokenized stocks' universe days in the forward window (research list):", stock_days[stock_days > 0].to_dict())
# -

# ## Step 2: every pair
#
# The archive universe: delisted coins in, pegged assets out, JUP and SYRUP
# back. Carry can trade any coin in it with a perp.

# +
uni2, fund2 = A["universe"], A["funding"].reindex(columns=A["universe"].columns)
s2 = frozen_sleeves(A, uni2, fund2)
step[2] = pd.concat([s2, book(s2)], axis=1)
carry_names = (uni2 & fund2.notna()).sum(axis=1)
print("coins Carry can trade, median by window:",
      {w: int(carry_names.loc[sl].median()) for w, sl in WINDOWS.items()})
# -

# Which coins drove the gate losses on every pair, and whether the research
# list had them (P&L before costs, summed over the gate's days):

# +
gate_rows, research_set = {}, set(RL["price"].columns)
for name, w in [("orderflow", strategies.orderflow_weights(A["taker_imbalance"], uni2)),
                ("carry", strategies.carry_weights(fund2, uni2))]:
    held = w.shift(1)
    pnl = (held * A["returns"]).fillna(0.0)
    if name == "carry":
        pnl = pnl - (held * fund2).fillna(0.0)
    by = pnl.loc[WINDOWS["gate"]].sum().sort_values()
    print(f"{name}: gate P&L before costs {by.sum():+.3f}")
    for coin, v in by.head(5).items():
        gate_rows[(name, coin)] = {"gate P&L": round(v, 3), "in the research list": coin in research_set}
pd.DataFrame(gate_rows).T
# -

# ## Step 3: Carry on perp prices

perp_returns = A["perp_returns"].reindex(columns=uni2.columns)
s3 = s2.assign(carry=strategies.carry_sleeve(fund2, perp_returns, uni2))
step[3] = pd.concat([s3, book(s3)], axis=1)

# ## Step 4: limit orders that have to be filled
#
# `backtest.run_limit_fills`: each day's trades are limit orders at that
# day's close, and one fills only if the next day trades through its price.
# Orderflow uses spot bars and Carry perp bars. A missed order is replaced
# at the next close, so positions lag their targets on exactly the days the
# price runs away.

# +
w_of = strategies.orderflow_weights(A["taker_imbalance"], uni2)
of4, held_of = backtest.run_limit_fills(w_of, A["returns"], A["price"], A["high"], A["low"], cost_bps=7.0)
w_ca = strategies.carry_weights(fund2, uni2)
perp_px = A["perp_price"].reindex(columns=uni2.columns)
ca4, held_ca = backtest.run_limit_fills(w_ca, perp_returns, perp_px, A["perp_high"].reindex(columns=uni2.columns),
                                        A["perp_low"].reindex(columns=uni2.columns), cost_bps=7.0)
carry4 = ca4.net_returns - (held_ca * fund2).sum(axis=1).reindex(ca4.net_returns.index)
s4 = s3.assign(orderflow=of4.net_returns.reindex(s3.index), carry=carry4.reindex(s3.index))
step[4] = pd.concat([s4, book(s4)], axis=1)

for name, w, held in [("orderflow", w_of, held_of), ("carry", w_ca, held_ca)]:
    ordered = (w.reindex_like(held).fillna(0.0).shift(1) - held.shift(1)).abs().sum(axis=1)
    done = held.diff().abs().sum(axis=1)
    sl = WINDOWS["dev"]
    print(f"{name}: share of the ordered notional that filled, dev: {done.loc[sl].sum() / ordered.loc[sl].sum():.1%}")
# -

# ## The steps side by side (Sharpe)

# +
labels = {0: "0 as run", "0b": "0b archive data", 1: "1 non-crypto out", 2: "2 every pair",
          3: "3 carry on perps", 4: "4 limit fills"}
steps = pd.DataFrame({(labels[k], col): by_window(df[col]) for k, df in step.items()
                      for col in ["walk_forward", "equal_weight"] + SLEEVES}).T
steps.round(2)
# -

# ## Where Orderflow's edge sits
#
# The same rule on the 30 and 50 largest coins, and on the coins ranked 51
# to 100 by volume, on the research coin list (step 1) and on every pair
# (step 2), standard fills.

# +
size = {}
for label, P, u100 in [("research list", p1, uni1), ("every pair", A, uni2)]:
    u30 = data.build_universe(P, top_n=30, min_adv_usd=1_000_000)
    u50 = data.build_universe(P, top_n=50, min_adv_usd=1_000_000)
    for bucket, u in [("top 30", u30), ("top 50", u50), ("ranks 51-100", u100 & ~u50), ("top 100", u100)]:
        size[f"{label}|{bucket}"] = strategies.orderflow_sleeve(P["taker_imbalance"], P["returns"], u)
size_table = pd.DataFrame({k: by_window(v) for k, v in size.items()}).T
size_table.index = pd.MultiIndex.from_tuples([tuple(k.split("|")) for k in size_table.index])
size_table.round(2)
# -

# On the research list, Orderflow's gate and lockbox results come from the
# smaller coins, ranks 51 to 100. On every pair those coins lose money on
# both windows. The research list's smaller coins were the ones that went on
# to be big in 2026, which is the survivorship bias at work. The largest
# coins do better on every pair, but not on the lockbox.

# ## How much money it could take
#
# Market impact per coin with the square-root law
# (`robustness.impact_drag`): trading Q dollars of a coin with daily volume V
# and daily volatility sigma costs about sigma * sqrt(Q / V) on top of the
# 7 bps fee, with y = 1. Estimates of y run from about 0.5 to 1, so both
# are shown. V is Binance spot volume alone, which makes both on the harsh
# side. Applied to the research-list sleeves of step 1, the version closest
# to the research, over 2020-01 to 2026-07.

# +
AUM = [1e5, 1e6, 1e7, 5e7]
span = slice("2020-01-01", "2026-07-06")
fund1 = RL["funding"][[c for c in RL["funding"].columns if c in crypto]].reindex(columns=uni1.columns)
cap = {}
for name, net, w in [("orderflow", s1["orderflow"], strategies.orderflow_weights(p1["taker_imbalance"], uni1)),
                     ("carry", s1["carry"], strategies.carry_weights(fund1, uni1))]:
    cap[(name, "no impact")] = {"sharpe y=0.5": metrics.sharpe(net.loc[span].dropna()),
                                "sharpe y=1": metrics.sharpe(net.loc[span].dropna())}
    for y in [0.5, 1.0]:
        c = robustness.capacity_by_coin(net.loc[span].dropna(), w, p1["returns"], p1["dollar_volume"], AUM, y=y)
        for aum, row in c.iterrows():
            cap.setdefault((name, f"${aum / 1e6:g}M"), {})[f"sharpe y={y:g}"] = row["sharpe"]
capacity = pd.DataFrame(cap).T
capacity.round(2)
# -

# ## The pairs baseline with its rebalance trades charged
#
# The pairs engine does not charge for positions a pair already has when it
# is selected, or for closing pairs dropped at a rebalance. Charging both at
# every rebalance, even for pairs selected again, gives an upper bound on
# what is missing. Research data, as in notebook 01.

# +
price_r, returns_r, uni_r = (pd.read_parquet(PROC / f"{n}.parquet") for n in ["price", "returns", "universe"])
pairs_charged = pairs.backtest_pairs(price_r, returns_r, uni_r, start="2020-01-01", cost_bps=7.0,
                                                entry=2.0, exit=0.0, zwin=30, min_corr=0.8, top_k=20,
                                                charge_boundaries=True)[0]
base = pd.read_parquet(PROC / "baselines_full.parquet")["reversal"]
pairs_table = pd.DataFrame({"as run": by_window(base), "rebalance trades charged": by_window(pairs_charged)}).T
pairs_table.drop(columns="forward").round(2)
# -

# ## The frozen book's out-of-sample record so far
#
# The lockbox year and the 82-day forward test are the only data the frozen
# book never influenced. Put together (as run, step 0):

# +
oos = step[0].loc["2025-07-01":"2026-09-26"]
record = {}
for col in ["walk_forward", "equal_weight"]:
    r = oos[col].dropna()
    lo, hi = metrics.bootstrap_sharpe_ci(r, **BOOT)
    record[col] = {"days": len(r), "sharpe": metrics.sharpe(r), "t_stat": metrics.sharpe(r) * np.sqrt(len(r) / 365),
                   "ci_low": lo, "ci_high": hi, "total_return": (1 + r).prod() - 1}
record = pd.DataFrame(record).T
record.round(3)
# -

# And the forward window's beta to BTC, against the rolling 82-day beta
# over the research years:

# +
btc = A["benchmarks"]["BTC"]
beta_rows = {}
for col in ["walk_forward", "orderflow"]:
    both = pd.concat([step[0][col], btc], axis=1, sort=True).dropna()
    both.columns = ["r", "btc"]
    roll = both["r"].rolling(82).cov(both["btc"]) / both["btc"].rolling(82).var()
    hist = roll.loc["2020-03-22":"2026-07-06"].dropna()
    fw = both.loc[WINDOWS["forward"]]
    b_fwd = np.cov(fw["r"], fw["btc"])[0, 1] / fw["btc"].var()
    beta_rows[col] = {"forward beta": b_fwd, "research 5th pct": hist.quantile(0.05),
                      "research median": hist.median(), "share of research windows lower": (hist < b_fwd).mean()}
pd.DataFrame(beta_rows).T.round(3)
# -

# ## What the checks say
#
# The data source is not the problem: rebuilt from the archive, the research
# coin list gives the same numbers (step 0b). Taking out the non-crypto
# assets costs the walk-forward book a little, 1.46 to 1.26 on the lockbox.
#
# The coin list is. On every pair (step 2), the frozen book's lockbox Sharpe
# is 0.16 walk-forward and -0.62 equal weight. Orderflow and Carry both lose
# money on the gate and the lockbox; Seasonality keeps its dev and gate
# numbers and still loses on the lockbox. The research list was picked by
# volume in July 2026, and 46% of the universe's coin-days before then
# belonged to coins it left out (notebook 09). The coins that did the damage
# are mostly ones the list never had. Four of Orderflow's five worst coins on
# the gate were missing from it, OM above all, which collapsed in April
# 2025. All five of z-score Carry's were: delisting casualties such as VIDT
# and BNX, which it bought because their funding had turned negative.
#
# The other steps matter less. Perp prices help Carry a little. Limit orders
# that have to be filled hurt Orderflow: only 58% of what it orders fills on
# dev, the misses fall on the days the price runs away, and its dev Sharpe
# drops from 0.44 to 0.09. Even on the research list, most of Orderflow's
# edge is gone at $1M under the square-root impact model (0.30 or -0.27,
# depending on the impact coefficient, from 0.87 with none); Carry lasts to
# somewhere between $1M and $10M. Charging the pairs baseline for its
# rebalance trades barely moves it.
#
# Put together, the lockbox year and the forward test give the frozen book,
# as run, a Sharpe of 0.37 over 453 days (t = 0.4). In the forward window
# the book's beta to BTC was lower than in 93% of earlier 82-day stretches,
# and Orderflow's lower than in 99%.
#
# The frozen book's lockbox result came mostly from its coin list. v2
# (notebook 10) was registered before these checks ran; it trades the
# every-pair universe with rank-weighted Carry on perps, and its test is the
# data from 2026-09-28 on.

# ## Save for the report

# +
series = pd.concat({labels[k]: df for k, df in step.items()}, axis=1, sort=True)
series.columns = [f"{a}|{b}" for a, b in series.columns]
series.to_parquet(PROC / "realism_steps.parquet")
pd.DataFrame(size).to_parquet(PROC / "realism_size.parquet")
capacity.rename_axis(["sleeve", "aum"]).to_csv(PROC / "realism_capacity.csv")
pd.DataFrame({"as run": base, "charged": pairs_charged}).to_parquet(PROC / "realism_pairs.parquet")
record.rename_axis("book").to_csv(PROC / "realism_oos.csv")
print("saved")
# -
