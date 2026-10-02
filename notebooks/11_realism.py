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
# none of it feeds into them. Two things found after that first run change
# v2's numbers but not its rules: a Carry position held into a day notebook
# 09's 20% rule drops its perp's data now earns what its contract did that
# day, and two holes in the archive's 2022 files are filled, which gives
# Carry funding data it lacked on those days (steps 2 and 3, notebook 09 and
# the last section). No trials are logged.
#
# The steps build on each other:
#
# 0. As run: the research data (notebooks 06 and 08).
# 0b. The same coin list rebuilt from the archive: a like-for-like check that
#     the data source alone changes nothing.
# 1. The coin list's non-crypto assets taken out: six pegged assets and,
#    from July 2026, five tokenized stocks.
# 2. Every Binance pair, delisted coins included (JUP and SYRUP too), with
#    funding for every perp instead of the 55 the research fetched. Carry's
#    signal reads only perp data that passes notebook 09's 20% rule, and a
#    position it holds is paid its contract's funding on every day the
#    contract traded.
# 3. Carry measured on perp prices instead of spot.
# 4. Orderflow and Carry trading with limit orders that fill only when the
#    next day's price trades through them.
#
# Then four checks that stand on their own: where Orderflow's edge sits by
# coin size and how much money it could take, notebook 05's hourly
# fast-reversal test on every pair, the pairs baseline with its rebalance
# trades charged, and the frozen book's whole out-of-sample record so far.

# +
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, data, metrics, pairs, robustness, signals, strategies, trials

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
      "perp_price", "perp_high", "perp_low", "perp_returns", "perp_dollar_volume", "funding", "benchmarks",
      "perp_price_traded", "perp_high_traded", "perp_low_traded", "perp_returns_traded", "funding_traded"]}
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
FUNDING_START = research["carry"].first_valid_index()   # the research Carry starts with its funding data


def from_funding_start(carry: pd.Series) -> pd.Series:
    """Blank before the research Carry's first day, so the walk-forward book
    starts on the same day as the research's and refits on the same dates.
    Every-pair funding only starts on 2020-01-01 (notebook 09), so there Carry
    holds nothing, and returns 0, from 2019-09-10 to 2019-12-31."""
    return carry.where(carry.index >= FUNDING_START)


def frozen_sleeves(p: dict, universe: pd.DataFrame, funding: pd.DataFrame, pnl_funding=None) -> pd.DataFrame:
    """The three frozen sleeves on a set of panels."""
    return pd.DataFrame({
        "seasonality": strategies.seasonal_momentum_sleeve(p["price"], p["returns"], universe),
        "orderflow": strategies.orderflow_sleeve(p["taker_imbalance"], p["returns"], universe),
        "carry": from_funding_start(strategies.carry_sleeve(
            funding.reindex(columns=universe.columns), p["returns"], universe, pnl_funding=pnl_funding)),
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
paid2 = A["funding_traded"].reindex(columns=uni2.columns)      # what a held perp paid, whatever its basis
s2 = frozen_sleeves(A, uni2, fund2, pnl_funding=paid2)
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
        pnl = pnl - (held * paid2).fillna(0.0)
    by = pnl.loc[WINDOWS["gate"]].sum().sort_values()
    print(f"{name}: gate P&L before costs {by.sum():+.3f}")
    for coin, v in by.head(5).items():
        gate_rows[(name, coin)] = {"gate P&L": round(v, 3), "in the research list": coin in research_set}
pd.DataFrame(gate_rows).T
# -

# ## Step 3: Carry on perp prices
#
# Carry's price leg on its perps instead of spot. Notebook 09's 20% rule
# drops a perp's data on days it trades far from spot or stops trading, and
# the signal only reads what passes. A position held into such a day earns
# what its contract did: its own move and funding if it traded, however far
# from spot, and nothing if it did not, as when a perp is delisted and
# settled at its last price (the `_traded` panels). The first run of this
# notebook counted those days as zero, funding included. (Two holes in the
# archive's own files, about 50 perps missing on 2022-02-26 to 02-28 and
# 2022-04-01 to 04-02, are filled from its daily files in notebook 09.)

perp_paid = A["perp_returns_traded"].reindex(columns=uni2.columns)
s3 = s2.assign(carry=from_funding_start(strategies.carry_sleeve(fund2, perp_paid, uni2, pnl_funding=paid2)))
step[3] = pd.concat([s3, book(s3)], axis=1)

# The held coin-days whose perp data the rule drops, and what the contract
# did on the ones it traded (weights and moves as shares of the sleeve), then
# Carry with those days at zero, as in the first run, and as measured here:

# +
perp_returns = A["perp_returns"].reindex(columns=uni2.columns)
spot_r = A["returns"].reindex_like(perp_returns)
held3 = strategies.carry_weights(fund2, uni2).shift(1).reindex_like(perp_returns)
gap = (held3.fillna(0.0) != 0) & perp_returns.isna()
rows, cols = np.nonzero(gap.to_numpy())
gaps = pd.DataFrame({(gap.index[i].date(), gap.columns[j]): {
    "held": held3.iat[i, j], "spot move": spot_r.iat[i, j], "perp move": perp_paid.iat[i, j], "funding": paid2.iat[i, j]}
    for i, j in zip(rows, cols)}).T
live = gaps["perp move"].notna()
print(f"held coin-days with the perp's data dropped: {len(gaps)}; the contract traded on {int(live.sum())} "
      f"and did not trade on {int((~live).sum())}")
print(gaps[live].round(4).to_string())
at_zero = from_funding_start(strategies.carry_sleeve(fund2, perp_returns, uni2))
pd.DataFrame({"dropped days at zero": by_window(at_zero), "held contracts' own data": by_window(s3["carry"])}).T.round(2)
# -

# ## Step 4: limit orders that have to be filled
#
# `backtest.run_limit_fills`: each day's trades are limit orders at that
# day's close, and one fills only if the next day trades through its price.
# Orderflow uses spot bars and Carry its perps' own bars on every day they
# traded, as in step 3. A missed order is replaced at the next close, so
# positions lag their targets on exactly the days the price runs away. An
# order cannot fill on a contract that does not trade, so a position there
# stays on the books; if the contract trades again within three days the
# position takes the move across the gap. After more than three days it is
# treated as delisted: settled at its last price and dropped, so a relaunch
# under the same name, like LUNA's in 2022, does not bring it back. A day's
# funding is credited to the position after that day's fill, although the
# prints before a fill belong to the old position; the daily funding sums
# cannot split the day.

# +
w_of = strategies.orderflow_weights(A["taker_imbalance"], uni2)
of4, held_of = backtest.run_limit_fills(w_of, A["returns"], A["price"], A["high"], A["low"], cost_bps=7.0)
w_ca = strategies.carry_weights(fund2, uni2)
perp_px, perp_hi, perp_lo = (A[f"{k}_traded"].reindex(columns=uni2.columns) for k in ["perp_price", "perp_high", "perp_low"])
ca4, held_ca = backtest.run_limit_fills(w_ca, perp_paid, perp_px, perp_hi, perp_lo, cost_bps=7.0)
carry4 = ca4.net_returns - (held_ca * paid2).sum(axis=1).reindex(ca4.net_returns.index)
s4 = s3.assign(orderflow=of4.net_returns.reindex(s3.index), carry=from_funding_start(carry4.reindex(s3.index)))
step[4] = pd.concat([s4, book(s4)], axis=1)

fills = {}
for name, w, held, rets, close, hi, lo in [
        ("orderflow", w_of, held_of, A["returns"], A["price"], A["high"], A["low"]),
        ("carry", w_ca, held_ca, perp_paid, perp_px, perp_hi, perp_lo)]:
    target = w.reindex_like(held).fillna(0.0).shift(1)          # what run() holds each day
    traded = hi.reindex_like(held).notna() & lo.reindex_like(held).notna() & close.reindex_like(held).shift(1).notna()
    missed = (target - held).where(traded)                       # a coin with no bar cannot fill at all
    r, sl = rets.reindex_like(held), WINDOWS["dev"]
    fills[name] = {
        "filled share of orders": held.diff().abs().where(traded).loc[sl].sum().sum()   # settling is not a fill
                                  / (target - held.shift(1)).abs().where(traded).loc[sl].sum().sum(),
        "missed share of positions": missed.abs().loc[sl].sum().sum() / target.abs().where(traded).loc[sl].sum().sum(),
        "return where a buy missed": r.where(missed > 1e-9).loc[sl].stack().mean(),
        "return where a sell missed": r.where(missed < -1e-9).loc[sl].stack().mean(),
        "price P&L, every order filled": (target * r).sum(axis=1).loc[sl].sum(),
        "price P&L, fill model": (held * r).sum(axis=1).loc[sl].sum(),
    }
print("dev, orders on coins that traded the next day; P&L is the price leg (all of it for Orderflow, "
      "Carry's before funding), as sums of daily returns")
pd.DataFrame(fills).T.round(4)
# -

# Almost every order fills; the few that miss are the ones the price ran
# away from, and they carry most of the damage.

# ## The steps side by side (Sharpe)

# +
labels = {0: "0 as run", "0b": "0b archive data", 1: "1 non-crypto out", 2: "2 every pair",
          3: "3 carry on perps", 4: "4 limit fills"}
steps = pd.DataFrame({(labels[k], col): by_window(df[col]) for k, df in step.items()
                      for col in ["walk_forward", "equal_weight"] + SLEEVES}).T
steps.round(2)
# -

# ## The walk-forward book's refit dates
#
# The walk-forward book refits every 63 days, counted from its first day, so
# its numbers depend on where those dates fall. Here is its lockbox Sharpe
# with the book started up to eight weeks later, which moves every refit
# date and nothing else (the equal-weight book has no refit dates):

# +
refits = {}
for key in [0, 2, 3, 4]:
    sl = step[key][SLEEVES].dropna()
    for later in range(0, 63, 7):
        wf, _ = robustness.walk_forward_weights(sl.iloc[later:], train_days=756, step_days=63, min_train=252)
        refits[(labels[key], later)] = by_window(wf)
refits = pd.DataFrame(refits).T.rename_axis(["step", "days later"])
refits["lockbox"].unstack().round(2)
# -

# The research list's book stays between 1.30 and 1.55 on the lockbox, and
# the every-pair book between 0.08 and 0.57, so the gap between them does
# not depend on the calendar, but the walk-forward figures are only good to
# a few tenths.

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

# ## Fast reversal on every pair
#
# Notebook 05's grid again: hourly cross-sectional reversal, lookback 4 to
# 48 hours, rebalanced every 1 or 6 hours, 7 bps, a coin tradable in an hour
# if it was in the daily top-100 universe that day, dev from 2020-06 and the
# gate, and the rule that gross return must be at least 3 times the cost
# drag. First on the research panel of 60 coins, which must reproduce
# notebook 05, then on hourly bars for every coin that was in the every-pair
# universe (notebook 09).

# +
PPY = 24 * 365
DEV_H = slice("2020-06-01", "2024-07-31")


def fast_grid(price_h: pd.DataFrame, uni_daily: pd.DataFrame) -> pd.DataFrame:
    """Notebook 05's eight configs on an hourly panel; no trials are logged."""
    price_h = price_h.loc["2020-06-01":"2025-06-30"]
    rets_h = price_h.pct_change(fill_method=None)
    uni_h = (uni_daily.loc[:"2025-06-30"].reindex(columns=price_h.columns)
             .reindex(price_h.index, method="ffill").fillna(False))
    rows = {}
    for lb in [4, 12, 24, 48]:
        for rb in [1, 6]:
            sig = -signals.cross_sectional_zscore(signals.trailing_return(price_h, lb), uni_h)
            w = signals.signal_to_weights(sig, uni_h, long_short=True)
            if rb > 1:
                w = w.where(pd.Series(np.arange(len(w)) % rb == 0, index=w.index)).ffill()
            res = backtest.run(w, rets_h, cost_bps=7)
            gross_dev = res.gross_returns.loc[DEV_H]
            gross_ann, drag_ann = gross_dev.mean() * PPY, res.turnover.mean() * 7 / 1e4 * PPY
            rows[(lb, rb)] = {"names per hour": uni_h.sum(axis=1).loc[DEV_H].mean(),
                              "gross_sharpe_dev": metrics.sharpe(gross_dev, periods_per_year=PPY),
                              "gross_ann": gross_ann, "cost_drag_ann": drag_ann,
                              "gross_over_drag": gross_ann / drag_ann,
                              "net_dev": metrics.sharpe(res.net_returns.loc[DEV_H], periods_per_year=PPY),
                              "net_gate": metrics.sharpe(res.net_returns.loc[WINDOWS["gate"]], periods_per_year=PPY)}
    out = pd.DataFrame(rows).T
    out.index.names = ["lookback_h", "rebal_h"]
    return out


fast = pd.concat({"research panel": fast_grid(pd.read_parquet(PROC / "price_1h.parquet"),
                                              pd.read_parquet(PROC / "universe.parquet")),
                  "every pair": fast_grid(pd.read_parquet(ALL / "price_1h.parquet"), uni2)})
fast.round(2)
# -

# The 60-coin panel reproduces notebook 05. On every pair, with about three
# times as many coins in a typical hour, the gross edge is larger, about as
# large as the trading cost at the best settings, but nowhere near three
# times it, and every config still loses money after costs on dev and gate.

# ## The pairs baseline, fully charged and without its look-ahead
#
# The pairs engine does not charge for positions a pair already has when it
# is selected, or for closing pairs dropped at a rebalance. Charging both at
# every rebalance, even for pairs selected again, gives an upper bound on
# what is missing. The engine also picks its pairs with the rebalance date's
# close and then books that date's return for them, a one-day look-ahead at
# each rebalance; `lag_selection=True` picks them with data up to the day
# before. Research data, as in notebook 01.

# +
price_r, returns_r, uni_r = (pd.read_parquet(PROC / f"{n}.parquet") for n in ["price", "returns", "universe"])
pairs_kw = dict(start="2020-01-01", cost_bps=7.0, entry=2.0, exit=0.0, zwin=30, min_corr=0.8, top_k=20)
pairs_charged = pairs.backtest_pairs(price_r, returns_r, uni_r, charge_boundaries=True, **pairs_kw)[0]
pairs_lagged = pairs.backtest_pairs(price_r, returns_r, uni_r, lag_selection=True, **pairs_kw)[0]
pairs_both = pairs.backtest_pairs(price_r, returns_r, uni_r, charge_boundaries=True, lag_selection=True,
                                  **pairs_kw)[0]
base = pd.read_parquet(PROC / "baselines_full.parquet")["reversal"]
pairs_table = pd.DataFrame({"as run": by_window(base), "rebalance trades charged": by_window(pairs_charged),
                            "picked a day earlier": by_window(pairs_lagged), "both": by_window(pairs_both)}).T
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

# ## v2 re-measured, dev and gate only
#
# v2's Carry sleeve, as registered and first run in notebook 10, ran on data
# with the two archive holes of 2022, and a position it held into a day the
# 20% rule drops earned nothing that day, as in the first run of step 3. Its
# test (notebook 12) runs on the filled data, and such a position earns what
# its contract did, as in steps 2 and 3. Here are its dev and gate numbers
# each way. The data is cut at the end of the gate first, as in notebook 10,
# so v2 is still never run on the lockbox or the forward window. With the
# registered rules, the filled data changes Carry only in the weeks after the
# two holes, and nothing else. The first check below stops if
# `v2_dev_gate.parquet` no longer holds notebook 10's first run, as the
# registry logged it.

# +
cut = {k: A[k].loc[:"2025-06-30"] for k in ["returns", "taker_imbalance", "universe", "perp_returns", "funding",
                                            "perp_returns_traded", "funding_traded"]}
u = cut["universe"]
v2_args = (cut["taker_imbalance"], cut["returns"], cut["perp_returns"].reindex(columns=u.columns),
           cut["funding"].reindex(columns=u.columns), u)
registered = pd.read_parquet(PROC / "v2_dev_gate.parquet")
logged = trials.load_registry(PROC / "trial_registry.csv").query("family == 'v2'")
assert np.allclose(sorted(logged["dev_sharpe"]), sorted(metrics.sharpe(registered[c].loc[WINDOWS["dev"]].dropna())
                                                        for c in ["orderflow", "carry", "v2"]), rtol=0, atol=1e-12)
as_registered = strategies.v2_sleeves(*v2_args)
assert as_registered["orderflow"].equals(registered["orderflow"])
moved = (as_registered["carry"] - registered["carry"]).abs() > 1e-12
holes = [("2022-02-26", "2022-03-12"), ("2022-04-01", "2022-04-14")]
in_holes = pd.Series(False, index=moved.index)
for a, b in holes:
    in_holes.loc[a:b] = True
assert not (moved & ~in_holes).any(), "the filled data changed Carry outside the weeks after the holes"
print(f"Carry days changed by the filled data: {int(moved.sum())}, from {moved[moved].index.min().date()} "
      f"to {moved[moved].index.max().date()}")
as_registered["v2"] = strategies.v2_book(as_registered)
v2_fixed = strategies.v2_sleeves(cut["taker_imbalance"], cut["returns"],
                                 cut["perp_returns_traded"].reindex(columns=u.columns),
                                 cut["funding"].reindex(columns=u.columns), u,
                                 pnl_funding=cut["funding_traded"].reindex(columns=u.columns))
v2_fixed["v2"] = strategies.v2_book(v2_fixed)
v2_table = pd.DataFrame({(label, col): {w: metrics.sharpe(s[col].loc[WINDOWS[w]].dropna()) for w in ["dev", "gate"]}
                         for label, s in [("as registered, first run", registered),
                                          ("registered rules, holes filled", as_registered),
                                          ("holes filled, held contracts' own data", v2_fixed)]
                         for col in ["v2", "orderflow", "carry"]}).T
v2_table.round(3)
# -

# ## What the checks say
#
# The data source is not the problem: rebuilt from the archive, the research
# coin list gives the same numbers (step 0b). Taking out the non-crypto
# assets costs the walk-forward book a little, 1.46 to 1.26 on the lockbox.
#
# The coin list is. On every pair (step 2), the frozen book's lockbox Sharpe is
# 0.35 walk-forward and -0.62 equal weight. Orderflow and Carry both lose money
# on the gate and the lockbox; Seasonality stays positive on dev and gate (1.57
# and 1.42, against 1.79 and 1.24 as run) and still loses on the lockbox. The
# research list was picked by volume in July 2026, and 46% of the universe's
# coin-days before then belonged to coins it left out (notebook 09). The coins
# that did the damage are mostly ones the list never had. Four of Orderflow's
# five worst coins on the gate were missing from it, OM above all, which
# collapsed in April 2025. All five of z-score Carry's were: delisting
# casualties such as VIDT and BNX, which it bought because their funding had
# turned negative.
#
# The other steps matter less, with one exception. Perp prices help Carry a
# little on the gate and the lockbox and hardly at all on dev. The first run of
# this notebook counted a day the perp data drops out as zero for a coin Carry
# held, funding included. Some of those days fell on two holes in the archive's
# own 2022 files, which notebook 09 now fills from its daily files. Of the 28
# held coin-days left, the contract traded on 8, among them LUNA's and FTT's
# crashes in 2022 and OMG's 27% discount to spot in November 2021, and those
# now get the contract's own move and funding; on the other 20 it did not
# trade, mostly because it had been delisted, and the position earns nothing.
# Together the two fixes take Carry's step-3 Sharpe from 0.40 to 0.25 on dev
# and from -1.58 to -1.51 on the gate. Limit orders that have to be filled hurt
# Orderflow even though about 99% of what it orders fills the next day. The
# misses are the days the price ran away: where a buy missed, the coin rose
# 7.1% that day on average, and where a sell missed it fell 5.7%. Missing 0.3%
# of the positions costs about 30 of Orderflow's 70 points of gross P&L on dev,
# and its dev Sharpe falls from 0.44 to 0.09. Even on the research list, most
# of Orderflow's edge is gone at $1M under the square-root impact model (0.30
# or -0.27, depending on the impact coefficient, from 0.87 with none); Carry
# lasts to somewhere between $1M and $10M. Notebook 05's hourly reversal, rerun
# on every pair, has a larger gross edge than on its 60 coins but still loses
# money after costs in every config. Charging the pairs baseline for its
# rebalance trades barely moves it, and neither does picking its pairs a day
# earlier, which removes its one-day look-ahead at each rebalance (dev -0.37
# against -0.40, gate 0.43 against 0.53).
#
# Put together, the lockbox year and the forward test give the frozen book,
# as run, a Sharpe of 0.37 over 453 days (t = 0.4). In the forward window
# the book's beta to BTC was lower than in 93% of earlier 82-day stretches,
# and Orderflow's lower than in 99%.
#
# The frozen book's lockbox result came mostly from its coin list. v2
# (notebook 10) was registered before these checks ran; it trades the
# every-pair universe with rank-weighted Carry on perps, and its test is the
# data from 2026-10-02 on. Its Carry sleeve counted the dropped days as zero,
# as the first run of step 3 did. With both fixes, which its test uses from
# the first day, v2 has a Sharpe of 1.35 on dev and 1.13 on the gate (1.39 and
# 1.12 as registered), and its Carry 1.68 and 2.42 (1.76 and 2.41).

# ## Save for the report

# +
series = pd.concat({labels[k]: df for k, df in step.items()}, axis=1, sort=True)
series.columns = [f"{a}|{b}" for a, b in series.columns]
series.to_parquet(PROC / "realism_steps.parquet")
pd.DataFrame(size).to_parquet(PROC / "realism_size.parquet")
capacity.rename_axis(["sleeve", "aum"]).to_csv(PROC / "realism_capacity.csv")
pd.DataFrame({"as run": base, "charged": pairs_charged, "picked a day earlier": pairs_lagged,
              "both": pairs_both}).to_parquet(PROC / "realism_pairs.parquet")
v2_fixed.to_parquet(PROC / "realism_v2.parquet")
pd.DataFrame({"dropped days at zero": at_zero, "held contracts' own data": s3["carry"]}).to_parquet(
    PROC / "realism_gaps.parquet")
record.rename_axis("book").to_csv(PROC / "realism_oos.csv")
pd.DataFrame(fills).T.rename_axis("sleeve").to_csv(PROC / "realism_fills.csv")
pd.DataFrame(beta_rows).T.rename_axis("series").to_csv(PROC / "realism_beta.csv")
fast.rename_axis(["panel", "lookback_h", "rebal_h"]).to_csv(PROC / "realism_fastrev.csv")
refits.reset_index().to_csv(PROC / "realism_refits.csv", index=False)
print("saved")
# -
