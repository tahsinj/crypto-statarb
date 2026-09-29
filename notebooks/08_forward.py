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

# # 08: Forward test on data after the freeze
#
# The research data ends on 2026-07-06. This notebook fetches the days since
# then and runs the frozen book on them: same coins, same universe rule, same
# sleeves, costs and walk-forward settings. Nothing is tuned and nothing is
# added to the trial registry. It is the only data the book has never seen.
#
# The window is short, 82 days, and a Sharpe ratio measured over 82 days has
# a standard error of about 2, so this is a check that the book keeps behaving
# as it did, not new evidence that it works.
#
# The research panels in `data/processed/` are not touched. New data goes to
# its own cache files and is joined to the cached research data on
# 2026-07-06, the day the original fetch caught only partly.

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from quantlib import backtest, data, fetch, metrics, robustness, signals, strategies

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

SPLICE = "2026-07-06"          # first day taken from the new fetch
FORWARD_START = "2026-07-07"   # first day the research data never had
END = "2026-09-26"             # last complete UTC day when this was run
OVERLAP = slice("2026-05-01", "2026-07-05")
FORWARD = slice(FORWARD_START, END)

# ## Fetch the new days for the same coins

research_raw = data.load_raw(RAW / "binance_1d.pkl.zip")
research_funding = data.load_raw(RAW / "funding.pkl.zip")
coins = sorted(set(research_raw.columns.get_level_values(0)))
perps = list(research_funding.columns)

new_raw = fetch.refresh(RAW / "forward_1d.pkl.zip",
                        lambda: fetch.fetch_binance(symbols=coins, start="2026-05-01", end=END),
                        max_age_days=float("inf"))
# funding runs to the next midnight so the last day has all its payments
new_funding = fetch.refresh(RAW / "forward_funding.pkl.zip",
                            lambda: fetch.fetch_funding(perps, start="2026-05-01", end="2026-09-27"),
                            max_age_days=float("inf")).loc[:END]
print("daily:", new_raw.shape, new_raw.index.min().date(), "->", new_raw.index.max().date())
print("funding:", new_funding.shape, new_funding.index.min().date(), "->", new_funding.index.max().date())
missing = sorted(set(coins) - set(new_raw.columns.get_level_values(0)))
print(f"{len(coins) - len(missing)} of {len(coins)} coins returned new data; missing: {missing}")
last_trade = {c: research_raw[c]["price"].dropna().index.max().date() for c in missing}
print("last close in the research data for the missing coins:", last_trade)

# The coins with no new data had all stopped trading on Binance by mid-April
# 2026, so nothing tradable is lost. Four more (PHB, D, HIGH and TON) stopped
# in May and June, before the forward window; their data here ends then.
# Coins listed after July are not added: the coin list stays the one the
# research used.

# ## Check the new fetch against the cached data where they overlap
# Past daily bars on Binance do not change, so every overlapping value
# should match exactly.

common = [c for c in research_raw.columns if c in new_raw.columns]
a = research_raw.loc[OVERLAP, common].to_numpy()
b = new_raw.loc[OVERLAP, common].reindex(research_raw.loc[OVERLAP].index).to_numpy()
daily_bad = int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum())
print(f"daily overlap: {int(np.isfinite(a).sum()):,} values compared, {daily_bad} differ")
fa = research_funding.loc[OVERLAP]
fb = new_funding.reindex(columns=fa.columns).loc[OVERLAP].reindex(fa.index)
funding_bad = int((~np.isclose(fa.to_numpy(), fb.to_numpy(), rtol=0, atol=0, equal_nan=True)).sum())
print(f"funding overlap: {int(np.isfinite(fa.to_numpy()).sum()):,} values compared, {funding_bad} differ")
assert daily_bad == 0 and funding_bad == 0

# ## Join the two and rebuild the panels with notebook 00's rules

raw = pd.concat([research_raw.loc[:"2026-07-05"], new_raw.loc[SPLICE:]]).sort_index()
funding_all = pd.concat([research_funding.loc[:"2026-07-05"],
                         new_funding.reindex(columns=perps).loc[SPLICE:]]).sort_index()
panels = data.to_panels(raw)
universe = data.build_universe(panels, top_n=100, min_adv_usd=1_000_000)
price, returns, taker = panels["price"], panels["returns"], panels["taker_imbalance"]
funding_all = funding_all.reindex(columns=universe.columns)
bench = pd.DataFrame({"BTC": returns["BTC"], "MKT": data.market_return(panels, universe)})
fwd = universe.loc[FORWARD].sum(axis=1)
print(f"universe over the forward window: {int(fwd.min())} to {int(fwd.max())} names")

# ## The frozen book

# +
seas = strategies.seasonal_momentum_sleeve(price, returns, universe)
orderflow = strategies.orderflow_sleeve(taker, returns, universe)
carry = strategies.carry_sleeve(funding_all, returns, universe)
seas_charged = strategies.seasonal_momentum_sleeve(price, returns, universe, charge_resizing=True)


def combine(seasonality):
    """Walk-forward and equal-weight books with notebook 06's settings."""
    sleeves = pd.concat([seasonality, orderflow, carry], axis=1, sort=True).dropna()
    wf, weights = robustness.walk_forward_weights(sleeves, train_days=756, step_days=63, min_train=252)
    return wf, sleeves.mean(axis=1), weights


wf, ew, weights = combine(seas)
wf_c, ew_c, _ = combine(seas_charged)
# -

# Up to the day before the splice, everything has to match the research
# exactly: same data, same code.

research = pd.read_parquet(PROC / "sleeves_full.parquet")
book = pd.read_parquet(PROC / "combined_portfolio.parquet")
checks = [("seasonality", seas, research["seasonality"]), ("orderflow", orderflow, research["orderflow"]),
          ("carry", carry, research["carry"]), ("walk-forward", wf, book["walk_forward"]),
          ("equal weight", ew, book["equal_weight"])]
for name, mine, theirs in checks:
    both = pd.concat([mine, theirs], axis=1, sort=True).loc[:"2026-07-05"].dropna()
    same = np.array_equal(both.iloc[:, 0], both.iloc[:, 1])
    print(f"{name}: {len(both)} days, identical to the research up to 2026-07-05: {same}")
    assert same

# ## Forward results (2026-07-07 to 2026-09-26)

# +
def stats(r):
    r = r.loc[FORWARD].dropna()
    return {"days": len(r), "total_return": (1 + r).prod() - 1, "ann_vol": metrics.ann_vol(r),
            "sharpe": metrics.sharpe(r), "max_dd": metrics.max_drawdown(r)}


series = {"book, walk-forward": wf, "book, equal weight": ew,
          "book, walk-forward (resizing charged)": wf_c, "book, equal weight (resizing charged)": ew_c,
          "seasonality": seas, "orderflow": orderflow, "carry": carry,
          "BTC": bench["BTC"], "equal-weight market": bench["MKT"]}
forward = pd.DataFrame({k: stats(s) for k, s in series.items()}).T
forward
# -

# Beta of the walk-forward book to BTC over the window, and the rough
# standard error of an 82-day Sharpe estimate:

fw = pd.concat([wf, bench["BTC"]], axis=1, sort=True).loc[FORWARD].dropna()
beta_btc = np.cov(fw.iloc[:, 0], fw.iloc[:, 1])[0, 1] / fw.iloc[:, 1].var()
print(f"walk-forward beta to BTC over the window: {beta_btc:+.3f}")
print(f"standard error of an annualised Sharpe over {len(fw)} days: about {np.sqrt(365 / len(fw)):.1f}")

# ## Where the walk-forward book's P&L came from
#
# The weights in force over the window (refitted on 2026-06-02 and
# 2026-08-04), and each sleeve's share of the book's P&L. Shares are sums of
# daily returns, which add up exactly; compounded returns do not.

# +
print(weights.loc[FORWARD].drop_duplicates().round(3))
book_split = (weights * pd.concat([seas, orderflow, carry], axis=1, sort=True)).loc[FORWARD].dropna()
assert np.allclose(book_split.sum(axis=1), wf.loc[FORWARD])
book_split = book_split.sum().rename("sum of daily returns")
book_split["total"] = book_split.sum()
book_split
# -

# How much of that is the market, and how unusual Orderflow's stretch was.
# With an intercept in the regression, the book's summed return splits
# exactly into beta times BTC's summed return plus the rest.

btc_sum = bench["BTC"].loc[FORWARD].sum()
print(f"BTC, sum of daily returns: {btc_sum:+.4f}")
print(f"walk-forward book: {wf.loc[FORWARD].sum():+.4f}, of which beta x BTC {beta_btc * btc_sum:+.4f}")
of_82 = orderflow.loc["2020-01-01":"2026-07-05"].rolling(82).sum().dropna()
of_fwd = orderflow.loc[FORWARD].sum()
print(f"Orderflow over the window: {of_fwd:+.4f}; 82-day stretches since 2020 that were worse: "
      f"{(of_82 < of_fwd).mean():.1%}; worst {of_82.min():+.4f}, ending {of_82.idxmin().date()}")

# ## Carry and one coin
#
# Carry's weights are rebuilt with the same calls as `strategies.carry_sleeve`
# so its P&L can be split by coin. The rebuild has to reproduce the sleeve.

# +
fuv = universe & funding_all.notna()
w_carry = signals.signal_to_weights(-signals.cross_sectional_zscore(funding_all.rolling(7).mean(), fuv),
                                    fuv, long_short=True)
held = w_carry.shift(1)
by_coin = (held * returns).fillna(0.0) - (held * funding_all).fillna(0.0)   # price leg + funding leg
carry_costs = backtest.run(w_carry, returns, cost_bps=7).costs
assert np.allclose((by_coin.sum(axis=1) - carry_costs).loc[FORWARD], carry.loc[FORWARD])

coin_pnl = by_coin.loc[FORWARD].sum().sort_values(ascending=False)
top = coin_pnl.index[0]
carry_split = pd.Series({top: coin_pnl[top], "all other coins": coin_pnl.drop(top).sum(),
                         "trading costs": -carry_costs.loc[FORWARD].sum()}, name="sum of daily returns")
carry_split["carry, net"] = carry_split.sum()
print("largest contributions:", coin_pnl.head(5).round(3).to_dict())
print("smallest contributions:", coin_pnl.tail(5).round(3).to_dict())
carry_split
# -

# One coin made all of it; the other coins together lost a little.

dollar_volume = panels["dollar_volume"][top]
print(f"{top} dollar volume on 2026-07-21: "
      f"{dollar_volume.loc['2026-07-21'] / dollar_volume.loc['2026-06-21':'2026-07-20'].median():.1f}"
      " times its median over the previous 30 days")
detail = pd.DataFrame({"close": price[top], "return": returns[top], "funding (daily sum)": funding_all[top],
                       "carry weight held": held[top]}).loc["2026-07-18":"2026-08-03"]
detail.round(4)

# DEXE fell 82.5% on 2026-07-21, on more than 16 times its usual volume, and
# kept swinging hard for a week. Shorts piled into the perp and funding went
# deeply negative: shorts paid longs 13% to 21% a day for three days. A
# signal that shorts high funding and buys low funding reads that as the best
# long in the market. The z-score weights have no cap per coin, so from
# 2026-07-22 the sleeve held DEXE at 47% to 50% of its gross, all or nearly
# all of its long side, for twelve days.
#
# The weighting always allowed this. The table shows how concentrated
# Carry's weights were in each window.

# +
top_weight = w_carry.abs().max(axis=1)
windows = {"dev": slice("2020-01-01", "2024-07-31"), "gate": slice("2024-08-01", "2025-06-30"),
           "lockbox": slice("2025-07-01", "2026-07-05"), "forward": FORWARD}
concentration = pd.DataFrame({
    k: {"days": len(top_weight.loc[sl]),
        "median largest weight": top_weight.loc[sl].median(),
        "share of days largest >= 0.25": (top_weight.loc[sl] >= 0.25).mean(),
        "days one coin holds a whole side": int((top_weight.loc[sl] >= 0.4999).sum())}
    for k, sl in windows.items()}).T
concentration
# -

# Which coin held a whole side, by month, and how many coins Carry could
# trade in January 2020:

whole_side = top_weight.loc["2020-01-01":END]
whole_side = whole_side[whole_side >= 0.4999]
print(pd.Series(w_carry.loc[whole_side.index].abs().idxmax(axis=1).to_numpy(),
                index=whole_side.index.to_period("M")).groupby(level=0).value_counts().to_string())
n_carry = fuv.sum(axis=1).loc["2020-01-01":"2020-01-31"]
print(f"coins with funding data in the universe, January 2020: {n_carry.min()} to {n_carry.max()}")

# In the research, one coin held a whole side on 23 days, all on dev. Sixteen
# were in January 2020, when only 3 to 9 coins had funding data. The other
# seven were in November 2022, when the sleeve held SOL as its whole long side
# during the FTX collapse, the one earlier case like this. In the 82 forward
# days it happened on 10, all DEXE, and the median largest weight was 0.27,
# against 0.17 to 0.20 in the research windows.

# ## What this says
#
# Over the 82 days the walk-forward book lost 5.9% (Sharpe -1.01) while BTC
# rose 32%. Most of the loss is Orderflow, which carried 69% to 80% of the
# weight and lost 10.8%. Orderflow was the best sleeve on the lockbox
# (+1.95). About one 82-day stretch in twenty since 2020 was worse, so runs
# like this have happened before. Part of it was the market: over this
# window the book's beta to BTC was -0.12, against about zero in the
# research, and beta times BTC's rally accounts for about 3.5 of the 5.5
# points the book lost (in sums of daily returns).
#
# The equal-weight book made 5.1%, but only because of Carry, and Carry's
# 11.3% came from DEXE. The other coins together lost a little before costs.
# The DEXE gain is also the part of the backtest to trust least: funding that
# deep comes with the perp trading far below spot, and the backtest adds
# funding to spot returns and ignores that gap.
#
# Seasonality made 2.6%. With its resizing trades charged, the walk-forward
# book loses 7.3% instead of 5.9%.
#
# So the forward test does not support the lockbox result. It does not
# contradict it either: with a standard error of about 2, an 82-day Sharpe of
# -1.01 fits a true Sharpe anywhere from about -5 to +3. What it adds is a
# clear case of a weakness the research showed only once, with SOL in 2022:
# Carry's uncapped z-score weights can put half the sleeve into one coin in
# the middle of a crash. A per-coin cap is the obvious fix, but picking one
# now would be tuning on this window, so the book stays as it was frozen.

# ## Save for the report

# +
pd.DataFrame({"seasonality": seas, "orderflow": orderflow, "carry": carry,
              "walk_forward": wf, "equal_weight": ew, "walk_forward_charged": wf_c,
              "equal_weight_charged": ew_c, "BTC": bench["BTC"], "MKT": bench["MKT"]}
             ).loc["2026-01-01":END].to_parquet(PROC / "forward_book.parquet")
pd.concat({"book": book_split, "carry": carry_split}).rename_axis(["part", "item"]).to_csv(
    PROC / "forward_attribution.csv")
concentration.rename_axis("window").to_csv(PROC / "forward_concentration.csv")
detail.rename_axis("date").assign(coin=top).to_csv(PROC / "forward_top_coin.csv")
print("saved forward_book.parquet and the attribution, concentration and top-coin tables")
# -
