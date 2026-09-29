"""Build the PDF research report, ``reports/REPORT.pdf``.

The report's results are computed here from the artifacts in data/processed/
(written by notebooks 00-11), from the twsq CSVs in alphas/, or, for the
fast-reversal grid, read from notebook 05's executed output. The script checks
the headline numbers against the values the notebooks printed, and the numbers
in README.md and alphas/README.md against the same results, and stops without
writing the PDF if anything disagrees. Run it after the notebooks:

    .venv/bin/python reports/build_report.py           # check, then write the PDF
    .venv/bin/python reports/build_report.py --check   # check only, and that REPORT.pdf is up to date

The PDF is written with fpdf2 (pure Python).
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# importing plotting also sets the chart style
from quantlib import backtest, metrics, plotting, robustness, signals, strategies  # noqa: E402,F401

PROC = ROOT / "data" / "processed"
ALPHAS = ROOT / "alphas"
OUT = ROOT / "reports" / "REPORT.pdf"
# The PDF carries a fixed creation date, the end of the day its content last
# changed, so the same inputs give the same file byte for byte (--check relies
# on it). Move it forward whenever the content changes.
BUILD_DATE = datetime(2026, 9, 29, 23, 59, tzinfo=timezone.utc)

DEV_START, DEV_END = "2020-01-01", "2024-07-31"
GATE_START, GATE_END = "2024-08-01", "2025-06-30"
LOCKBOX_START, LOCKBOX_END = "2025-07-01", "2026-07-06"
WINDOWS = {
    "Dev": slice(DEV_START, DEV_END),
    "Gate": slice(GATE_START, GATE_END),
    "Lockbox": slice(LOCKBOX_START, LOCKBOX_END),
    "Full": slice(DEV_START, LOCKBOX_END),
}
# Notebook 08: the frozen book on data fetched after the research ended.
FORWARD_START, FORWARD_END = "2026-07-07", "2026-09-26"
FORWARD = slice(FORWARD_START, FORWARD_END)
RWIN = {"Dev": slice(DEV_START, DEV_END), "Gate": slice(GATE_START, GATE_END),
        "Lockbox": slice(LOCKBOX_START, LOCKBOX_END), "Forward": FORWARD}
ALL = PROC / "all_pairs"
N_TRIALS = 48          # research trials in the registry (rows 1-48); rows 49-50 are the two books
FIRST_VERSION_TRIALS = 26   # configurations the project's first version searched (not in the registry)
BOOT = dict(n_boot=2000, method="block", seed=42)   # same settings as notebooks 06 and 07

# fpdf2's core fonts are Latin-1 only.
_SUBS = {"−": "-", "≈": "~", "×": "x", "·": ".", "‘": "'", "’": "'", "“": '"', "”": '"',
         "σ": "sigma", "≥": ">=", "≤": "<="}


def _latin1(text: str) -> str:
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def pts(x: float) -> str:
    """A sum of daily returns in percentage points."""
    return f"{x * 100:.1f}"


def sr(x: float) -> str:
    x = round(x, 2) + 0.0          # no "-0.00"
    return f"{x:+.2f}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_inputs() -> dict:
    need = ["combined_portfolio", "benchmarks", "sleeve_baseline_momentum",
            "sleeve_baseline_reversal", "baselines_full", "sleeve_seasonality",
            "sleeve_orderflow", "sleeve_carry", "sleeves_full", "posthoc_book", "price",
            "returns", "universe", "taker_imbalance", "funding", "price_1h", "forward_book"]
    missing = [n for n in need if not (PROC / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {PROC}. Run notebooks 00-11 first.")
    d = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in need}
    d["registry"] = pd.read_csv(PROC / "trial_registry.csv")
    d["cost_stress"] = pd.read_csv(PROC / "posthoc_cost_stress.csv")
    d["carry_split"] = pd.read_csv(PROC / "posthoc_carry_split.csv").set_index("window")
    d["calendar"] = pd.read_csv(PROC / "posthoc_calendar.csv").set_index(["test", "timing"])
    d["drift"] = pd.read_csv(PROC / "posthoc_drift.csv").set_index("sleeve")
    d["rebalance"] = pd.read_csv(PROC / "posthoc_rebalance.csv").set_index("book")
    d["refits"] = pd.read_csv(PROC / "realism_refits.csv").set_index(["step", "days later"])
    d["fwd_split"] = pd.read_csv(PROC / "forward_attribution.csv").set_index(["part", "item"]).iloc[:, 0]
    d["fwd_conc"] = pd.read_csv(PROC / "forward_concentration.csv").set_index("window")
    d["fwd_coin"] = pd.read_csv(PROC / "forward_top_coin.csv", parse_dates=["date"]).set_index("date")
    # notebooks 09 to 11: the every-pair universe, the realism checks and v2
    for name in ["realism_steps", "realism_size", "realism_pairs", "realism_gaps", "realism_v2", "v2_dev_gate"]:
        d[name] = pd.read_parquet(PROC / f"{name}.parquet")
    d["realism_capacity"] = pd.read_csv(PROC / "realism_capacity.csv").set_index(["sleeve", "aum"])
    for name, key in [("realism_oos", "book"), ("realism_fills", "sleeve"), ("realism_beta", "series")]:
        d[name] = pd.read_csv(PROC / f"{name}.csv").set_index(key)
    d["realism_fastrev"] = pd.read_csv(PROC / "realism_fastrev.csv").set_index(["panel", "lookback_h", "rebal_h"])
    d["all_universe"] = pd.read_parquet(ALL / "universe.parquet")
    d["all_bench"] = pd.read_parquet(ALL / "benchmarks.parquet")
    return d


def fastrev_grid() -> pd.DataFrame:
    """The gross-vs-cost grid as printed by notebook 05 (its gross returns are not saved)."""
    nb = json.loads((ROOT / "notebooks" / "05_fastrev.ipynb").read_text())
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", ""))
            if "gross_over_drag" not in text:
                continue
            lines = text.strip().splitlines()
            cols = lines[0].split()
            rows, lb = [], None
            for line in lines[1:]:
                tok = line.split()
                if len(tok) == len(cols) + 2:
                    lb, tok = int(tok[0]), tok[1:]
                rows.append([lb, int(tok[0])] + [float(t) for t in tok[1:-1]] + [tok[-1] == "True"])
            return pd.DataFrame(rows, columns=["lookback_h", "rebal_h"] + cols)
    raise ValueError("fast-reversal grid not found in notebook 05 output")


def twsq_results() -> pd.DataFrame:
    rows = {}
    for name in ["SeasonalMomentum", "OrderflowFollow", "FundingCarry"]:
        stats = pd.read_csv(ALPHAS / name / "stats.csv").iloc[0]
        pnl = pd.read_csv(ALPHAS / name / "backtest" / "pos_pnl.csv")["pnl"]
        check = pnl.mean() / pnl.std() * np.sqrt(365)
        assert abs(check - stats["sharpe"]) < 1e-9, f"{name}: stats.csv out of date"
        rows[name] = stats
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def perf(r: pd.Series, bench: pd.DataFrame, regress: bool = True) -> dict:
    r = r.dropna()
    out = {"ret": metrics.ann_return(r), "vol": metrics.ann_vol(r), "sharpe": metrics.sharpe(r),
           "mdd": metrics.max_drawdown(r), "n": len(r), "start": r.index.min()}
    if regress:
        ab = metrics.alpha_beta(r, bench["BTC"], bench["MKT"], names=["BTC", "MKT"])
        out.update(beta=ab["beta_BTC"], beta_mkt=ab["beta_MKT"], alpha=ab["alpha_ann"],
                   alpha_t=ab["alpha_tstat"])
    return out


def fwd_perf(r: pd.Series, btc: pd.Series) -> dict:
    """Stats for the 82-day forward window: total return, not an annualised one."""
    both = pd.concat([r, btc], axis=1, sort=True).dropna()
    r, b = both.iloc[:, 0], both.iloc[:, 1]
    return {"total": (1 + r).prod() - 1, "vol": metrics.ann_vol(r), "sharpe": metrics.sharpe(r),
            "mdd": metrics.max_drawdown(r), "beta": np.cov(r, b)[0, 1] / b.var(), "n": len(r)}


def check(label: str, got: float, expected: float, tol: float, failures: list) -> None:
    if not abs(got - expected) <= tol:
        failures.append(f"  {label}: got {got:.4f}, expected {expected:.4f} (tol {tol})")


def same_series(a: pd.Series, b: pd.Series) -> bool:
    both = pd.concat([a, b], axis=1, sort=True).dropna()
    return len(both) > 0 and np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=0, atol=1e-12)


def compute(d: dict) -> dict:
    bench = d["benchmarks"]
    cp = d["combined_portfolio"]
    wf, ew = cp["walk_forward"], cp["equal_weight"]
    sleeves = d["sleeves_full"]
    bases = d["baselines_full"]
    two = sleeves[["orderflow", "carry"]].loc[:LOCKBOX_END].dropna()
    wf_two, _ = robustness.walk_forward_weights(two, train_days=756, step_days=63, min_train=252)
    v_no_seas = {"wf": metrics.sharpe(wf_two.loc[LOCKBOX_START:LOCKBOX_END].dropna()),
                 "ew": metrics.sharpe(two.mean(axis=1).loc[LOCKBOX_START:LOCKBOX_END])}
    ph = d["posthoc_book"]
    failures: list[str] = []

    # sleeves_full and baselines_full (notebook 07) must match what notebooks
    # 01-04 saved up to the gate, and sleeves_full must rebuild notebook 06's
    # equal-weight book.
    for name in ["seasonality", "orderflow", "carry"]:
        if not same_series(sleeves[name].loc[DEV_START:GATE_END], d[f"sleeve_{name}"][name].loc[DEV_START:GATE_END]):
            failures.append(f"  sleeves_full[{name}] differs from sleeve_{name}.parquet")
    for name in ["momentum", "reversal"]:
        if not same_series(bases[name].loc[:GATE_END], d[f"sleeve_baseline_{name}"][name]):
            failures.append(f"  baselines_full[{name}] differs from sleeve_baseline_{name}.parquet")
    ew_check = pd.concat([sleeves.dropna().mean(axis=1), ew], axis=1, sort=True).dropna()
    if not np.array_equal(ew_check.iloc[:, 0].to_numpy(), ew_check.iloc[:, 1].to_numpy()):
        failures.append("  sleeves_full does not reproduce the equal-weight book")

    reg = d["registry"]
    fam = reg["family"]
    if not ((~fam.isin(["portfolio", "v2"])).sum() == N_TRIALS and (fam == "portfolio").sum() == 2
            and (fam == "v2").sum() == 3):
        failures.append(f"  registry has {len(reg)} rows; expected {N_TRIALS} trials, 2 books and 3 v2 rows")

    v: dict = {"no_seas": v_no_seas}
    v["book"] = {(b, w): perf(s.loc[sl], bench) for b, s in [("WF", wf), ("EW", ew)]
                 for w, sl in WINDOWS.items()}
    v["sleeve"] = {(n, w): perf(sleeves[n].loc[sl], bench) for n in sleeves.columns
                   for w, sl in WINDOWS.items() if w != "Full"}
    lk, full = wf.loc[WINDOWS["Lockbox"]].dropna(), wf.loc[WINDOWS["Full"]].dropna()
    v["lk_n"] = len(lk)
    v["dsr_full"] = metrics.deflated_sharpe(full, n_trials=N_TRIALS)
    v["dsr_lk"] = metrics.deflated_sharpe(lk, n_trials=N_TRIALS)
    v["dsr_lk_all"] = metrics.deflated_sharpe(lk, n_trials=N_TRIALS + FIRST_VERSION_TRIALS)
    v["ci"] = metrics.bootstrap_sharpe_ci(lk, **BOOT)
    v["corr"] = sleeves.loc[WINDOWS["Dev"]].corr()
    # The walk-forward book starts in 2021-10, so compare equal weight over the same dates too.
    v["same_dates"] = {}
    for w in WINDOWS:
        dates = wf.loc[WINDOWS[w]].dropna().index
        v["same_dates"][w] = {"wf": metrics.sharpe(wf.loc[dates]), "ew": metrics.sharpe(ew.loc[dates])}
    v["dsr_wf"] = {w: metrics.deflated_sharpe(wf.loc[sl].dropna(), n_trials=N_TRIALS) for w, sl in WINDOWS.items()}
    v["dsr_lk_ew"] = metrics.deflated_sharpe(ew.loc[WINDOWS["Lockbox"]].dropna(), n_trials=N_TRIALS)
    sd = v["same_dates"]
    if not (all(sd[w]["ew"] > sd[w]["wf"] for w in ["Dev", "Gate", "Full"]) and sd["Lockbox"]["wf"] > sd["Lockbox"]["ew"]):
        failures.append("  section 4 says equal weight beats walk-forward on every window but the lockbox")
    if not all(x < 0.95 for x in v["dsr_wf"].values()):
        failures.append("  section 9 says the walk-forward book's deflated Sharpe is below 0.95 on every window")

    # Every strategy and both benchmarks over the whole sample.
    full_sl = WINDOWS["Full"]
    v["full"] = {
        "Baseline momentum": perf(bases["momentum"].loc[full_sl], bench),
        "Baseline pairs": perf(bases["reversal"].loc[full_sl], bench),
        "Seasonality": perf(sleeves["seasonality"].loc[full_sl], bench),
        "Orderflow": perf(sleeves["orderflow"].loc[full_sl], bench),
        "Carry": perf(sleeves["carry"].loc[full_sl], bench),
        "Book, walk-forward": v["book"][("WF", "Full")],
        "Book, equal weight": v["book"][("EW", "Full")],
        "BTC": perf(bench["BTC"].loc[full_sl], bench, regress=False),
        "Equal-weight market": perf(bench["MKT"].loc[full_sl], bench, regress=False),
    }

    # Market-timing regressions for the two trend sleeves on dev.
    dev = WINDOWS["Dev"]
    v["tm"] = {"momentum": metrics.treynor_mazuy(bases["momentum"].loc[dev], bench["MKT"].loc[dev]),
               "seasonality": metrics.treynor_mazuy(sleeves["seasonality"].loc[dev], bench["MKT"].loc[dev])}

    # Carry: alpha scaled to 10% volatility, its gate interval, its best months.
    cs = {}
    for w in ["Gate", "Lockbox"]:
        s = v["sleeve"][("carry", w)]
        se = s["alpha"] / s["alpha_t"]
        cs[w] = {"alpha": s["alpha"], "vol": s["vol"], "at10": s["alpha"] * 0.10 / s["vol"],
                 "lo": s["alpha"] - 1.96 * se, "hi": s["alpha"] + 1.96 * se}
    months = (1 + sleeves["carry"].loc[WINDOWS["Gate"]]).resample("ME").prod() - 1
    cs["best3"] = (1 + months.sort_values(ascending=False).head(3)).prod() - 1
    cs["gate_total"] = (1 + months).prod() - 1
    v["carry_scale"] = cs

    # Post-hoc re-score (notebook 07).
    wf_c, ew_c = ph["walk_forward_charged"], ph["equal_weight_charged"]
    v["seas_c"] = {w: metrics.sharpe(ph["seasonality_charged"].loc[sl].dropna())
                   for w, sl in WINDOWS.items() if w != "Full"}
    v["resize_cost"] = (sleeves["seasonality"] - ph["seasonality_charged"]).loc[WINDOWS["Dev"]].mean() * 365
    v["book_c"] = {(b, w): perf(s.loc[sl], bench) for b, s in [("WF", wf_c), ("EW", ew_c)]
                   for w, sl in WINDOWS.items()}
    lk_c, full_c = wf_c.loc[WINDOWS["Lockbox"]].dropna(), wf_c.loc[WINDOWS["Full"]].dropna()
    v["dsr_full_c"] = metrics.deflated_sharpe(full_c, n_trials=N_TRIALS)
    v["dsr_lk_c"] = metrics.deflated_sharpe(lk_c, n_trials=N_TRIALS)
    v["ci_c"] = metrics.bootstrap_sharpe_ci(lk_c, **BOOT)
    charged = sleeves.assign(seasonality=ph["seasonality_charged"])
    v["lk_weight"] = {}
    for label, X in [("as run", sleeves), ("charged", charged)]:
        _, wp = robustness.walk_forward_weights(X.dropna(), train_days=756, step_days=63, min_train=252)
        v["lk_weight"][label] = wp.loc[WINDOWS["Lockbox"]].mean()

    # The untilted momentum sleeve with and without resizing costs, and the
    # turnover of the two limit-order sleeves (dev and gate data only).
    upto = slice(None, GATE_END)
    price, rets, uni = d["price"].loc[upto], d["returns"].loc[upto], d["universe"].loc[upto]
    for key, charge in [("untilted", False), ("untilted_c", True)]:
        s = strategies.seasonal_momentum_sleeve(price, rets, uni, weekend=1.0, weekday=1.0,
                                                charge_resizing=charge)
        v[key] = {w: metrics.sharpe(s.loc[WINDOWS[w]].dropna()) for w in ["Dev", "Gate"]}
    imb = d["taker_imbalance"].loc[upto]
    w_of = signals.signal_to_weights(
        signals.cross_sectional_zscore((imb - 0.5).rolling(10).mean(), uni), uni, long_short=True)
    fund = d["funding"].loc[upto].reindex(columns=uni.columns)
    fuv = uni & fund.notna()
    w_ca = signals.signal_to_weights(
        -signals.cross_sectional_zscore(fund.rolling(7).mean(), fuv), fuv, long_short=True)
    v["turnover"] = {name: backtest.run(w, rets, cost_bps=7).turnover.loc[WINDOWS["Dev"]].mean()
                     for name, w in [("orderflow", w_of), ("carry", w_ca)]}

    # Baselines (notebook 01) and the selected configs (registry).
    v["base"] = {(n, w): perf(d[f"sleeve_baseline_{n}"][n].loc[WINDOWS[w]], bench)
                 for n in ["momentum", "reversal"] for w in ["Dev", "Gate"]}

    def reg_sharpes(family: str, match: dict) -> tuple[float, float]:
        for _, row in reg[reg["family"] == family].iterrows():
            cfg = json.loads(row["config"])
            if all(cfg.get(k) == val for k, val in match.items()):
                return float(row["dev_sharpe"]), float(row["gate_sharpe"])
        raise ValueError(f"no registry row for {family} {match}")

    v["sel_seas"] = reg_sharpes("seasonality", {"probe": "P1", "weekday": 0.5, "weekend": 1.0})
    v["sel_untilted"] = reg_sharpes("seasonality", {"probe": "P1", "weekday": 1.0, "weekend": 1.0})
    v["sel_of"] = reg_sharpes("orderflow", {"probe": "P3", "dir": "follow", "smooth": 10})
    v["sel_carry"] = reg_sharpes("carry", {"probe": "P1", "smooth": 7})

    tw = twsq_results()
    v["twsq"] = tw
    if tw["n_days"].astype(int).nunique() != 1:
        failures.append("  the twsq runs cover different numbers of days")
    v["twsq_days"] = int(tw["n_days"].iloc[0])

    # Forward test (notebook 08). Its series must match the research up to the
    # day before the splice, and refitting the walk-forward weights here on
    # the joined sleeves must rebuild its book.
    fb = d["forward_book"]
    before = slice(fb.index.min(), "2026-07-05")
    for name, ref in [("seasonality", sleeves["seasonality"]), ("orderflow", sleeves["orderflow"]),
                      ("carry", sleeves["carry"]), ("walk_forward", wf), ("equal_weight", ew)]:
        if not same_series(fb[name].loc[before], ref.loc[before]):
            failures.append(f"  forward_book[{name}] differs from the research before 2026-07-06")
    joined = pd.concat([sleeves.loc[:"2026-07-05"], fb[list(sleeves.columns)].loc["2026-07-06":]]).dropna()
    wf_fwd, wp_fwd = robustness.walk_forward_weights(joined, train_days=756, step_days=63, min_train=252)
    if not same_series(wf_fwd.loc[FORWARD], fb["walk_forward"].loc[FORWARD]):
        failures.append("  refitted walk-forward weights do not rebuild notebook 08's book")
    fwd = fb.loc[FORWARD]
    v["fwd"] = {label: fwd_perf(fwd[col], fwd["BTC"]) for label, col in [
        ("Book, walk-forward", "walk_forward"), ("Book, equal weight", "equal_weight"),
        ("Walk-forward, resizing charged", "walk_forward_charged"),
        ("Equal weight, resizing charged", "equal_weight_charged"),
        ("Seasonality", "seasonality"), ("Orderflow", "orderflow"), ("Carry", "carry"),
        ("BTC", "BTC"), ("Equal-weight market", "MKT")]}
    v["fwd_of_weight"] = wp_fwd["orderflow"].loc[FORWARD].agg(["min", "max"])
    v["fwd_wf_sum"] = fwd["walk_forward"].sum()
    v["fwd_beta_part"] = v["fwd"]["Book, walk-forward"]["beta"] * fwd["BTC"].sum()
    of_82 = sleeves["orderflow"].loc[DEV_START:"2026-07-05"].rolling(82).sum().dropna()
    v["fwd_of_worse"] = (of_82 < fwd["orderflow"].sum()).mean()
    fsplit, conc, coin = d["fwd_split"], d["fwd_conc"], d["fwd_coin"]
    if not (abs(fsplit[("book", "total")] - v["fwd_wf_sum"]) < 1e-9
            and abs(fsplit[("carry", "carry, net")] - fwd["carry"].sum()) < 1e-9):
        failures.append("  forward_attribution.csv does not add up to forward_book.parquet")
    held = coin["carry weight held"]
    v["fwd_coin"] = {"name": coin["coin"].iloc[0], "crash": coin.loc["2026-07-21", "return"],
                     "funding": coin.loc["2026-07-21":"2026-07-23", "funding (daily sum)"],
                     "held": held[held >= 0.47]}
    # The days one coin held a whole side of Carry on dev, from the research panels.
    top_w = w_ca.abs().max(axis=1).loc[WINDOWS["Dev"]]
    whole = w_ca.loc[top_w.index[top_w >= 0.4999]].abs().idxmax(axis=1)
    v["whole"] = {"dev": len(whole), "jan20": len(whole.loc["2020-01"]),
                  "nov22": len(whole.loc["2022-11"]), "nov22_coins": set(whole.loc["2022-11"]),
                  "n_jan20": fuv.sum(axis=1).loc["2020-01"].agg(["min", "max"])}

    # Realism checks (notebooks 09 and 11). Step 0 has to be the research
    # series, and the pairs "as run" series the notebook 07 baseline.
    rs = d["realism_steps"]
    v["steps"] = {tuple(c.split("|")): {w: metrics.sharpe(rs[c].loc[sl].dropna()) for w, sl in RWIN.items()}
                  for c in rs.columns}
    for name, ref in [("walk_forward", wf), ("equal_weight", ew), ("carry", sleeves["carry"])]:
        if not same_series(rs[f"0 as run|{name}"].loc[:LOCKBOX_END], ref.loc[:LOCKBOX_END]):
            failures.append(f"  realism step 0 {name} is not the research series")
    sz = d["realism_size"]
    v["size"] = {tuple(c.split("|")): {w: metrics.sharpe(sz[c].loc[sl].dropna()) for w, sl in RWIN.items()}
                 for c in sz.columns}
    rg = d["realism_gaps"]
    v["gaps"] = {c: {w: metrics.sharpe(rg[c].loc[sl].dropna()) for w, sl in RWIN.items()} for c in rg.columns}
    if not same_series(rg["held contracts' own data"], rs["3 carry on perps|carry"]):
        failures.append("  realism_gaps does not hold step 3's Carry")
    rp = d["realism_pairs"]
    if not same_series(rp["as run"], bases["reversal"]):
        failures.append("  realism_pairs 'as run' differs from baselines_full")
    v["pairs_c"] = {c: {w: metrics.sharpe(rp[c].loc[RWIN[w]].dropna()) for w in ["Dev", "Gate", "Lockbox"]}
                    for c in rp.columns}
    cal = d["calendar"]
    base_seas = v["sel_untilted"]
    for test in cal.index.get_level_values(0).unique():
        meant = cal.loc[test].iloc[1]                    # the timing the test meant
        if meant["dev"] > max(0, base_seas[0]) and meant["gate"] > max(0, base_seas[1]):
            failures.append(f"  section 3.1 says the calendar tests still fail on the right days, but {test} passes")
    v["calendar"], v["drift"], v["rebalance"] = cal, d["drift"], d["rebalance"]
    rf = d["refits"]
    v["refits"] = {st_: (rf.loc[st_, "lockbox"].min(), rf.loc[st_, "lockbox"].max()) for st_ in ["0 as run", "2 every pair"]}
    for st_ in rf.index.get_level_values(0).unique():
        if not np.isclose(rf.loc[(st_, 0), "lockbox"], v["steps"][(st_, "walk_forward")]["Lockbox"], rtol=0, atol=1e-9):
            failures.append(f"  realism_refits at 0 days later is not step {st_}'s walk-forward book")
    if not v["refits"]["2 every pair"][1] < v["refits"]["0 as run"][0]:
        failures.append("  section 5.2 says the research book beats the every-pair book for any refit dates")
    if not all(p["Dev"] < 0 for p in v["pairs_c"].values()):
        failures.append("  section 5.6 says pairs loses money on dev in every version")
    oos = {k: rs[f"0 as run|{k}"].loc[LOCKBOX_START:FORWARD_END].dropna() for k in ["walk_forward", "equal_weight"]}
    v["oos"] = {"n": len(oos["walk_forward"]), "sharpe": metrics.sharpe(oos["walk_forward"]),
                "ci": metrics.bootstrap_sharpe_ci(oos["walk_forward"], **BOOT),
                "sharpe_ew": metrics.sharpe(oos["equal_weight"])}
    v["oos"]["t"] = v["oos"]["sharpe"] * np.sqrt(v["oos"]["n"] / 365)
    ua = d["all_universe"].loc[DEV_START:LOCKBOX_END]
    outside = ~ua.columns.isin(d["price"].columns)
    v["surv"] = ua.loc[:, outside].to_numpy().sum() / ua.to_numpy().sum()
    share = ua.loc[:, outside].sum(axis=1) / ua.sum(axis=1)
    v["surv_year"] = share.groupby(share.index.year).mean()
    v["fills"], v["cap"], v["fwd_beta"] = d["realism_fills"], d["realism_capacity"], d["realism_beta"]
    v["n_all"] = d["all_universe"].shape[1]

    # v2 (notebook 10), on dev and gate only, against the every-pair benchmarks.
    v2d = d["v2_dev_gate"]
    v["v2"] = {(n, w): perf(v2d[c].loc[RWIN[w]], d["all_bench"])
               for n, c in [("Book", "v2"), ("Orderflow", "orderflow"), ("Carry", "carry")] for w in ["Dev", "Gate"]}
    for key, c in [('"sleeve": "orderflow"', "orderflow"), ('"sleeve": "carry"', "carry"),
                   ('"rule": "equal_weight"', "v2")]:
        row = reg[(reg["family"] == "v2") & reg["config"].str.contains(key, regex=False)].iloc[0]
        if not (abs(row["dev_sharpe"] - metrics.sharpe(v2d[c].loc[RWIN["Dev"]].dropna())) < 1e-9
                and abs(row["gate_sharpe"] - metrics.sharpe(v2d[c].loc[RWIN["Gate"]].dropna())) < 1e-9):
            failures.append(f"  v2 registry row for {c} does not match v2_dev_gate.parquet")
    if v2d.index.max() > pd.Timestamp(GATE_END):
        failures.append("  v2_dev_gate.parquet goes past the gate")
    # v2 re-measured as its test is: held contracts' own data, holes filled (notebook 11), dev and gate only.
    v2f = d["realism_v2"]
    v["v2f"] = {(n, w): perf(v2f[c].loc[RWIN[w]], d["all_bench"])
                for n, c in [("Book", "v2"), ("Orderflow", "orderflow"), ("Carry", "carry")] for w in ["Dev", "Gate"]}
    if v2f.index.max() > pd.Timestamp(GATE_END):
        failures.append("  realism_v2.parquet goes past the gate")
    if not same_series(v2f["orderflow"], v2d["orderflow"]):
        failures.append("  the re-measured v2 Orderflow differs from the registered one")

    # Numbers as printed by notebooks 06 to 11, and signs the text relies on.
    lk_wf, lk_ew = v["book"][("WF", "Lockbox")], v["book"][("EW", "Lockbox")]
    fl, F = v["full"], v["fwd"]
    for label, got, exp, tol in [
        ("lockbox WF Sharpe", lk_wf["sharpe"], 1.458, 0.0005),
        ("lockbox EW Sharpe", lk_ew["sharpe"], 1.233, 0.0005),
        ("lockbox WF beta to BTC", lk_wf["beta"], -0.020, 0.0005),
        ("lockbox WF alpha", lk_wf["alpha"], 0.0948, 0.0001),
        ("lockbox WF alpha t", lk_wf["alpha_t"], 1.253, 0.0005),
        ("deflated Sharpe, full", v["dsr_full"], 0.4901, 0.00005),
        ("deflated Sharpe, lockbox", v["dsr_lk"], 0.2049, 0.00005),
        ("deflated Sharpe, lockbox, first version's trials too", v["dsr_lk_all"], 0.1620, 0.00005),
        ("lockbox WF Sharpe without Seasonality", v["no_seas"]["wf"], 2.1110, 0.0005),
        ("lockbox EW Sharpe without Seasonality", v["no_seas"]["ew"], 1.3992, 0.0005),
        ("bootstrap CI low", v["ci"][0], -0.423, 0.0005),
        ("bootstrap CI high", v["ci"][1], 3.444, 0.0005),
        ("dev WF Sharpe", v["book"][("WF", "Dev")]["sharpe"], 0.397, 0.0005),
        ("dev EW Sharpe", v["book"][("EW", "Dev")]["sharpe"], 1.681, 0.0005),
        ("full WF Sharpe", v["book"][("WF", "Full")]["sharpe"], 1.020, 0.0005),
        ("full EW Sharpe", v["book"][("EW", "Full")]["sharpe"], 1.763, 0.0005),
        ("full baseline momentum Sharpe", fl["Baseline momentum"]["sharpe"], 1.167, 0.0005),
        ("full baseline pairs Sharpe", fl["Baseline pairs"]["sharpe"], -0.313, 0.0005),
        ("full seasonality Sharpe", fl["Seasonality"]["sharpe"], 1.374, 0.0005),
        ("full orderflow Sharpe", fl["Orderflow"]["sharpe"], 1.020, 0.0005),
        ("full carry Sharpe", fl["Carry"]["sharpe"], 1.148, 0.0005),
        ("full BTC Sharpe", fl["BTC"]["sharpe"], 0.855, 0.0005),
        ("full market Sharpe", fl["Equal-weight market"]["sharpe"], 0.769, 0.0005),
        ("lockbox seasonality", v["sleeve"][("seasonality", "Lockbox")]["sharpe"], -0.565, 0.0005),
        ("lockbox orderflow", v["sleeve"][("orderflow", "Lockbox")]["sharpe"], 1.949, 0.0005),
        ("lockbox carry", v["sleeve"][("carry", "Lockbox")]["sharpe"], 0.881, 0.0005),
        ("momentum timing-adjusted alpha", v["tm"]["momentum"]["alpha_ann"], -0.0076, 0.0005),
        ("momentum timing t", v["tm"]["momentum"]["timing_tstat"], 6.786, 0.005),
        ("seasonality timing-adjusted alpha", v["tm"]["seasonality"]["alpha_ann"], 0.0093, 0.0005),
        ("carry gate alpha", cs["Gate"]["alpha"], 0.7661, 0.0005),
        ("carry gate alpha, 95% low", cs["Gate"]["lo"], 0.221, 0.0005),
        ("carry lockbox alpha", cs["Lockbox"]["alpha"], 0.2716, 0.0005),
        ("charged seasonality, dev", v["seas_c"]["Dev"], 1.515, 0.0005),
        ("charged seasonality, gate", v["seas_c"]["Gate"], 0.993, 0.0005),
        ("charged seasonality, lockbox", v["seas_c"]["Lockbox"], -0.892, 0.0005),
        ("charged untilted, dev", v["untilted_c"]["Dev"], 1.636, 0.0005),
        ("charged untilted, gate", v["untilted_c"]["Gate"], 0.421, 0.0005),
        ("untilted as run vs registry, dev", v["untilted"]["Dev"], v["sel_untilted"][0], 1e-9),
        ("resizing cost on dev", v["resize_cost"], 0.0269, 0.0005),
        ("charged WF, lockbox", v["book_c"][("WF", "Lockbox")]["sharpe"], 1.490, 0.0005),
        ("charged EW, lockbox", v["book_c"][("EW", "Lockbox")]["sharpe"], 1.158, 0.0005),
        ("charged WF, full", v["book_c"][("WF", "Full")]["sharpe"], 0.703, 0.0005),
        ("charged deflated Sharpe, full", v["dsr_full_c"], 0.234, 0.0005),
        ("charged deflated Sharpe, lockbox", v["dsr_lk_c"], 0.214, 0.0005),
        ("forward WF total return", F["Book, walk-forward"]["total"], -0.0594, 0.0005),
        ("forward WF Sharpe", F["Book, walk-forward"]["sharpe"], -1.007, 0.0005),
        ("forward WF beta to BTC", F["Book, walk-forward"]["beta"], -0.120, 0.0005),
        ("forward EW total return", F["Book, equal weight"]["total"], 0.0513, 0.0005),
        ("forward EW Sharpe", F["Book, equal weight"]["sharpe"], 0.634, 0.0005),
        ("forward charged WF total return", F["Walk-forward, resizing charged"]["total"], -0.0733, 0.0005),
        ("forward seasonality total return", F["Seasonality"]["total"], 0.0255, 0.0005),
        ("forward orderflow total return", F["Orderflow"]["total"], -0.1078, 0.0005),
        ("forward orderflow Sharpe", F["Orderflow"]["sharpe"], -2.563, 0.0005),
        ("forward carry total return", F["Carry"]["total"], 0.1127, 0.0005),
        ("forward carry vol", F["Carry"]["vol"], 1.666, 0.0005),
        ("forward BTC total return", F["BTC"]["total"], 0.3184, 0.0005),
        ("forward book, orderflow part", fsplit[("book", "orderflow")], -0.0878, 0.0005),
        ("forward book, beta part", v["fwd_beta_part"], -0.0354, 0.0005),
        ("forward orderflow, worse 82-day stretches", v["fwd_of_worse"], 0.0488, 0.0005),
        ("forward carry, DEXE", fsplit[("carry", "DEXE")], 0.3766, 0.0005),
        ("forward carry, other coins", fsplit[("carry", "all other coins")], -0.0282, 0.0005),
        ("forward carry, median largest weight", conc.loc["forward", "median largest weight"], 0.2693, 0.0005),
        ("forward DEXE crash day", v["fwd_coin"]["crash"], -0.8252, 0.0005),
        ("step 0b WF lockbox", v["steps"][("0b archive data", "walk_forward")]["Lockbox"], 1.4544, 0.0005),
        ("step 1 WF lockbox", v["steps"][("1 non-crypto out", "walk_forward")]["Lockbox"], 1.2625, 0.0005),
        ("step 2 WF lockbox", v["steps"][("2 every pair", "walk_forward")]["Lockbox"], 0.3543, 0.0005),
        ("step 2 EW lockbox", v["steps"][("2 every pair", "equal_weight")]["Lockbox"], -0.6166, 0.0005),
        ("step 2 EW gate", v["steps"][("2 every pair", "equal_weight")]["Gate"], -1.4372, 0.0005),
        ("step 2 orderflow gate", v["steps"][("2 every pair", "orderflow")]["Gate"], -0.7837, 0.0005),
        ("step 2 orderflow lockbox", v["steps"][("2 every pair", "orderflow")]["Lockbox"], -0.3999, 0.0005),
        ("step 2 carry gate", v["steps"][("2 every pair", "carry")]["Gate"], -1.8152, 0.0005),
        ("step 3 orderflow dev", v["steps"][("3 carry on perps", "orderflow")]["Dev"], 0.4449, 0.0005),
        ("step 4 orderflow dev", v["steps"][("4 limit fills", "orderflow")]["Dev"], 0.0877, 0.0005),
        ("size, research list 51-100, gate", v["size"][("research list", "ranks 51-100")]["Gate"], 1.5767, 0.0005),
        ("size, research list 51-100, lockbox", v["size"][("research list", "ranks 51-100")]["Lockbox"], 1.6634, 0.0005),
        ("size, every pair 51-100, gate", v["size"][("every pair", "ranks 51-100")]["Gate"], -2.1442, 0.0005),
        ("size, every pair 51-100, lockbox", v["size"][("every pair", "ranks 51-100")]["Lockbox"], -1.0093, 0.0005),
        ("capacity, orderflow, no impact", v["cap"].loc[("orderflow", "no impact"), "sharpe y=1"], 0.8678, 0.0005),
        ("capacity, orderflow, $1M, y=0.5", v["cap"].loc[("orderflow", "$1M"), "sharpe y=0.5"], 0.2960, 0.0005),
        ("capacity, orderflow, $1M, y=1", v["cap"].loc[("orderflow", "$1M"), "sharpe y=1"], -0.2707, 0.0005),
        ("pairs charged, dev", v["pairs_c"]["charged"]["Dev"], -0.4331, 0.0005),
        ("out-of-sample WF Sharpe", v["oos"]["sharpe"], 0.3744, 0.0005),
        ("out-of-sample days", v["oos"]["n"], 453, 0),
        ("out-of-sample CI low", v["oos"]["ci"][0], -1.2196, 0.0005),
        ("out-of-sample CI high", v["oos"]["ci"][1], 1.9532, 0.0005),
        ("out-of-sample EW Sharpe", v["oos"]["sharpe_ew"], 0.7151, 0.0005),
        ("survivorship share", v["surv"], 0.4641, 0.00005),
        ("orderflow filled share", v["fills"].loc["orderflow", "filled share of orders"], 0.9895, 0.00005),
        ("orderflow missed share", v["fills"].loc["orderflow", "missed share of positions"], 0.0029, 0.00005),
        ("return where a buy missed", v["fills"].loc["orderflow", "return where a buy missed"], 0.0708, 0.00005),
        ("return where a sell missed", v["fills"].loc["orderflow", "return where a sell missed"], -0.0572, 0.00005),
        ("orderflow price P&L, all filled", v["fills"].loc["orderflow", "price P&L, every order filled"], 0.6989, 0.00005),
        ("orderflow price P&L, fill model", v["fills"].loc["orderflow", "price P&L, fill model"], 0.3984, 0.00005),
        ("forward beta, share of windows lower", v["fwd_beta"].loc["walk_forward", "share of research windows lower"], 0.0695, 0.00005),
        ("v2 book dev", v["v2"][("Book", "Dev")]["sharpe"], 1.3949, 0.0005),
        ("v2 book gate", v["v2"][("Book", "Gate")]["sharpe"], 1.1161, 0.0005),
        ("v2 orderflow gate", v["v2"][("Orderflow", "Gate")]["sharpe"], -0.7837, 0.0005),
        ("v2 carry dev", v["v2"][("Carry", "Dev")]["sharpe"], 1.7590, 0.0005),
        ("v2 carry gate", v["v2"][("Carry", "Gate")]["sharpe"], 2.4116, 0.0005),
        ("v2 re-measured book dev", v["v2f"][("Book", "Dev")]["sharpe"], 1.3479, 0.0005),
        ("v2 re-measured book gate", v["v2f"][("Book", "Gate")]["sharpe"], 1.1264, 0.0005),
        ("v2 re-measured carry dev", v["v2f"][("Carry", "Dev")]["sharpe"], 1.6796, 0.0005),
        ("v2 re-measured carry gate", v["v2f"][("Carry", "Gate")]["sharpe"], 2.4241, 0.0005),
        ("step 3 carry dev, dropped days at zero", v["gaps"]["dropped days at zero"]["Dev"], 0.3744, 0.0005),
        ("step 3 carry gate, dropped days at zero", v["gaps"]["dropped days at zero"]["Gate"], -1.5827, 0.0005),
        ("step 3 carry dev", v["gaps"]["held contracts' own data"]["Dev"], 0.2544, 0.0005),
        ("step 3 carry gate", v["gaps"]["held contracts' own data"]["Gate"], -1.5104, 0.0005),
        ("pairs picked a day earlier, dev", v["pairs_c"]["picked a day earlier"]["Dev"], -0.3684, 0.0005),
        ("pairs picked a day earlier, gate", v["pairs_c"]["picked a day earlier"]["Gate"], 0.4297, 0.0005),
        ("EW dev Sharpe on the walk-forward dates", v["same_dates"]["Dev"]["ew"], 0.4590, 0.0005),
        ("EW full Sharpe on the walk-forward dates", v["same_dates"]["Full"]["ew"], 1.1667, 0.0005),
        ("EW lockbox deflated Sharpe", v["dsr_lk_ew"], 0.1621, 0.00005),
        ("twsq days", v["twsq_days"], 701, 0),
    ]:
        check(label, got, exp, tol, failures)
    if not (F["Book, walk-forward"]["sharpe"] < 0 < F["Book, equal weight"]["total"]
            and fsplit[("carry", "all other coins")] < 0 < fsplit[("carry", "DEXE")]
            and v["fwd_of_weight"]["min"] > 0.6 and len(v["fwd_coin"]["held"]) == 12):
        failures.append("  section 6.1 text no longer matches the forward results")
    wh = v["whole"]
    if not (wh["dev"] == conc.loc["dev", "days one coin holds a whole side"] == 23
            and conc.loc["gate", "days one coin holds a whole side"] == 0
            and conc.loc["lockbox", "days one coin holds a whole side"] == 0
            and conc.loc["forward", "days one coin holds a whole side"] == 10
            and wh["jan20"] == 16 and wh["nov22"] == 7 and wh["nov22_coins"] == {"SOL"}):
        failures.append("  section 6.1 text on Carry's concentration no longer matches the data")
    fg, g05 = d["realism_fastrev"], fastrev_grid().set_index(["lookback_h", "rebal_h"])
    for (lb, rb), r in g05.iterrows():
        for col in ["gross_sharpe_dev", "gross_ann", "cost_drag_ann", "net_dev", "net_gate"]:
            check(f"fast reversal {lb}h/{rb}h {col}, 60 coins", fg.loc[("research panel", lb, rb), col], r[col], 0.006,
                  failures)
    ep = fg.loc["every pair"]
    if not ((ep["net_dev"] < 0).all() and (ep["net_gate"] < 0).all() and ep["gross_over_drag"].max() < 3):
        failures.append("  section 5.5 text says every hourly config loses after costs on every pair")
    check("fast reversal 4h/1h gross Sharpe, every pair", ep.loc[(4, 1), "gross_sharpe_dev"], 6.9227, 0.0005, failures)
    check("fast reversal 4h/1h gross/cost, every pair", ep.loc[(4, 1), "gross_over_drag"], 0.9913, 0.0005, failures)
    st = v["steps"]
    if not (all(st[("2 every pair", s)][w] < 0 for s in ["orderflow", "carry"] for w in ["Gate", "Lockbox"])
            and st[("2 every pair", "seasonality")]["Dev"] > 1 and st[("2 every pair", "seasonality")]["Gate"] > 1
            and st[("2 every pair", "seasonality")]["Lockbox"] < 0
            and abs(st[("0b archive data", "walk_forward")]["Lockbox"] - lk_wf["sharpe"]) < 0.01):
        failures.append("  section 5 text no longer matches notebook 11")
    if not (v["untilted_c"]["Dev"] > v["seas_c"]["Dev"] and v["seas_c"]["Gate"] > v["untilted_c"]["Gate"]):
        failures.append("  appendix A.1 text assumes the charged overlay loses on dev and wins on the gate")
    if not (abs(tw.loc["SeasonalMomentum", "sharpe"]) < 0.3 and tw.loc["OrderflowFollow", "sharpe"] < 0
            and tw.loc["FundingCarry", "sharpe"] > 0):
        failures.append("  section 8 text no longer matches the twsq results")
    if not all(abs(fl[k][b]) < 0.03 for k in ["Book, walk-forward", "Book, equal weight"]
               for b in ["beta", "beta_mkt"]):
        failures.append("  the summary says both books have betas within 0.03 of zero")
    if failures:
        raise AssertionError("Report numbers disagree with the notebooks; PDF not written.\n"
                             + "\n".join(failures))
    return v


def doc_failures(v: dict, d: dict) -> list:
    """Numbers in README.md and alphas/README.md that no longer match the results.

    Each piece of text is rebuilt from the same values as the report and has
    to appear in the file as written (line breaks aside), so a number that
    drifts in either README stops the build like one in the report.
    """
    st, v2f, v2s, sd, fl, F, pc, oos = (v[k] for k in ["steps", "v2f", "v2", "same_dates", "full", "fwd",
                                                       "pairs_c", "oos"])
    lk_wf, sl, base, tw = v["book"][("WF", "Lockbox")], v["sleeve"], v["base"], v["twsq"]
    fo = v["fills"].loc["orderflow"]
    miss_cost = 1 - fo["price P&L, fill model"] / fo["price P&L, every order filled"]
    fr4 = fastrev_grid().set_index(["lookback_h", "rebal_h"]).loc[(4, 1)]
    split = d["carry_split"]
    n_reg = len(d["registry"])
    s = lambda key: v2f[key]["sharpe"]                       # noqa: E731
    readme = [
        f"a registry of the {n_reg} configurations the research chose between (the two baselines kept "
        f"parameters set before it), deflated Sharpe ratios that charge for the {N_TRIALS} research ones",
        f"{pct(v['surv'], 0)} of the historical top-100 universe was missing from it",
        f"with all {v['n_all']} coins, delisted ones included, the book's lockbox Sharpe falls from "
        f"{lk_wf['sharpe']:.2f} to {st[('2 every pair', 'walk_forward')]['Lockbox']:.2f}",
        f"{pct(fo['filled share of orders'], 0)} of limit orders fill",
        f"costs Orderflow {pct(miss_cost, 0)} of its price P&L on dev",
        f"weighted by rank on perp prices, Sharpe {s(('Carry', 'Dev')):.2f} and {s(('Carry', 'Gate')):.2f}",
        f"those dates the equal-weight book's is {sr(sd['Dev']['ew'])}",
        f"as first run they were {sr(v2s[('Book', 'Dev')]['sharpe'])} and {sr(v2s[('Book', 'Gate')]['sharpe'])}, "
        f"and {sr(v2s[('Carry', 'Dev')]['sharpe'])} and {sr(v2s[('Carry', 'Gate')]['sharpe'])} for Carry",
        f"{N_TRIALS} research configs, which the deflated Sharpe ratio charges for",
        f"the three v2 rows, {n_reg} in all",
        f"had a Sharpe of {lk_wf['sharpe']:.2f}, an alpha of {pct(lk_wf['alpha'])} a year (Newey-West t = "
        f"{lk_wf['alpha_t']:.2f}) and a beta to BTC of {lk_wf['beta']:.2f}. Its deflated Sharpe was {v['dsr_lk']:.2f}",
        f"Carry runs at about {pct(v['carry_scale']['Gate']['vol'], 0)} volatility",
        f"left out {pct(v['surv'], 0)} of the historical universe",
        f"moves its dev Sharpe from {pc['as run']['Dev']:.2f} to {pc['picked a day earlier']['Dev']:.2f}",
        f"Adding the {F['Book, walk-forward']['n']}-day forward test",
        f"a Sharpe of {oos['sharpe']:.2f} over {oos['n']} days",
        f"Sharpe of {s(('Book', 'Dev')):.2f} and {s(('Book', 'Gate')):.2f}, nearly all from Carry "
        f"({s(('Carry', 'Dev')):.2f} and {s(('Carry', 'Gate')):.2f}); its Orderflow sleeve fails the gate "
        f"({s(('Orderflow', 'Gate')):.2f})",
        f"As first run the figures were {v2s[('Book', 'Dev')]['sharpe']:.2f} and {v2s[('Book', 'Gate')]['sharpe']:.2f} "
        f"(Carry {v2s[('Carry', 'Dev')]['sharpe']:.2f} and {v2s[('Carry', 'Gate')]['sharpe']:.2f})",
        f"(Sharpe {s(('Carry', 'Dev')):.2f} and {s(('Carry', 'Gate')):.2f})",
        f"(lockbox {sr(sl[('orderflow', 'Lockbox')]['sharpe'])})",
        f"(dev {sr(base[('momentum', 'Dev')]['sharpe'])}, gate {sr(base[('momentum', 'Gate')]['sharpe'])})",
        f"Pairs trading lost money on dev ({base[('reversal', 'Dev')]['sharpe']:.2f})",
        f"(gross Sharpe {fr4.gross_sharpe_dev:+.2f} at a 4-hour lookback) but costs {fr4.cost_drag_ann:.0%} a year "
        f"against a {fr4.gross_ann:.0%} gross return",
        f"on dev only by {sd['Dev']['ew']:.2f} to {sd['Dev']['wf']:.2f}",
        "(Sharpe " + ", ".join(sr(tw.loc[n, "sharpe"]) for n in ["SeasonalMomentum", "OrderflowFollow"])
        + f" and {sr(tw.loc['FundingCarry', 'sharpe'])})",
        f"an {F['Book, walk-forward']['n']}-day forward test",
        f"holds exactly {N_TRIALS} rows",
    ]
    readme += [f"or {v['dsr_lk_all']:.2f} counting the first version's configurations",
               f"the book's lockbox deflated Sharpe would be {v['dsr_lk_all']:.2f} instead of {v['dsr_lk']:.2f}",
               f"{FIRST_VERSION_TRIALS} configurations by its own count"]
    rfa, rfe = v["refits"]["0 as run"], v["refits"]["2 every pair"]
    readme.append(f"the research book's lockbox Sharpe ranges from {rfa[0]:.2f} to {rfa[1]:.2f} and the every-pair "
                  f"book's from {rfe[0]:.2f} to {rfe[1]:.2f}")
    drift = {pct(v["drift"].loc[n, "dev"]) for n in ["orderflow", "carry"]}
    reb = v["rebalance"]
    readme.append(f"for Orderflow and Carry they would cost about {drift.pop()} a year on dev, and the books' "
                  f"daily moves of money between sleeves {pct(reb.loc['equal weight', 'dev'], 2)} for the "
                  f"equal-weight book and {pct(reb.loc['walk-forward', 'dev'], 2)} for the walk-forward one"
                  if len(drift) == 1 else "Orderflow's and Carry's drift costs on dev no longer round to one figure")
    for key, lab in [("walk_forward", "walk-forward"), ("equal_weight", "equal weight")]:
        for step, lst in [("0 as run", "research coin list"), ("2 every pair", "every pair")]:
            readme.append(f"| Frozen book, {lab}, {lst} | " + " | ".join(sr(st[(step, key)][w]) for w in RWIN) + " |")
    for n, lab in [("Book", "v2 book"), ("Carry", "v2 Carry sleeve")]:
        readme.append(f"| {lab}, every pair | {sr(s((n, 'Dev')))} | {sr(s((n, 'Gate')))} | not run | from {strategies.V2_START} |")
    for key, lab in [("Book, equal weight", "Book, equal weight"),
                     ("Book, walk-forward", f"Book, walk-forward (from {fl['Book, walk-forward']['start']:%Y-%m})"),
                     ("Seasonality", "Seasonality"), ("Orderflow", "Orderflow"), ("Carry", "Carry"),
                     ("Baseline momentum", "Baseline momentum"), ("Baseline pairs", "Baseline pairs"), ("BTC", "BTC")]:
        r = fl[key]
        tail = f"{sr(r['beta'])} | {alpha_cell(r)}" if "alpha" in r else " |"
        readme.append(f"| {lab} | {pct(r['ret'])} | {pct(r['vol'])} | {r['sharpe']:.2f} | {pct(r['mdd'])} | {tail} |")
    for step, lab in [("0 as run", "As run (research coin list)"), ("0b archive data", "Same list, archive data"),
                      ("1 non-crypto out", "Pegged assets and tokenized stocks out"),
                      ("2 every pair", "Every pair, delisted coins included"),
                      ("3 carry on perps", "and Carry on perp prices"),
                      ("4 limit fills", "and limit orders that have to fill")]:
        readme.append(f"| {lab} | {sr(st[(step, 'walk_forward')]['Lockbox'])} | "
                      f"{sr(st[(step, 'equal_weight')]['Lockbox'])} |")
    alphas = [f"All three runs cover the same {v['twsq_days']} days",
              f"funding income is {pct(split.loc['dev', 'funding share of total'], 0)} of the carry sleeve's P&L "
              f"on dev and {pct(split.loc['gate', 'funding share of total'], 0)} on the gate"]
    for name, r in tw.iterrows():
        alphas.append(f"| {name} | {pct(r['ann_return'])} | {pct(r['ann_vol'])} | {sr(r['sharpe'])} | "
                      f"{pct(r['max_drawdown'])} | {pct(r['avg_daily_turnover'], 0)} |")
    failures = []
    for path, texts in [(ROOT / "README.md", readme), (ALPHAS / "README.md", alphas)]:
        doc = " ".join(path.read_text().split())
        failures += [f"  {path.relative_to(ROOT)}: expected \"{t}\"" for t in texts if " ".join(t.split()) not in doc]
    return failures


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

FAMILY_LABEL = {"incumbent_momentum": "baseline", "incumbent_pairs": "baseline",
                "seasonality": "seasonality", "orderflow": "orderflow", "carry": "carry",
                "fastrev": "fast reversal", "portfolio": "book", "v2": "v2"}


def describe(family: str, cfg: dict) -> str:
    """Readable label for a registry config."""
    p = cfg.get("probe")
    if family == "incumbent_momentum":
        return f"momentum, {cfg['lookback']}d lookback, {cfg['target_vol']:.0%} vol target"
    if family == "incumbent_pairs":
        return f"pairs, entry {cfg['entry']:g} sigma, {cfg['zwin']}d z-score, top {cfg['top_k']}"
    if family == "seasonality":
        if p == "P1":
            return f"P1 momentum, weekend x{cfg['weekend']:g}, weekday x{cfg['weekday']:g}"
        if p == "P2":
            where = {"weekday": "on weekdays", "weekend": "at weekends",
                     "turn_of_month": "around the turn of the month"}[cfg["bucket"]]
            return f"P2 hold market {where} only"
        if p == "P3":
            return f"P3 hold market {cfg['bucket'].replace('_', ' ')} only, hourly"
        return f"P4 momentum, turn of month x{cfg['tom_mult']:g}"
    if family == "orderflow":
        if p == "P1" and cfg["cond"] == "none":
            return f"P1 reversal {cfg['lookback']}d, no filter"
        if p == "P1":
            return f"P1 reversal {cfg['lookback']}d, volume z < {cfg['thresh']:g}"
        if p == "P2":
            return f"P2 reversal {cfg['lookback']}d, flow-unconfirmed"
        return f"P3 imbalance {cfg['smooth']}d, {cfg['dir']}"
    if family == "carry":
        if p == "P1":
            return f"P1 carry, {cfg['smooth']}d funding"
        if p == "P2":
            return f"P2 reversal {cfg['lookback']}d, |funding z| > {cfg['fz_thresh']:g}"
        return f"P3 funding change, {cfg['smooth']}d, {cfg['dir']}"
    if family == "v2":
        if cfg.get("sleeve") == "orderflow":
            return "Orderflow, every pair"
        if cfg.get("sleeve") == "carry":
            return "Carry, rank weights on perps, every pair"
        return "v2 book, equal weight"
    if family == "fastrev":
        return f"{cfg['lookback_h']}h lookback, rebalance every {cfg['rebal_h']}h"
    if cfg.get("rule") == "walk_forward_mv":
        return "walk-forward mean-variance, 756d fit, 63d step"
    return "equal weight"


def family_rows(reg: pd.DataFrame, family: str) -> list:
    rows = [["Config", "Dev Sharpe", "Gate Sharpe"]]
    for _, r in reg[reg["family"] == family].iterrows():
        dev, gate = float(r["dev_sharpe"]), float(r["gate_sharpe"])
        rows.append([describe(family, json.loads(r["config"])),
                     sr(dev) if np.isfinite(dev) else "n/a", sr(gate) if np.isfinite(gate) else "n/a"])
    return rows


def alpha_cell(s: dict) -> str:
    return f"{pct(s['alpha'])} ({s['alpha_t']:.1f})"


def perf_rows(stats: dict, keys: list, labels: list) -> list:
    rows = [["", "Ann. return", "Ann. vol", "Sharpe", "Max DD", "Beta BTC", "Beta mkt", "Alpha (t)"]]
    for key, lab in zip(keys, labels):
        s = stats[key]
        rows.append([lab, pct(s["ret"]), pct(s["vol"]), sr(s["sharpe"]), pct(s["mdd"]),
                     sr(s["beta"]), sr(s["beta_mkt"]), alpha_cell(s)])
    return rows


def full_rows(full: dict) -> list:
    rows = [["", "From", "Ann. return", "Ann. vol", "Sharpe", "Max DD", "Beta BTC", "Beta mkt", "Alpha (t)"]]
    for name, s in full.items():
        reg = "alpha" in s
        rows.append([name, f"{s['start']:%Y-%m}", pct(s["ret"]), pct(s["vol"]), sr(s["sharpe"]),
                     pct(s["mdd"]), sr(s["beta"]) if reg else "", sr(s["beta_mkt"]) if reg else "",
                     alpha_cell(s) if reg else ""])
    return rows


PERF_WIDTHS = (31, 15, 13, 12, 13, 13, 13, 18)
SLEEVE_NAMES = ["seasonality", "orderflow", "carry"]
FULL_WIDTHS = (31, 12, 15, 13, 12, 13, 13, 13, 18)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(d: dict, tmp: Path) -> dict:
    cp, bench, sleeves = d["combined_portfolio"], d["benchmarks"], d["sleeves_full"]
    wf = cp["walk_forward"].loc[DEV_START:LOCKBOX_END].dropna()
    ew = cp["equal_weight"].loc[DEV_START:LOCKBOX_END].dropna()
    btc = bench["BTC"].loc[DEV_START:LOCKBOX_END].dropna()
    paths = {}

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1, 1]})
    ax = axes[0]
    for ser, label, color, ls in [(wf, "Walk-forward", "steelblue", "-"),
                                  (ew, "Equal weight", "darkorange", "-"),
                                  (btc, "BTC", "gray", "--")]:
        eq = (1.0 + ser).cumprod()
        ax.plot(eq.index, eq.values, color=color, ls=ls, lw=0.9 if label == "BTC" else 1.2,
                label=f"{label} (Sharpe {metrics.sharpe(ser):.2f})")
    for a in axes:
        a.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0)
        a.axvline(pd.Timestamp(LOCKBOX_START), color="red", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_title("Growth of $1, log scale (dotted line: gate starts; dashed line: lockbox opens)")
    ax.legend(loc="upper left", fontsize=8)
    dd = metrics.drawdown_curve(wf)
    axes[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.35)
    axes[1].set_title("Walk-forward drawdown")
    roll = wf.rolling(180).mean() / wf.rolling(180).std() * np.sqrt(metrics.TRADING_DAYS)
    axes[2].plot(roll.index, roll.values, color="steelblue", lw=0.9)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_title("Walk-forward rolling 180-day Sharpe")
    fig.tight_layout()
    paths["equity"] = tmp / "equity.png"
    fig.savefig(paths["equity"], dpi=130)
    plt.close(fig)

    # The weights the book actually used: same inputs and settings as notebook 06.
    _, weight_path = robustness.walk_forward_weights(sleeves.dropna(), train_days=756,
                                                     step_days=63, min_train=252)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    colors = {"seasonality": "goldenrod", "orderflow": "steelblue", "carry": "forestgreen"}
    for col in weight_path.columns:
        ax.plot(weight_path.index, weight_path[col], label=col, color=colors[col], lw=1.1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0)
    ax.axvline(pd.Timestamp(LOCKBOX_START), color="red", ls="--", lw=1.2)
    ax.set_title("Walk-forward sleeve weights (fit on the previous 756 days, refit every 63 days)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    paths["weights"] = tmp / "weights.png"
    fig.savefig(paths["weights"], dpi=130)
    plt.close(fig)

    fb = d["forward_book"].loc[FORWARD]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, lines in [(axes[0], [("walk_forward", "Book, walk-forward", "steelblue", "-"),
                                 ("equal_weight", "Book, equal weight", "darkorange", "-"),
                                 ("BTC", "BTC", "gray", "--")]),
                      (axes[1], [("seasonality", "Seasonality", "goldenrod", "-"),
                                 ("orderflow", "Orderflow", "steelblue", "-"),
                                 ("carry", "Carry", "forestgreen", "-")])]:
        for col, label, color, ls in lines:
            eq = (1.0 + fb[col]).cumprod()
            ax.plot(eq.index, eq.values, color=color, ls=ls, lw=1.1, label=label)
        ax.axhline(1.0, color="k", lw=0.6)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
        ax.legend(fontsize=8, loc="upper left")
    axes[1].axvline(pd.Timestamp("2026-07-21"), color="red", ls=":", lw=1.0)
    axes[0].set_title("Books and BTC")
    axes[1].set_title("Sleeves")
    fig.tight_layout()
    paths["forward"] = tmp / "forward.png"
    fig.savefig(paths["forward"], dpi=130)
    plt.close(fig)

    rs = d["realism_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key, title in [(axes[0], "walk_forward", "Walk-forward book"), (axes[1], "equal_weight", "Equal-weight book")]:
        for step, label, color in [("0 as run", "Research coin list (as run)", "steelblue"),
                                   ("2 every pair", "Every pair", "crimson")]:
            s = rs[f"{step}|{key}"].loc[DEV_START:FORWARD_END].dropna()
            eq = (1.0 + s).cumprod()
            ax.plot(eq.index, eq.values, color=color, lw=1.1, label=label)
        for x, ls, c in [(GATE_START, ":", "navy"), (LOCKBOX_START, "--", "red"), (FORWARD_START, "--", "gray")]:
            ax.axvline(pd.Timestamp(x), color=c, ls=ls, lw=1.0)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    paths["survivorship"] = tmp / "survivorship.png"
    fig.savefig(paths["survivorship"], dpi=130)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, "Statistical Arbitrage in Cryptocurrencies", align="R",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, str(self.page_no()), align="C")
        self.set_text_color(0)

    def h1(self, text: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.multi_cell(0, 7, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 6, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullets(self, items: list):
        self.set_font("Helvetica", "", 10)
        for item in items:
            self.multi_cell(0, 5, _latin1("- " + item), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def caption(self, text: str):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90)
        self.multi_cell(0, 4, _latin1(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(1)

    def table_block(self, rows: list, title: str | None = None, widths: tuple | None = None,
                    align: tuple | None = None):
        if title:
            self.set_font("Helvetica", "B", 9)
            self.multi_cell(0, 5, _latin1(title), new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        # text in the first column, numbers to the right, unless told otherwise
        align = align or ("LEFT",) + ("RIGHT",) * (len(rows[0]) - 1)
        with self.table(first_row_as_headings=True, line_height=4.6, padding=1,
                        col_widths=widths, text_align=align) as table:
            for row in rows:
                tr = table.row()
                for cell in row:
                    tr.cell(_latin1(str(cell)))
        self.ln(2)

    def figure(self, path: Path, w: int = 172):
        self.image(str(path), w=w)
        self.ln(2)


def build_pdf(d: dict, v: dict, figs: dict, out: Path = OUT) -> int:
    reg = d["registry"]
    book, bookc, sl, base, fl, F = v["book"], v["book_c"], v["sleeve"], v["base"], v["full"], v["fwd"]
    st, v2s, v2f, sd = v["steps"], v["v2"], v["v2f"], v["same_dates"]
    lk_wf, lk_ew = book[("WF", "Lockbox")], book[("EW", "Lockbox")]
    full_wf, full_ew = fl["Book, walk-forward"], fl["Book, equal weight"]
    grid = fastrev_grid()
    tw = v["twsq"]
    stress = d["cost_stress"].set_index(["sleeve", "cost_bps"])
    split = d["carry_split"]
    price, uni, funding, price_1h = d["price"], d["universe"], d["funding"], d["price_1h"]
    seas_dev, seas_gate = v["sel_seas"]
    unt_dev, unt_gate = v["sel_untilted"]
    of_dev, of_gate = v["sel_of"]
    ca_dev, ca_gate = v["sel_carry"]
    fr4 = grid[(grid.lookback_h == 4) & (grid.rebal_h == 1)].iloc[0]
    seas_c, unt_c = v["seas_c"], v["untilted_c"]
    tm_mom, tm_seas = v["tm"]["momentum"], v["tm"]["seasonality"]
    cs = v["carry_scale"]
    lo, hi = v["ci"]
    lo_c, hi_c = v["ci_c"]
    uni_size = int(uni.loc[DEV_START:].sum(axis=1).median())
    n_hourly = price_1h.shape[1]
    n_stopped = int((price.apply(lambda c: c.last_valid_index()) < price.index.max()).sum())
    of_cut = 1 - stress.loc[("orderflow", 20), "dev"] / stress.loc[("orderflow", 7), "dev"]

    pdf = Report(format="A4")
    pdf.set_creation_date(BUILD_DATE)
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 15, 18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 19)
    pdf.multi_cell(0, 9, "Statistical Arbitrage in Cryptocurrencies", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90)
    pdf.multi_cell(0, 6, "Momentum, reversal and funding carry "
                         "on Binance data, 2020 to 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)

    reg_rows = len(reg)
    pdf.h1("Summary")
    pdf.body(
        "I tested momentum, reversal and funding-carry strategies on Binance data from 2020 to "
        "September 2026, charged realistic costs (20 bps for market orders, 7 bps for limit "
        "orders) and combined the ones that passed into a book. Parameters were chosen on a "
        "development window (2020 to July 2024) and screened on a gate window (August 2024 to June "
        "2025), which also decided between configs that passed both, and the final book was run "
        "once on a lockbox year (July 2025 to July 2026). Every configuration the research chose "
        f"between is in a registry, {reg_rows} rows in all: {N_TRIALS} research configurations, which "
        "the deflated Sharpe charges for, the book's two combination rules and v2's three rows. The "
        "post-hoc checks of section 5 and appendix A re-score frozen strategies and log nothing. The "
        "two baselines kept parameters set before the registry existed (section 2), and untilted "
        "momentum is logged twice, as the baseline and as one of Seasonality's configurations, so "
        f"the {N_TRIALS} research rows hold {N_TRIALS - 1} distinct strategies.\n\n"
        "On the research data the book made money on its lockbox: a walk-forward Sharpe of "
        f"{lk_wf['sharpe']:.2f} with a beta to BTC of {lk_wf['beta']:+.2f}. I then rebuilt the data "
        "from Binance's public archive to test that result, and found that the research coin list "
        "was mostly survivors. It was the 150 USDT pairs with the most 24-hour volume on 2026-07-06, "
        f"and {pct(v['surv'], 0)} of the universe's coin-days before then were coins it left out. With "
        "every pair included, the frozen book's lockbox Sharpe falls to "
        f"{st[('2 every pair', 'walk_forward')]['Lockbox']:.2f} (walk-forward) and "
        f"{st[('2 every pair', 'equal_weight')]['Lockbox']:.2f} (equal weight), so most of the result "
        "came from the coin list. Fills and market impact tell the same story for Orderflow, the "
        "sleeve that looked best.\n\n"
        "What holds up on the full universe, on dev and gate, is funding carry: weighted by rank and "
        f"measured on perp prices, it has a Sharpe of {v2f[('Carry', 'Dev')]['sharpe']:.2f} on dev and "
        f"{v2f[('Carry', 'Gate')]['sharpe']:.2f} on the gate at about "
        f"{pct(v2f[('Carry', 'Dev')]['vol'], 0)} volatility. A second version of the book built around "
        "it (v2) was registered in the repository before it was tested; its test is every day from "
        "2026-09-28, recorded month by month in notebook 12."
    )
    rows = [["Sharpe ratio", "Dev", "Gate", "Lockbox", "Forward"]]
    for key, lab in [("walk_forward", "walk-forward"), ("equal_weight", "equal weight")]:
        for step, lst in [("0 as run", "research coin list"), ("2 every pair", "every pair")]:
            rows.append([f"Frozen book, {lab}, {lst}"] + [sr(st[(step, key)][w]) for w in RWIN])
    for n, lab in [("Book", "v2 book"), ("Carry", "v2 Carry sleeve")]:
        rows.append([f"{lab}, every pair", sr(v2f[(n, "Dev")]["sharpe"]), sr(v2f[(n, "Gate")]["sharpe"]),
                     "not run", "from 2026-09-28"])
    pdf.table_block(rows, title="The main results at a glance", widths=(70, 22, 22, 22, 34))
    pdf.caption("Every pair: the daily top 100 chosen from all Binance pairs, delisted ones included "
                "(section 5). The forward window is 2026-07-07 to 2026-09-26 (section 6). The "
                "walk-forward book needs 756 days of history, so its dev figure covers 2021-10 to "
                f"2024-07; over those dates the equal-weight book's is {sr(sd['Dev']['ew'])}. v2 is never "
                "evaluated on the lockbox or on that forward window, since both had been seen before it was "
                "written. Its figures include two fixes made after its first run and before any test "
                f"result was computed; as first run they were {sr(v2s[('Book', 'Dev')]['sharpe'])} and "
                f"{sr(v2s[('Book', 'Gate')]['sharpe'])} for the book and {sr(v2s[('Carry', 'Dev')]['sharpe'])} "
                f"and {sr(v2s[('Carry', 'Gate')]['sharpe'])} for Carry (section 7).")

    fo = v["fills"].loc["orderflow"]
    miss_cost = 1 - fo["price P&L, fill model"] / fo["price P&L, every order filled"]
    pdf.h1("What the project shows")
    pdf.bullets([
        "A protocol that can fail. The history is split into development, gate and lockbox windows "
        "in time order, every configuration the research chose between is logged "
        f"({reg_rows} rows, the {N_TRIALS} research ones charged for with deflated Sharpe ratios), the "
        "protocol opened the lockbox once, and v2's rules "
        "were committed before any v2 number existed. The problems below were found because of it.",
        "A survivorship check on its own result. The daily data was rebuilt from Binance's public "
        f"archive for {v['n_all']} coins, delisted ones included, and its prices and funding rates "
        f"match the research data exactly where the two overlap. It showed that {pct(v['surv'], 0)} of "
        "the historical universe "
        "was missing from the research coin list, and that the book's lockbox result came mostly "
        "from that gap (section 5).",
        f"Execution measured, not assumed. A fill model shows that {pct(fo['filled share of orders'], 0)} "
        "of limit orders fill, but the ones that miss are the days the price runs away, which costs "
        f"Orderflow {pct(miss_cost, 0)} of its price P&L on dev. A per-coin market-impact model shows "
        "its edge is mostly gone by $1M (section 5).",
        "An edge that holds up on the full universe, on dev and gate: funding carry weighted by rank "
        f"on perp prices, with a Sharpe of {v2f[('Carry', 'Dev')]['sharpe']:.2f} and "
        f"{v2f[('Carry', 'Gate')]['sharpe']:.2f}. v2 is built on it and is being tested on new data "
        "(section 7).",
        "Code that checks itself. The shared library has tests for look-ahead, dollar neutrality, "
        "costs, the fill model and the data pipeline; the report is rebuilt from the notebooks' "
        "saved results and stops before writing the PDF if a headline number, in the report or the "
        "README, disagrees with them; and check.py runs every check in one command.",
    ])

    # 1. Data and method
    pdf.h1("1. Data and method")
    pdf.bullets([
        f"Daily klines for the top 150 Binance USDT pairs by volume, 2018-01 to 2026-07 "
        f"({price.shape[1]} names returned data, {n_stopped} of which had stopped trading before the "
        "fetch). Close, USD volume and taker-buy volume are kept.",
        f"Hourly klines for the {n_hourly} names with the highest 30-day volume at fetch time, "
        "2020-01 to 2026-07. The hourly research starts in 2020-06.",
        f"Perpetual funding rates for the same {n_hourly} names where a perp trades under the coin's "
        f"own name ({funding.shape[1]} names; the others are four pegged assets and PEPE, whose perp "
        f"is 1000PEPE), summed per day, from {funding.index.min():%Y-%m}.",
        "The data was fetched on 2026-07-06, so that last day is a partial day in every panel.",
        "For section 5 and v2, the same daily data for every Binance USDT pair that ever traded, "
        f"from the public archive ({v['n_all']} coins once stablecoins, wrapped coins, pegged assets, "
        "tokenized stocks and leveraged tokens are removed), hourly bars for the coins that were ever "
        "in the "
        "top 100, and prices and funding for every perp.",
    ])
    pdf.body(
        "The tradable universe is point-in-time: on each day, the 100 coins with the highest "
        "30-day median dollar volume, using data up to the day before, with a $1M floor and at "
        "least 30 days of history. Stablecoins, wrapped coins and leveraged tokens are excluded. "
        f"The median universe since 2020 has {uni_size} names. The 2018-2019 data serves as history "
        "for the first signals, and from September 2019 the sleeves' returns also train the first "
        "walk-forward weights.\n\n"
        "The backtest applies weights set at the close of day t to the return of day t+1, so no "
        "position uses information it could not have had, with two exceptions. The pairs baseline "
        "picked its pairs with the close of the day whose return they then earned (section 5.6). "
        "And the coin lists themselves, the daily one and the 60 names with hourly bars and funding, "
        "were picked by volume on the fetch date; section 5 shows that most of the research result "
        "came from that choice. Costs are charged on turnover every "
        "day: 20 bps per dollar traded for market orders (7 bps commission plus 13 bps "
        "slippage) and 7 bps for limit orders. The momentum sleeves take liquidity and pay "
        "20 bps; orderflow, carry and pairs rebalance passively and pay 7 bps. Backtests are "
        "unconstrained, and Sharpe ratios are annualised with 365 days."
    )
    pdf.table_block([
        ["Window", "Dates", "Used for"],
        ["Development", f"{DEV_START} to {DEV_END}", "choosing signals and parameters"],
        ["Gate", f"{GATE_START} to {GATE_END}", "a check before anything is kept, and the pick between configs that pass"],
        ["Lockbox", f"{LOCKBOX_START} to {LOCKBOX_END}", "one final run of the frozen book"],
    ], title="Validation windows", widths=(22, 38, 80), align=("LEFT", "LEFT", "LEFT"))
    pdf.caption("The windows split the history roughly 70/15/15 in time, the usual train, "
                "validation and test split. The gate also decided between configs that passed both "
                "windows, so the chosen sleeves' gate numbers are not out of sample; the lockbox is. "
                "The protocol read the lockbox once, in notebook 06, after every selection decision had "
                "been made; the project's first version had already run the two baselines over it "
                "(section 2). Appendix A and section 5 re-score the same frozen book with corrected "
                "inputs and say so.")

    # 2. Baselines
    pdf.h1("2. Baselines: momentum and pairs")
    pdf.body(
        "Two strategies serve as baselines that every later idea has to "
        "beat. Time-series momentum holds each coin long or short by the sign of its 30-day "
        "return (skipping the latest day) and scales the book to 15% volatility. The pairs sleeve "
        "re-selects up to 20 correlated pairs every 63 days, opens a spread when its z-score "
        "passes 2 and holds it until the z-score passes 2 on the other side. Their settings were "
        "fixed before the gate window, so the gate shows how they hold up on data they were not "
        "chosen on.\n\n"
        "Those settings come from the project's first version, built on CoinGecko data from "
        "2026-06-29 to 07-02, before this protocol existed. It picked them from small grids on "
        "2020-2022 data and also tested holding the market only at weekends: "
        f"{FIRST_VERSION_TRIALS} configurations by its own count, and about ten exploratory variants "
        "besides, none of them in the registry. It also ran both baselines as twsq alphas up to "
        "2026-07-02, momentum from 2024-08 and pairs from 2025-07 (Sharpe -0.29 and -1.11), so for "
        "these two the lockbox was not unseen, and the plan for this protocol, written before its data "
        "was fetched, started from knowing that momentum had gone flat after mid-2024 and set out to "
        "find signals alive in 2024-26. Orderflow, Carry, the Seasonality rule and the combination "
        "rules came later and were chosen on dev and gate only. Counting the first version's "
        f"{FIRST_VERSION_TRIALS} configurations as trials, the book's lockbox deflated Sharpe would be "
        f"{v['dsr_lk_all']:.2f} instead of {v['dsr_lk']:.2f} (section 4).\n\n"
        "The July part of the git history was rebuilt in September 2026 from the original commits. "
        "They keep their original dates and results, but their prose and comments were rewritten "
        "then and a few pieces tied to the first version were cut; the first version itself is left "
        "out, and each July commit carries the time of the last original commit it combines. So the "
        "freeze commit also holds a fix to Carry's funding P&L made three minutes after the freeze, "
        "before notebook 06 ran, and the research commit holds only the final wording of "
        "Seasonality's selection rule (section 3.1)."
    )
    pdf.table_block(perf_rows(base, [("momentum", "Dev"), ("momentum", "Gate"),
                                     ("reversal", "Dev"), ("reversal", "Gate")],
                              ["Momentum, dev", "Momentum, gate", "Pairs, dev", "Pairs, gate"]),
                    title="Baseline sleeves (t-stat of alpha in brackets)", widths=PERF_WIDTHS)
    pdf.body(
        f"Momentum's Sharpe falls from {sr(base[('momentum', 'Dev')]['sharpe'])} on dev to "
        f"{sr(base[('momentum', 'Gate')]['sharpe'])} on the gate. Pairs loses money on dev "
        f"({sr(base[('reversal', 'Dev')]['sharpe'])}), so it is dropped even though the gate was "
        f"positive ({sr(base[('reversal', 'Gate')]['sharpe'])}). Momentum stays as the starting "
        "point for the seasonality work.\n\n"
        "The momentum alpha needs a caveat. Alpha in this report is the intercept of a regression "
        "of daily returns on BTC and the equal-weight market, annualised. Trend following is long "
        "while prices rise and short while they fall, so its market beta keeps changing sign and "
        "averages out near zero, and almost all of its return is labelled alpha. A regression that "
        "allows for that timing (Treynor and Mazuy, on weekly returns) gives momentum a dev alpha "
        f"of {pct(tm_mom['alpha_ann'])} a year (t = {tm_mom['alpha_tstat']:.1f}), with the timing "
        f"term at t = {tm_mom['timing_tstat']:.1f}. The Seasonality sleeve, which is built on "
        f"momentum, comes out the same way ({pct(tm_seas['alpha_ann'])}, t = "
        f"{tm_seas['alpha_tstat']:.1f}). Their dev alphas are the payoff from riding crypto's "
        "large trends, not returns unrelated to the market."
    )

    # 3. Research families
    pdf.h1("3. Research families")
    pdf.body(
        "Each family starts from a simple trading idea and was tested on dev and gate "
        "only. A config survives if it is positive on both windows and passes its family's "
        "extra rule: beat the untilted sleeve (seasonality), beat the unfiltered baseline "
        "(conditioned reversals), or have a neighbouring parameter that is also positive. Where "
        "more than one survives, the seasonality and order-flow notebooks keep the best gate "
        "Sharpe and the carry notebook the best dev Sharpe, for reasons it gives. "
        f"Appendix B lists all {reg_rows} registry rows."
    )
    pdf.h2("3.1 Seasonality: who trades when")
    pdf.body(
        "Institutions trade more on weekdays and in US hours, retail at night and at weekends. "
        "If informed flow trends, momentum should behave differently by calendar bucket. Twelve "
        "configs were tested: weekday/weekend scaling of the momentum sleeve, holding the market "
        "only in some calendar buckets, an hourly US-hours test and a turn-of-month tilt.\n\n"
        f"The survivor halves momentum's exposure on weekdays: dev {sr(seas_dev)} and gate "
        f"{sr(seas_gate)}, against {sr(unt_dev)} and {sr(unt_gate)} for the untilted sleeve. "
        "Almost all of the equal-weight market's return came outside US hours, but trading on "
        "that split lost money on the gate.\n\n"
        "The neighbour rule was written two ways before any result: a neighbouring parameter that "
        "is positive on both windows, and, in a comment in the selection cell, a neighbour that is "
        "itself a survivor. No tilt meets the second, and eight minutes after the results were "
        "logged the comment was changed to match the first (notebook 02). Without Seasonality the "
        f"book's lockbox Sharpe would have been {v['no_seas']['wf']:.2f} walk-forward and "
        f"{v['no_seas']['ew']:.2f} equal weight, against {lk_wf['sharpe']:.2f} and "
        f"{book[('EW', 'Lockbox')]['sharpe']:.2f} as run."
    )
    pdf.table_block(family_rows(reg, "seasonality"), title="Seasonality configs", widths=(80, 20, 20))
    cal = v["calendar"]
    meant = [(t, cal.loc[t].iloc[1]) for t in cal.index.get_level_values(0).unique()]
    pdf.caption(
        "The P2 and P3 rows are as run, and they held each bucket a day late (P3 an hour late): the "
        "weekday test held Tuesday to Saturday. On the days they meant they score "
        + "; ".join(f"{t.replace('_', ' ')} {sr(r['dev'])} and {sr(r['gate'])}" for t, r in meant)
        + " (dev and gate), so they still fail and the survivor stands (notebook 07, section 5).")
    pdf.h2("3.2 Order flow: the taker-buy share")
    pdf.body(
        "Binance klines report how much volume came from aggressive buyers. The uninformed-flow "
        "argument says reversal should work best on moves without real flow behind them, so "
        "reversal was tested on quiet names and on moves the flow did not confirm. The "
        "imbalance itself was also tested as a signal, in both directions.\n\n"
        f"Every reversal variant loses on dev. Following the 10-day average imbalance survives "
        f"(dev {sr(of_dev)}, gate {sr(of_gate)}), and the 5-day version is also positive; the "
        "1-day version loses on both windows. Fading the imbalance is the mirror image and loses."
    )
    pdf.table_block(family_rows(reg, "orderflow"), title="Order-flow configs", widths=(80, 20, 20))
    pdf.h2("3.3 Funding carry and crowding")
    pdf.body(
        "Longs pay funding when it is positive, so high funding is both carry for the short side "
        "and a sign of crowded positioning. The carry sleeve shorts high-funding names and buys "
        "low-funding ones, and the backtest adds the funding payments to the price P&L.\n\n"
        f"Carry on 7-day smoothed funding survives (dev {sr(ca_dev)}, gate {sr(ca_gate)}), and "
        "so do its neighbours. Reversal in crowded names loses. Fading the change in funding "
        "also passes, but it is a crowding trade with a much weaker dev result, so carry was kept."
    )
    pdf.table_block(family_rows(reg, "carry"), title="Funding configs", widths=(80, 20, 20))
    pdf.h2("3.4 Fast reversal (hourly)")
    pdf.body(
        "If short-horizon reversal comes from providing liquidity, it should be stronger "
        "intraday. The rule, set before testing: a config must earn a gross return of at least "
        "3 times its cost drag before it is worth tuning.\n\n"
        f"The 4-hour lookback has a gross Sharpe of {fr4.gross_sharpe_dev:+.2f}, so the edge is "
        f"real, but hourly rebalancing costs {fr4.cost_drag_ann:.0%} a year against a gross "
        f"return of {fr4.gross_ann:.0%}, about {fr4.cost_drag_ann / fr4.gross_ann:.1f} times as "
        "much. No config comes close to the 3x rule, and net Sharpe is negative for all eight "
        "on both windows."
    )
    fr_rows = [["Lookback, rebalance", "Gross Sharpe", "Gross/yr", "Cost/yr", "Ratio", "Net dev", "Net gate"]]
    for _, g in grid.iterrows():
        fr_rows.append([f"{int(g.lookback_h)}h, every {int(g.rebal_h)}h", f"{g.gross_sharpe_dev:+.2f}",
                        f"{g.gross_ann:.0%}", f"{g.cost_drag_ann:.0%}", f"{g.gross_over_drag:.2f}",
                        f"{g.net_dev:+.2f}", f"{g.net_gate:+.2f}"])
    pdf.table_block(fr_rows, title="Fast reversal: gross on the dev window, cost from turnover over dev and gate (hourly, 8,760 hours a year)",
                    widths=(34, 16, 15, 14, 12, 14, 15))
    pdf.caption("Section 5.5 reruns this grid on hourly bars for every coin that was in the top 100.")

    # 4. The book
    pdf.h1("4. The book")
    pdf.body(
        "The book holds Seasonality, Orderflow and Carry. The main version uses walk-forward "
        "max-Sharpe weights, refitted every 63 days on the previous 756 days; an equal-weight "
        "version runs alongside. Both combination rules were fixed before the lockbox and are "
        "logged in the registry."
    )
    c = v["corr"]
    pdf.table_block([[""] + [n.capitalize() for n in c.columns]]
                    + [[i.capitalize()] + [f"{x:.2f}" for x in c.loc[i]] for i in c.index],
                    title="Sleeve correlations on the dev window", widths=(30, 25, 25, 25))
    pdf.table_block(perf_rows(book, [("WF", w) for w in WINDOWS] + [("EW", w) for w in WINDOWS],
                              [f"Walk-forward, {w.lower()}" for w in WINDOWS]
                              + [f"Equal weight, {w.lower()}" for w in WINDOWS]),
                    title="Book performance by window (t-stat of alpha in brackets)", widths=PERF_WIDTHS)
    pdf.caption(
        "The walk-forward book needs 756 days of history before its first weights, so it starts "
        "in 2021-10 and its dev figure covers only 2021-10 to 2024-07. Equal weight beats "
        "walk-forward on every window except the lockbox: with three nearly uncorrelated sleeves "
        "there is little for mean-variance weights to add. Over the dates both books cover the "
        f"margin is smaller: {sd['Dev']['ew']:.2f} against {sd['Dev']['wf']:.2f} on dev and "
        f"{sd['Full']['ew']:.2f} against {sd['Full']['wf']:.2f} over the full sample."
    )
    names = ["seasonality", "orderflow", "carry"]
    pdf.table_block(perf_rows(sl, [(n, w) for n in names for w in ["Dev", "Gate", "Lockbox"]],
                              [f"{n.capitalize()}, {w.lower()}" for n in names
                               for w in ["Dev", "Gate", "Lockbox"]]),
                    title="Sleeve performance by window (t-stat of alpha in brackets)", widths=PERF_WIDTHS)
    pdf.body(
        "The sleeve alphas look large because alpha scales with volatility. Orderflow and Carry "
        "hold $1 of positions, half long and half short, with no volatility target, and the "
        "high-funding coins Carry trades are small and very volatile, so it runs at about "
        f"{pct(cs['Gate']['vol'], 0)} volatility. Scaled to 10% volatility, Carry's gate alpha of "
        f"{pct(cs['Gate']['alpha'])} would be {pct(cs['Gate']['at10'])}, and its lockbox alpha of "
        f"{pct(cs['Lockbox']['alpha'])} would be {pct(cs['Lockbox']['at10'])}. The gate is also "
        f"short: Carry's three best months returned {pct(cs['best3'], 0)} together, out of "
        f"{pct(cs['gate_total'], 0)} for the whole 11 months, and the 95% interval around its gate "
        f"alpha runs from {pct(cs['Gate']['lo'], 0)} to {pct(cs['Gate']['hi'], 0)}. The combined "
        f"walk-forward book runs at {pct(lk_wf['vol'])} volatility on the lockbox, with an alpha of "
        f"{pct(lk_wf['alpha'])}."
    )
    pdf.figure(figs["equity"])
    pdf.caption("Top: growth of $1 on a log scale. Middle: walk-forward drawdown. Bottom: rolling "
                "180-day Sharpe of the walk-forward book.")
    pdf.figure(figs["weights"])
    pdf.caption("The weights the walk-forward book used. They can go negative, and their absolute "
                "values sum to 1.")

    pdf.h2("4.1 The lockbox result")
    pdf.body(
        f"Over the lockbox ({v['lk_n']} days) the walk-forward book made {pct(lk_wf['ret'])} a "
        f"year with {pct(lk_wf['vol'])} volatility and a worst drawdown of {pct(lk_wf['mdd'])}. "
        f"Its Sharpe of {lk_wf['sharpe']:.3f} has a block-bootstrap 95% interval of "
        f"[{lo:.2f}, {hi:.2f}], which includes zero. Regressed on BTC and the equal-weight market "
        f"it has an alpha of {pct(lk_wf['alpha'], 2)} a year (Newey-West t = "
        f"{lk_wf['alpha_t']:.2f}) and a BTC beta of {lk_wf['beta']:+.3f}.\n\n"
        f"By sleeve, Orderflow ({sr(sl[('orderflow', 'Lockbox')]['sharpe'])}) and Carry "
        f"({sr(sl[('carry', 'Lockbox')]['sharpe'])}) held up, while Seasonality lost money "
        f"({sr(sl[('seasonality', 'Lockbox')]['sharpe'])}). Its strong gate result did not carry "
        "over, which is the risk notebook 02 flagged for an 11-month gate."
    )
    pdf.table_block([
        ["", "Full sample", "Lockbox"],
        ["Walk-forward Sharpe", f"{book[('WF', 'Full')]['sharpe']:.2f}", f"{lk_wf['sharpe']:.2f}"],
        ["Block-bootstrap 95% CI", "", f"[{lo:.2f}, {hi:.2f}]"],
        [f"Deflated Sharpe ({N_TRIALS} trials)", f"{v['dsr_full']:.3f}", f"{v['dsr_lk']:.3f}"],
    ], title="Significance", widths=(60, 30, 30))
    pdf.body(
        "The deflated Sharpe is the probability that the observed Sharpe beats the best Sharpe "
        f"that {N_TRIALS} strategies with no real edge would show by luck. It is "
        f"{v['dsr_full']:.2f} on the full "
        f"sample and {v['dsr_lk']:.2f} on the lockbox ({v['dsr_lk_all']:.2f} counting the first version's "
        f"{FIRST_VERSION_TRIALS} configurations, section 2), both far below 0.95. One year is also "
        f"short: the bootstrap interval above is {hi - lo:.1f} Sharpe points wide. The book did "
        "make money with almost no market exposure, but this sample cannot separate that from "
        "luck. Section 5 reruns the same book on a universe with every pair, where its lockbox "
        f"Sharpe is {sr(st[('2 every pair', 'walk_forward')]['Lockbox'])}."
    )
    pdf.h2("4.2 The whole sample, as run")
    pdf.table_block(full_rows(fl), title=f"Results over the whole sample ({DEV_START} to {LOCKBOX_END}), "
                                         "research coin list", widths=FULL_WIDTHS)
    pdf.caption(
        "Section 5 shows how little of this survives a universe with every pair. Alpha and the two "
        "betas come from a regression of daily returns on BTC and the equal-weight market; the t-stat "
        "in brackets uses Newey-West errors. These figures mix the windows the strategies were chosen "
        "on with the windows they were tested on. Section 2 explains why the momentum alphas are "
        "mostly market timing, and the start of this section why Carry's alpha is large."
    )

    # 5. Making the backtest realistic
    sz, fills, cap, oos = v["size"], v["fills"], v["cap"], v["oos"]
    pdf.h1("5. Making the backtest realistic")
    pdf.body(
        "Notebooks 09 and 11 were added on 2026-09-27, after the forward test (section 6). Notebook "
        "09 rebuilds "
        "the daily data from Binance's public archive, which keeps the files of delisted pairs, and "
        "checks it against the research data: every price and funding rate the two have in common "
        "is the same. Notebook 11 reruns the frozen book with one assumption at a time made more "
        "realistic. Like appendix A, it re-scores the lockbox with corrected inputs and chooses "
        "nothing on it, and v2 (section 7) was registered before notebook 11 was first run."
    )
    pdf.h2("5.1 The coin list")
    pdf.body(
        "The research coin list is the 150 USDT pairs with the most 24-hour volume on the day of the "
        "fetch, 2026-07-06, so it holds mostly coins that were still big then, plus "
        f"{n_stopped} that had in fact stopped trading. From 2020 to July 2026, "
        f"{pct(v['surv'], 0)} of the universe's coin-days belonged to coins it left out: coins that "
        "were delisted or renamed, such as EOS, MATIC, XMR, WAVES and MKR, and coins that had just "
        "shrunk, such as VET, SAND, GRT, KAVA and EGLD. The share is highest in the early years, "
        "when the list was furthest from its fetch date:"
    )
    yrs = v["surv_year"]
    pdf.table_block([["Year"] + [str(y) for y in yrs.index],
                     ["Coin-days the list did not have"] + [pct(x, 0) for x in yrs]],
                    title="Share of the daily top-100 universe outside the research coin list",
                    widths=(54,) + (16,) * len(yrs))
    pdf.body(
        "The list had three smaller faults. Its filter for leveraged tokens dropped any name ending "
        "in UP, which removed JUP and SYRUP. Six pegged assets got in: the euro, the gold tokens PAXG "
        "and XAUT, and three newer dollar tokens. EUR and PAXG sat in the universe for most of the "
        "sample, and Carry could trade the two gold perps. And five tokenized US stocks, which "
        "Binance began listing in June 2026, entered the universe during the forward test. The "
        "archive needed care too: it keeps writing a frozen, zero-volume bar every day for a perp "
        "that has been delisted, and some perps stop tracking their coin after a token migration. "
        "So a strategy only picks a perp on days the contract traded and priced within 20% of spot, "
        "while a position already held earns its contract's own move and funding on every day the "
        "contract traded (section 5.2)."
    )
    pdf.h2("5.2 The frozen book on corrected data")
    step_labels = [("0 as run", "0 As run"), ("0b archive data", "0b Same list, archive"),
                   ("1 non-crypto out", "1 Non-crypto out"), ("2 every pair", "2 Every pair"),
                   ("3 carry on perps", "3 Carry on perps"), ("4 limit fills", "4 Limit fills")]
    rows = [["Step", "WF dev", "WF gate", "WF lockbox", "WF fwd", "EW dev", "EW gate", "EW lockbox", "EW fwd"]]
    for key, lab in step_labels:
        rows.append([lab] + [sr(st[(key, "walk_forward")][w]) for w in RWIN]
                    + [sr(st[(key, "equal_weight")][w]) for w in RWIN])
    pdf.table_block(rows, title="The frozen book, one fix at a time (Sharpe; each step keeps the ones before it)",
                    widths=(38, 16, 16, 18, 15, 16, 16, 18, 15))
    pdf.body(
        "Rebuilt from the archive, the research list gives the same numbers (step 0b), so the data "
        "source is not the issue. Taking out the non-crypto assets costs a little: the walk-forward "
        f"book's lockbox Sharpe goes from {lk_wf['sharpe']:.2f} to "
        f"{st[('1 non-crypto out', 'walk_forward')]['Lockbox']:.2f}. Step 2 is the one that matters. "
        "With every pair in the universe the walk-forward book's lockbox Sharpe is "
        f"{sr(st[('2 every pair', 'walk_forward')]['Lockbox'])} and the equal-weight book's "
        f"{sr(st[('2 every pair', 'equal_weight')]['Lockbox'])}, and the equal-weight book loses money "
        f"on the gate as well ({sr(st[('2 every pair', 'equal_weight')]['Gate'])}). The walk-forward "
        "numbers also depend on the book's refit dates: started up to eight weeks later, the research "
        f"book's lockbox Sharpe ranges from {v['refits']['0 as run'][0]:.2f} to "
        f"{v['refits']['0 as run'][1]:.2f} and the every-pair book's from "
        f"{v['refits']['2 every pair'][0]:.2f} to {v['refits']['2 every pair'][1]:.2f}. The gap between "
        "them holds for any start; the equal-weight book has no refit dates. The every-pair books "
        "start on the research's first Carry day, 2019-09-10, so that they refit on the research's "
        "dates. The first run of notebook 11 started them in 2018, with zero Carry returns before any "
        "funding data, and put the every-pair walk-forward lockbox Sharpe at 0.16."
    )
    rows = [["Sleeve, step", "Dev", "Gate", "Lockbox", "Forward"]]
    for s in SLEEVE_NAMES:
        for key, lab in [("0 as run", "as run"), ("2 every pair", "every pair"), ("4 limit fills", "limit fills")]:
            rows.append([f"{s.capitalize()}, {lab}"] + [sr(st[(key, s)][w]) for w in RWIN])
    pdf.table_block(rows, title="The three sleeves at steps 0, 2 and 4 (Sharpe)", widths=(52, 20, 20, 20, 20))
    at_zero, own = v["gaps"]["dropped days at zero"], v["gaps"]["held contracts' own data"]
    pdf.body(
        "Orderflow and Carry lose money on the gate and the lockbox once every pair is in. "
        "Seasonality keeps its dev and gate numbers and still loses on the lockbox. The coins that "
        "did the damage are mostly ones the research list never had. On the gate, four of "
        "Orderflow's five worst coins were missing from it, OM above all, which collapsed in April "
        "2025, and so were all five of Carry's: delisting casualties such as VIDT and BNX, which the "
        "z-score weights bought because their funding had turned negative. Measuring Carry on perp "
        "prices (step 3) helps it a little on the gate and the lockbox. Two things were fixed on the "
        "way. The archive's monthly perp files for February and April 2022 are missing days for "
        "about 50 contracts (the last three of February, the first two of April), and notebook 09 "
        "now fills them from the archive's daily files. And on a day the 20% rule drops, the first run gave a coin "
        "Carry already held no funding (steps 2 and 3) and no price move (step 3). A held position "
        "earns what its contract did. On a few of those days the perp still traded, however far from "
        "spot, as in LUNA's and FTT's crashes in 2022 and OMG's steep discount to spot in November "
        "2021, and its own move and funding now count; on the rest it did not trade, mostly because "
        "it had been delisted, and the position earns nothing. With those days at zero, Carry's "
        f"step-3 Sharpe would read {sr(at_zero['Dev'])} on dev and {sr(at_zero['Gate'])} on the gate "
        f"instead of {sr(own['Dev'])} and {sr(own['Gate'])}."
    )
    pdf.h2("5.3 Limit orders that have to fill")
    pdf.body(
        "The research charges 7 bps and assumes every limit order fills at the close. In step 4 an "
        "order placed at the close only fills if the next day trades through its price, and a "
        "missed order is replaced at the next close. Almost all of them fill:"
    )
    rows = [["", "Orders filled", "Positions missed", "Return, missed buy", "Return, missed sell",
             "P&L, all filled", "P&L, fill model"]]
    for s in ["orderflow", "carry"]:
        r = fills.loc[s]
        rows.append([s.capitalize(), pct(r["filled share of orders"]), pct(r["missed share of positions"], 2),
                     pct(r["return where a buy missed"]), pct(r["return where a sell missed"]),
                     pts(r["price P&L, every order filled"]), pts(r["price P&L, fill model"])])
    pdf.table_block(rows, title="Limit orders on dev, on coins that traded the next day (P&L in points: sums of "
                                "daily price returns, Carry's before funding)",
                    widths=(22, 22, 25, 27, 27, 24, 23))
    of_fill = fills.loc["orderflow"]
    pdf.body(
        "The orders that miss are the ones the price ran away from. Where a buy missed, the coin "
        f"rose {pct(of_fill['return where a buy missed'])} that day on average, and where a sell "
        f"missed it fell {pct(-of_fill['return where a sell missed'])}. Missing "
        f"{pct(of_fill['missed share of positions'], 1)} of its positions costs Orderflow "
        f"{(of_fill['price P&L, every order filled'] - of_fill['price P&L, fill model']) * 100:.0f} of its "
        f"{of_fill['price P&L, every order filled'] * 100:.0f} points of price P&L on dev, and its dev "
        f"Sharpe falls from {st[('3 carry on perps', 'orderflow')]['Dev']:.2f} to "
        f"{st[('4 limit fills', 'orderflow')]['Dev']:.2f}. Carry trades less and loses less. Its "
        "orders fill on its perps' own bars, whatever their basis to spot. An order cannot fill on a "
        "day the contract does not trade; if it trades again within three days, a position left open "
        "takes the move across the gap, and after more than three days the contract counts as "
        "delisted: the position is settled at its last price and dropped, so a relaunch under the "
        "same name, like LUNA's in 2022, cannot revive it. A day's funding is credited to the "
        "position after that day's fill, although the prints before a fill belong to the old "
        "position; daily funding data cannot split the day."
    )
    pdf.h2("5.4 Size and capacity")
    pdf.body("The same Orderflow rule on the largest coins and on the smaller ones shows where the "
             "research list's edge came from:")
    rows = [["Universe, coins", "Dev", "Gate", "Lockbox", "Forward"]]
    for lst in ["research list", "every pair"]:
        for b in ["top 30", "top 50", "ranks 51-100", "top 100"]:
            rows.append([f"{lst.capitalize()}, {b}"] + [sr(sz[(lst, b)][w]) for w in RWIN])
    pdf.table_block(rows, title="Orderflow by coin size (Sharpe)", widths=(52, 20, 20, 20, 20))
    pdf.body(
        "On the research list, Orderflow's gate and lockbox results come from the coins ranked 51 "
        f"to 100 ({sr(sz[('research list', 'ranks 51-100')]['Gate'])} and "
        f"{sr(sz[('research list', 'ranks 51-100')]['Lockbox'])}). On every pair those coins lose money "
        f"on both ({sr(sz[('every pair', 'ranks 51-100')]['Gate'])} and "
        f"{sr(sz[('every pair', 'ranks 51-100')]['Lockbox'])}). The research list's small coins were the "
        "ones that went on to be big in 2026, which is how survivorship shows up in a signal.\n\n"
        "Market impact is modelled per coin with the square-root law: trading Q dollars of a coin "
        "with daily volume V and daily volatility sigma costs about y x sigma x sqrt(Q/V) on top of "
        "the 7 bps fee, with y between 0.5 and 1. Binance volume alone makes both settings on the "
        "harsh side. On the research-list sleeves, over 2020 to July 2026:"
    )
    rows = [["Money run", "Orderflow, y = 0.5", "Orderflow, y = 1", "Carry, y = 0.5", "Carry, y = 1"]]
    for aum in ["no impact", "$0.1M", "$1M", "$10M", "$50M"]:
        rows.append([aum] + [sr(cap.loc[(s, aum), f"sharpe y={y}"]) for s in ["orderflow", "carry"] for y in ["0.5", "1"]])
    pdf.table_block(rows, title="Research-list sleeves with square-root market impact (Sharpe)",
                    widths=(30, 32, 32, 32, 32))
    pdf.body(
        "Even taken at face value, the research-list Orderflow could not have run much money: its "
        f"Sharpe falls from {cap.loc[('orderflow', 'no impact'), 'sharpe y=1']:.2f} to "
        f"{cap.loc[('orderflow', '$1M'), 'sharpe y=0.5']:.2f} at $1M with y = 0.5, and below zero with "
        "y = 1. Carry holds up to somewhere between $1M and $10M."
    )
    pdf.h2("5.5 Hourly reversal on every pair")
    fg = d["realism_fastrev"]
    rows = [["Lookback, rebalance", "Gross Sharpe, 60 coins", "Gross Sharpe, every pair",
             "Gross/cost, 60 coins", "Gross/cost, every pair", "Net dev, every pair", "Net gate, every pair"]]
    for (lb, rb), r in fg.loc["every pair"].iterrows():
        r0 = fg.loc[("research panel", lb, rb)]
        rows.append([f"{lb}h, every {rb}h", f"{r0['gross_sharpe_dev']:+.2f}", f"{r['gross_sharpe_dev']:+.2f}",
                     f"{r0['gross_over_drag']:.2f}", f"{r['gross_over_drag']:.2f}", sr(r["net_dev"]), sr(r["net_gate"])])
    fr0, fr1 = fg.loc[("research panel", 4, 1)], fg.loc[("every pair", 4, 1)]
    pdf.body(
        "Notebook 05's hourly test (section 3.4) ran on 60 coins picked by volume in July 2026, about "
        f"{fr0['names per hour']:.0f} of them tradable in a typical hour. Notebook 11 reruns its eight "
        "configs, with the same rules and costs, on hourly bars for every coin that was in the daily "
        f"top 100, about {fr1['names per hour']:.0f} an hour. On the 60 coins it reproduces notebook 05 "
        "exactly."
    )
    pdf.table_block(rows, title="Notebook 05's grid on its 60 coins and on every pair (dev, 8,760 hours a year)",
                    widths=(28, 24, 24, 24, 24, 23, 23))
    pdf.body(
        "With the wider universe the gross edge is larger, which is the opposite of what the daily "
        "sleeves showed: the 4-hour lookback rebalanced every hour has a gross Sharpe of "
        f"{fr1['gross_sharpe_dev']:.2f} against {fr0['gross_sharpe_dev']:.2f}, and its gross return is "
        f"about equal to its trading cost ({fr1['gross_over_drag']:.2f} times). That is still far from the "
        "rule of 3 times the cost, and every config loses money after costs on dev and on the gate, so "
        "the conclusion of section 3.4 stands."
    )
    pdf.h2("5.6 The pairs baseline, fully charged and without its look-ahead")
    pc = v["pairs_c"]
    lag, both = pc["picked a day earlier"], pc["both"]
    pdf.body(
        "Charging the pairs engine for every position carried across a rebalance, an upper bound on "
        f"what it missed, moves its dev Sharpe from {sr(pc['as run']['Dev'])} to "
        f"{sr(pc['charged']['Dev'])}, its gate Sharpe from {sr(pc['as run']['Gate'])} to "
        f"{sr(pc['charged']['Gate'])} and its lockbox Sharpe from {sr(pc['as run']['Lockbox'])} to "
        f"{sr(pc['charged']['Lockbox'])}. The engine also picked its pairs with the close of the "
        "rebalance date and then booked that date's return for them, a one-day look-ahead at each "
        "rebalance. Picking them with data up to the day before gives "
        f"{sr(lag['Dev'])}, {sr(lag['Gate'])} and {sr(lag['Lockbox'])} on dev, gate and lockbox, "
        f"and with both changes {sr(both['Dev'])}, {sr(both['Gate'])} and {sr(both['Lockbox'])}. "
        "Pairs loses money on dev either way."
    )
    # 6. Forward test
    fc, wh = v["fwd_coin"], v["whole"]
    n_fwd = F["Book, walk-forward"]["n"]
    pdf.h1("6. Out-of-sample evidence")
    pdf.body("The frozen book has two stretches of data it never influenced: the lockbox year and the "
             f"{n_fwd} days after the research data was fetched.")
    pdf.h2("6.1 The forward test, July to September 2026")
    pdf.body(
        "Notebook 08, added on 2026-09-27, downloads the days after the research data ends for the "
        "same coins and runs the frozen book on them, with the same universe rule, sleeves, costs "
        "and walk-forward settings. Where the new download overlaps the cached data (May to early "
        "July 2026) every daily bar and funding rate matches, and up to 2026-07-05 the rebuilt "
        "sleeves and books match the research exactly. Nothing was tuned and no trials were "
        f"logged. The window runs from {FORWARD_START} to {FORWARD_END}, {n_fwd} days."
    )
    rows = [["", "Return", "Ann. vol", "Sharpe", "Max DD", "Beta BTC"]]
    for name, s in F.items():
        bench_row = name in ("BTC", "Equal-weight market")
        rows.append([name, pct(s["total"]), pct(s["vol"]), sr(s["sharpe"]), pct(s["mdd"]),
                     "" if bench_row else sr(s["beta"])])
    pdf.table_block(rows, title=f"The frozen book on the forward window ({n_fwd} days; return is the "
                                "total over the window)", widths=(58, 18, 18, 16, 18, 18))
    pdf.body(
        f"The walk-forward book lost {pct(-F['Book, walk-forward']['total'])} while BTC gained "
        f"{pct(F['BTC']['total'])}. Orderflow, with {pct(v['fwd_of_weight']['min'], 0)} to "
        f"{pct(v['fwd_of_weight']['max'], 0)} of the book's weight, lost {pct(-F['Orderflow']['total'])} "
        "after being the best sleeve on the lockbox. It has had runs like this before: "
        f"{pct(v['fwd_of_worse'])} of its 82-day stretches since 2020 were worse, about one in "
        "twenty. Some of the loss was the market. Over this window the book's beta to BTC was "
        f"{F['Book, walk-forward']['beta']:+.2f}, against about zero in the research, and beta times "
        f"BTC's rise accounts for {-v['fwd_beta_part'] * 100:.1f} of the {-v['fwd_wf_sum'] * 100:.1f} "
        "points the book lost, counting in sums of daily returns."
    )
    pdf.body(
        f"The equal-weight book made {pct(F['Book, equal weight']['total'])} only because of Carry, "
        f"and Carry's {pct(F['Carry']['total'])} came from one coin. {fc['name']} fell "
        f"{pct(-fc['crash'])} on 2026-07-21 and swung hard for the next week. Traders shorted "
        "its perp heavily and funding went deeply negative: shorts paid longs "
        f"{pct(-fc['funding'].max(), 0)} to {pct(-fc['funding'].min(), 0)} a day for three days. "
        "Carry buys low-funding coins, and its z-score weights have no cap per coin, so from "
        f"{fc['held'].index.min():%Y-%m-%d} it held {fc['name']} at {pct(fc['held'].min(), 0)} to "
        f"{pct(fc['held'].max(), 0)} of its gross for {len(fc['held'])} days, all or nearly all of "
        f"its long side. Counted in sums of daily returns, {fc['name']} added "
        f"{pts(d['fwd_split'][('carry', fc['name'])])} points; all the other coins together lost "
        f"{pts(-d['fwd_split'][('carry', 'all other coins')])}, and trading costs took "
        f"{pts(-d['fwd_split'][('carry', 'trading costs')])}. This is also the part of the "
        "backtest to trust least: funding that negative comes with the perp trading far below "
        "spot, and the backtest adds funding to spot returns without that gap."
    )
    conc = d["fwd_conc"]
    pdf.table_block(
        [["Window", "Days", "Median largest weight", "Share of days >= 0.25", "Days one coin held a whole side"]]
        + [["Lockbox to 2026-07-05" if w == "lockbox" else w.capitalize(), f"{int(r['days'])}",
            f"{r['median largest weight']:.2f}",
            pct(r["share of days largest >= 0.25"], 0), f"{int(r['days one coin holds a whole side'])}"]
           for w, r in conc.iterrows()],
        title="Carry: the largest single-coin weight each day, as a share of gross exposure",
        widths=(38, 14, 34, 34, 40))
    pdf.body(
        "The weighting always allowed this. In the research, one coin held a whole side on "
        f"{wh['dev']} days, all on dev: {wh['jan20']} in January 2020, when only "
        f"{wh['n_jan20']['min']} to {wh['n_jan20']['max']} coins had funding data, and "
        f"{wh['nov22']} in November 2022, when the sleeve held SOL through the FTX collapse. The "
        f"forward window had {int(conc.loc['forward', 'days one coin holds a whole side'])} such "
        f"days, all {fc['name']}.\n\n"
        f"An {n_fwd}-day Sharpe has a standard error of about {np.sqrt(365 / n_fwd):.1f}, so this "
        "window can neither confirm nor overturn the lockbox. It does show, more plainly than the "
        "research did, that Carry's uncapped weights can put half the sleeve into one coin in the "
        "middle of a crash. A per-coin cap is the obvious fix, but a cap chosen after seeing this "
        "window would be fitted to it, so the book is reported as it was frozen. Section 6.2 adds "
        "this window to the lockbox."
    )
    pdf.figure(figs["forward"])
    pdf.caption("Growth of $1 over the forward window. Left: the two books and BTC. Right: the three "
                "sleeves; the dotted line marks DEXE's crash on 2026-07-21.")
    pdf.h2("6.2 The whole out-of-sample record")
    fb_ = v["fwd_beta"]
    pdf.body(
        "The lockbox year and the forward test are the only data the frozen book never influenced. "
        f"Together, as run, they give the walk-forward book a Sharpe of {oos['sharpe']:.2f} over "
        f"{oos['n']} days (t = {oos['t']:.1f}; block-bootstrap 95% interval {oos['ci'][0]:.2f} to "
        f"{oos['ci'][1]:.2f}) and the equal-weight book {oos['sharpe_ew']:.2f}. In the forward window "
        "the walk-forward book's beta to BTC was lower than in "
        f"{pct(1 - fb_.loc['walk_forward', 'share of research windows lower'], 0)} of the earlier 82-day "
        f"stretches, and Orderflow's lower than in "
        f"{pct(1 - fb_.loc['orderflow', 'share of research windows lower'], 0)}."
    )
    pdf.figure(figs["survivorship"])
    pdf.caption("Growth of $1 on a log scale for the frozen book as run (research coin list) and on "
                "every pair. Dotted line: gate starts; dashed red: lockbox opens; dashed grey: forward "
                "test starts.")

    # 7. v2
    v2_lab = {"Book": "v2 book", "Orderflow": "Orderflow", "Carry": "Carry"}
    pdf.h1("7. Version 2, registered before testing")
    pdf.body(
        "v2 is the book rebuilt around what the audits found. Its rules were committed to the "
        "repository on 2026-09-27, in notebook 10 and strategies.py, before any v2 number existed, "
        "and the next commit adds the results. It makes five changes, each answering a problem the "
        "audits found or a result of the frozen book on dev and gate, none chosen on v2's own "
        "results. One of those problems showed up in the forward window: the DEXE case of section "
        "6.1, which rank weights also answer. It drops Seasonality, which fails notebook 02's rule once its resizing "
        "trades are charged. It weights Carry by rank instead of z-score, so once more than a handful "
        "of coins have funding data no coin can take a whole side of the sleeve. It measures Carry on "
        "perp prices. It trades the every-pair universe. And "
        "it combines its two sleeves with equal weights, which beat the walk-forward weights on dev "
        f"({sd['Dev']['ew']:.2f} against {sd['Dev']['wf']:.2f} over the same dates) and on the gate "
        f"({sd['Gate']['ew']:.2f} against {sd['Gate']['wf']:.2f}). v2 is never evaluated "
        "on the lockbox or on the forward-test window, since both have been seen; notebook 12 uses "
        "those months only as history for v2's first positions.\n\n"
        "Two fixes came after v2's first run, the same two as in section 5.2. A Carry position held "
        "into a day the 20% rule drops its perp's data now earns what its contract did, its own move "
        "and funding, or nothing once the contract has stopped trading (strategies.v2_sleeves with "
        "pnl_funding and the archive's traded panels). And the two 2022 holes in the archive are "
        "filled. Neither changes v2's rules. The first changes what v2 earns on a few days; the "
        "second gives Carry funding data it was missing on those days of 2022, which also changes "
        "its positions for about a week after each hole. Both were made on 2026-09-29 (UTC), after "
        "v2's first test day had ended but before any test result was computed. An earlier version "
        "of the first, committed on 2026-09-28, gave a held coin its "
        "spot move instead, which is wrong both on days the perp trades far from spot and after a "
        "delisting. The table uses both fixes. "
        "As first run, v2 had a Sharpe of "
        f"{v2s[('Book', 'Dev')]['sharpe']:.2f} on dev and {v2s[('Book', 'Gate')]['sharpe']:.2f} on the "
        f"gate, and its Carry {v2s[('Carry', 'Dev')]['sharpe']:.2f} and {v2s[('Carry', 'Gate')]['sharpe']:.2f}; "
        "the registry and notebook 10 keep those numbers."
    )
    pdf.table_block(perf_rows(v2f, [(n, w) for n in v2_lab for w in ["Dev", "Gate"]],
                              [f"{v2_lab[n]}, {w.lower()}" for n in v2_lab for w in ["Dev", "Gate"]]),
                    title="v2 on dev and gate, every pair (t-stat of alpha in brackets)", widths=PERF_WIDTHS)
    pdf.body(
        f"On dev and gate v2 has a Sharpe of {v2f[('Book', 'Dev')]['sharpe']:.2f} and "
        f"{v2f[('Book', 'Gate')]['sharpe']:.2f}, and nearly all of it comes from Carry: weighted by "
        f"rank, on perp prices and across every pair, Carry has {v2f[('Carry', 'Dev')]['sharpe']:.2f} "
        f"and {v2f[('Carry', 'Gate')]['sharpe']:.2f}, at about half the frozen Carry's volatility "
        f"({pct(v2f[('Carry', 'Gate')]['vol'], 0)} against {pct(sl[('carry', 'Gate')]['vol'], 0)} on the "
        "gate). Orderflow does not survive the wider universe "
        f"({sr(v2f[('Orderflow', 'Dev')]['sharpe'])} on dev, {sr(v2f[('Orderflow', 'Gate')]['sharpe'])} on "
        "the gate), so by notebook 03's rule it would not have been kept; it stays because v2's "
        "rules were fixed before this check. The three rows are in the registry.\n\n"
        "The test is every day from 2026-09-28, the first full day after v2 was registered. Notebook "
        "12 adds each month from the archive, delisted coins included, and reports v2's book and each "
        "sleeve alongside the frozen book. At a Sharpe near 1.5, a t-stat of 2 takes about two years "
        "of data."
    )

    # 8. twsq
    pdf.h1("8. The sleeves in twsq")
    pdf.body(
        "alphas/ holds the three sleeves written as alphas for twsq, an execution "
        f"framework. twsq backtests them on Binance daily bars over {v['twsq_days']} days (2024-08-06 to "
        "2026-07-07) with the same costs. They differ from the research in ways that matter: "
        "they trade a fixed list of 20 large coins instead of the point-in-time top 100, twsq "
        "charges every trade (including the Seasonality resizing), and the carry alpha holds "
        "spot, so it earns no funding. The signals only use data up to the previous day's close."
    )
    rows = [["Alpha", "Ann. return", "Ann. vol", "Sharpe", "Max DD", "Turnover/day", "Fees, total"]]
    for name, r in tw.iterrows():
        rows.append([name, pct(r["ann_return"]), pct(r["ann_vol"]), sr(r["sharpe"]), pct(r["max_drawdown"]),
                     pct(r["avg_daily_turnover"], 0), pct(r["fees"])])
    pdf.table_block(rows, title="twsq backtests, 2024-08-06 to 2026-07-07 (returns on $1M of capital)",
                    widths=(34, 16, 14, 13, 14, 16, 13))
    pdf.body(
        "On 20 large coins none of the three does much. SeasonalMomentum is roughly flat, which "
        "fits the charged Seasonality numbers in appendix A.1. OrderflowFollow loses a little; "
        "section 5.4 shows why, since the research list's order-flow edge came from its smaller "
        "coins, and those were survivors. FundingCarry's price leg is slightly positive, without "
        "the funding income that the research sleeve earns."
    )

    # 9. Limitations
    pdf.h1("9. Limitations")
    pdf.bullets([
        "One exchange. Prices, volumes, taker flow and funding all come from Binance, and taker "
        "flow has no second source here.",
        f"Short samples. The lockbox is one year and the forward test {F['Book, walk-forward']['n']} "
        "days. The walk-forward book's deflated Sharpe is below 0.95 on every window, and on the "
        f"lockbox both books are far below it ({v['dsr_lk']:.2f} walk-forward, {v['dsr_lk_ew']:.2f} "
        "equal weight). v2 has no out-of-sample data yet.",
        "Approximate models. The fill model works from daily highs and lows, and the market-impact "
        "model is a rough guide that uses Binance volume only.",
        "A hand-kept list. Tokenized stocks are excluded by a list in data.TOKENIZED_STOCKS; "
        "notebook 12 flags any new listing that looks like one.",
    ])

    # Appendix A: the first audit
    pdf.h1("Appendix A: Checks from the first audit")
    pdf.body(
        "Notebook 07 was added in September 2026, after the lockbox had been opened. It changes "
        "nothing in the book and logs no trials; it measures the things the audit raised and "
        "adds the whole-sample figures in section 4.2."
    )
    pdf.h2("A.1 The cost of resizing the Seasonality sleeve")
    pdf.body(
        "The Seasonality overlay multiplies each day's return by 0.5 on weekdays and 1.0 at "
        "weekends. That is the same as trading down to half size every Monday and back up every "
        "Saturday, but the backtest never charged for those trades. Rebuilding the positions "
        "actually held and charging 20 bps on every change costs the sleeve "
        f"{pct(v['resize_cost'])} a year on dev:"
    )
    pdf.table_block([
        ["", "Dev", "Gate", "Lockbox"],
        ["Seasonality, as run", sr(sl[("seasonality", "Dev")]["sharpe"]),
         sr(sl[("seasonality", "Gate")]["sharpe"]), sr(sl[("seasonality", "Lockbox")]["sharpe"])],
        ["Seasonality, every trade charged", sr(seas_c["Dev"]), sr(seas_c["Gate"]), sr(seas_c["Lockbox"])],
        ["Untilted momentum, every trade charged", sr(unt_c["Dev"]), sr(unt_c["Gate"]), "not run"],
    ], title="Seasonality with and without the resizing cost (Sharpe)", widths=(70, 17, 17, 17))
    pdf.body(
        "Charged properly, the overlay still beats untilted momentum on the gate but not on dev, "
        "so it would have failed notebook 02's rule. The error is in the cost accounting; the "
        "signal itself is computed correctly. The frozen book was left alone, and the whole book "
        "was re-scored with the charged sleeve instead:"
    )
    pdf.table_block(
        [["", "As run", "Resizing charged"]]
        + [[f"{'Walk-forward' if b == 'WF' else 'Equal weight'} Sharpe, {w.lower()}",
            f"{book[(b, w)]['sharpe']:.2f}", f"{bookc[(b, w)]['sharpe']:.2f}"]
           for b in ["WF", "EW"] for w in WINDOWS]
        + [["Walk-forward lockbox 95% CI", f"[{lo:.2f}, {hi:.2f}]", f"[{lo_c:.2f}, {hi_c:.2f}]"],
           ["Deflated Sharpe, full", f"{v['dsr_full']:.3f}", f"{v['dsr_full_c']:.3f}"],
           ["Deflated Sharpe, lockbox", f"{v['dsr_lk']:.3f}", f"{v['dsr_lk_c']:.3f}"],
           ["Walk-forward lockbox beta to BTC", f"{lk_wf['beta']:+.3f}",
            f"{bookc[('WF', 'Lockbox')]['beta']:+.3f}"]],
        title="The frozen book, as run and with the resizing charged", widths=(70, 25, 25))
    pdf.body(
        "On the lockbox the change is small and goes both ways. The walk-forward book gains "
        "slightly: with the weaker charged sleeve in its training data it gives Seasonality less "
        f"weight (an average of {v['lk_weight']['charged']['seasonality']:.2f} over the lockbox, "
        f"against {v['lk_weight']['as run']['seasonality']:.2f} as run), and Seasonality lost "
        "money on the lockbox. The equal-weight book loses a little. Over the full sample the "
        "walk-forward Sharpe drops from "
        f"{book[('WF', 'Full')]['sharpe']:.2f} to {bookc[('WF', 'Full')]['sharpe']:.2f}. The same "
        "shortcut applies wherever returns are scaled by a volatility target, but there it is small: "
        f"charging it moves plain momentum's dev Sharpe by {v['untilted']['Dev'] - unt_c['Dev']:.2f}."
    )
    pdf.h2("A.2 Higher costs for Orderflow and Carry")
    rows = [["Cost per dollar traded", "Orderflow dev", "Orderflow gate", "Carry dev", "Carry gate"]]
    for bps in [7, 14, 20]:
        rows.append([f"{bps} bps", sr(stress.loc[("orderflow", bps), "dev"]),
                     sr(stress.loc[("orderflow", bps), "gate"]), sr(stress.loc[("carry", bps), "dev"]),
                     sr(stress.loc[("carry", bps), "gate"])])
    pdf.table_block(rows, title="Sharpe at higher costs (dev and gate only)", widths=(40, 20, 20, 20, 20))
    pdf.body(
        "Both sleeves assume their daily rebalancing fills as limit orders at 7 bps. If every "
        "trade had to cross the spread at 20 bps, both would stay positive, but Orderflow would "
        f"lose {of_cut:.0%} of its dev Sharpe; it trades {pct(v['turnover']['orderflow'], 0)} of "
        f"its book a day on dev. Carry trades {pct(v['turnover']['carry'], 0)} a day and earns "
        "much of its P&L from funding, so it is less exposed to costs. Holding constant targets also "
        "takes small daily trades as positions drift with prices, which no backtest here charges; at "
        f"7 bps they would cost Orderflow {pct(v['drift'].loc['orderflow', 'dev'])} a year on dev and "
        f"Carry {pct(v['drift'].loc['carry', 'dev'])} (notebook 07, section 6). Nor do the books' daily "
        "moves of money between their sleeves get charged: before any netting between sleeves, they "
        f"would cost the equal-weight book {pct(v['rebalance'].loc['equal weight', 'dev'], 2)} a year on "
        f"dev and the walk-forward book {pct(v['rebalance'].loc['walk-forward', 'dev'], 2)}."
    )
    pdf.h2("A.3 Where the carry P&L comes from")
    pdf.table_block([
        ["", "Price leg/yr", "Funding leg/yr", "Funding share", "Price Sharpe", "Funding Sharpe"],
        *[[w.capitalize(), pct(split.loc[w, "price leg, mean %/yr"]), pct(split.loc[w, "funding leg, mean %/yr"]),
           pct(split.loc[w, "funding share of total"], 0), f"{split.loc[w, 'price leg sharpe']:.2f}",
           f"{split.loc[w, 'funding leg sharpe']:.2f}"] for w in ["dev", "gate"]],
    ], title="Carry sleeve: price and funding legs (arithmetic annual means)",
        widths=(18, 22, 24, 20, 18, 20))
    pdf.body(
        "On dev most of the carry return is funding income, and the funding leg on its own is "
        "very smooth. On the gate the price leg did more of the work. A spot-only version of the "
        "trade, like the twsq one in section 8, earns only the price leg."
    )

    # Appendix B
    pdf.h1("Appendix B: Trial registry")
    pdf.body(
        "Every configuration tested, in the order it was run, with its dev "
        f"({DEV_START} to {DEV_END}) and gate ({GATE_START} to {GATE_END}) Sharpe. Rows 1 to "
        f"{N_TRIALS} are the research trials used for the deflated Sharpe; rows {N_TRIALS + 1} and "
        f"{N_TRIALS + 2} are the two combination rules for the book, and rows {N_TRIALS + 3} to "
        f"{N_TRIALS + 5} are v2, logged after it was registered (section 7)."
    )
    rows = [["#", "Family", "Config", "Dev", "Gate"]]
    for i, r in reg.iterrows():
        dev, gate = float(r["dev_sharpe"]), float(r["gate_sharpe"])
        rows.append([str(i + 1), FAMILY_LABEL[r["family"]], describe(r["family"], json.loads(r["config"])),
                     sr(dev) if np.isfinite(dev) else "n/a", sr(gate) if np.isfinite(gate) else "n/a"])
    pdf.table_block(rows, widths=(8, 22, 70, 12, 12), align=("RIGHT", "LEFT", "LEFT", "RIGHT", "RIGHT"))

    out.parent.mkdir(exist_ok=True)
    pdf.output(str(out))
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {pdf.page} pages)")
    return pdf.page


def main():
    """Build the PDF, or with --check only check: the report's and READMEs' numbers, and
    that reports/REPORT.pdf is what the current code and data produce, byte for byte."""
    check_only = "--check" in sys.argv[1:]
    d = load_inputs()
    print("Checking report numbers against the notebooks...")
    v = compute(d)
    docs = doc_failures(v, d)
    if docs:
        raise AssertionError("README numbers disagree with the results; PDF not written.\n" + "\n".join(docs))
    print("  all checks passed, README.md and alphas/README.md included")
    with tempfile.TemporaryDirectory() as td:
        figs = make_figures(d, Path(td))
        if not check_only:
            build_pdf(d, v, figs)
            return
        fresh = Path(td) / "REPORT.pdf"
        build_pdf(d, v, figs, out=fresh)
        if fresh.read_bytes() != OUT.read_bytes():
            raise AssertionError("reports/REPORT.pdf is out of date; run reports/build_report.py")
        print("  reports/REPORT.pdf matches the code and data")


if __name__ == "__main__":
    main()
