"""Build the PDF research report, ``reports/REPORT.pdf``.

Every number in the report is computed here from the artifacts in
data/processed/ (written by notebooks 00-07), from the twsq CSVs in alphas/,
or, for the fast-reversal grid, read from notebook 05's executed output. The
script checks the headline numbers against the values the notebooks printed
and stops without writing the PDF if anything disagrees. Run it after the
notebooks:

    .venv/bin/python reports/build_report.py

The PDF is written with fpdf2 (pure Python).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

DEV_START, DEV_END = "2020-01-01", "2024-07-31"
GATE_START, GATE_END = "2024-08-01", "2025-06-30"
LOCKBOX_START, LOCKBOX_END = "2025-07-01", "2026-07-06"
WINDOWS = {
    "Dev": slice(DEV_START, DEV_END),
    "Gate": slice(GATE_START, GATE_END),
    "Lockbox": slice(LOCKBOX_START, LOCKBOX_END),
    "Full": slice(DEV_START, LOCKBOX_END),
}
N_TRIALS = 48          # research trials in the registry (rows 1-48); rows 49-50 are the two books
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


def sr(x: float) -> str:
    return f"{x:+.2f}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_inputs() -> dict:
    need = ["combined_portfolio", "benchmarks", "sleeve_baseline_momentum", "sleeve_baseline_reversal",
            "sleeve_seasonality", "sleeve_orderflow", "sleeve_carry", "sleeves_full",
            "posthoc_book", "price", "returns", "universe", "taker_imbalance", "funding",
            "price_1h"]
    missing = [n for n in need if not (PROC / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {PROC}. Run notebooks 00-07 first.")
    d = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in need}
    d["registry"] = pd.read_csv(PROC / "trial_registry.csv")
    d["cost_stress"] = pd.read_csv(PROC / "posthoc_cost_stress.csv")
    d["carry_split"] = pd.read_csv(PROC / "posthoc_carry_split.csv").set_index("window")
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

def perf(r: pd.Series, bench: pd.DataFrame) -> dict:
    r = r.dropna()
    ab = metrics.alpha_beta(r, bench["BTC"], bench["MKT"], names=["BTC", "MKT"])
    return {"ret": metrics.ann_return(r), "vol": metrics.ann_vol(r), "sharpe": metrics.sharpe(r),
            "mdd": metrics.max_drawdown(r), "beta": ab["beta_BTC"], "alpha": ab["alpha_ann"],
            "alpha_t": ab["alpha_tstat"], "n": len(r)}


def check(label: str, got: float, expected: float, tol: float, failures: list) -> None:
    if not abs(got - expected) <= tol:
        failures.append(f"  {label}: got {got:.4f}, expected {expected:.4f} (tol {tol})")


def compute(d: dict) -> dict:
    bench = d["benchmarks"]
    cp = d["combined_portfolio"]
    wf, ew = cp["walk_forward"], cp["equal_weight"]
    sleeves = d["sleeves_full"]
    ph = d["posthoc_book"]
    failures: list[str] = []

    # sleeves_full (notebook 07) must be the book notebook 06 built: the same
    # research parquets up to the gate, and the same equal-weight average.
    for name in ["seasonality", "orderflow", "carry"]:
        sealed = d[f"sleeve_{name}"][name].loc[DEV_START:GATE_END]
        both = pd.concat([sleeves[name].loc[DEV_START:GATE_END], sealed], axis=1).dropna()
        if not np.array_equal(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()):
            failures.append(f"  sleeves_full[{name}] differs from sleeve_{name}.parquet")
    ew_check = pd.concat([sleeves.dropna().mean(axis=1), ew], axis=1, sort=True).dropna()
    if not np.array_equal(ew_check.iloc[:, 0].to_numpy(), ew_check.iloc[:, 1].to_numpy()):
        failures.append("  sleeves_full does not reproduce the equal-weight book")

    reg = d["registry"]
    if len(reg) != N_TRIALS + 2 or (reg["family"] != "portfolio").sum() != N_TRIALS:
        failures.append(f"  registry has {len(reg)} rows; expected {N_TRIALS} trials + 2 books")

    v: dict = {}
    v["book"] = {(b, w): perf(s.loc[sl], bench) for b, s in [("WF", wf), ("EW", ew)]
                 for w, sl in WINDOWS.items()}
    v["sleeve"] = {(n, w): perf(sleeves[n].loc[sl], bench) for n in sleeves.columns
                   for w, sl in WINDOWS.items() if w != "Full"}
    lk, full = wf.loc[WINDOWS["Lockbox"]].dropna(), wf.loc[WINDOWS["Full"]].dropna()
    v["lk_n"] = len(lk)
    v["dsr_full"] = metrics.deflated_sharpe(full, n_trials=N_TRIALS)
    v["dsr_lk"] = metrics.deflated_sharpe(lk, n_trials=N_TRIALS)
    v["ci"] = metrics.bootstrap_sharpe_ci(lk, **BOOT)
    v["corr"] = sleeves.loc[WINDOWS["Dev"]].corr()

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

    # Numbers as printed by notebooks 06 and 07, and signs the text relies on.
    lk_wf, lk_ew = v["book"][("WF", "Lockbox")], v["book"][("EW", "Lockbox")]
    for label, got, exp, tol in [
        ("lockbox WF Sharpe", lk_wf["sharpe"], 1.458, 0.0005),
        ("lockbox EW Sharpe", lk_ew["sharpe"], 1.233, 0.0005),
        ("lockbox WF beta to BTC", lk_wf["beta"], -0.020, 0.0005),
        ("lockbox WF alpha", lk_wf["alpha"], 0.0948, 0.0001),
        ("lockbox WF alpha t", lk_wf["alpha_t"], 1.253, 0.0005),
        ("deflated Sharpe, full", v["dsr_full"], 0.4901, 0.00005),
        ("deflated Sharpe, lockbox", v["dsr_lk"], 0.2049, 0.00005),
        ("bootstrap CI low", v["ci"][0], -0.423, 0.0005),
        ("bootstrap CI high", v["ci"][1], 3.444, 0.0005),
        ("dev WF Sharpe", v["book"][("WF", "Dev")]["sharpe"], 0.397, 0.0005),
        ("dev EW Sharpe", v["book"][("EW", "Dev")]["sharpe"], 1.681, 0.0005),
        ("full WF Sharpe", v["book"][("WF", "Full")]["sharpe"], 1.020, 0.0005),
        ("full EW Sharpe", v["book"][("EW", "Full")]["sharpe"], 1.763, 0.0005),
        ("lockbox seasonality", v["sleeve"][("seasonality", "Lockbox")]["sharpe"], -0.565, 0.0005),
        ("lockbox orderflow", v["sleeve"][("orderflow", "Lockbox")]["sharpe"], 1.949, 0.0005),
        ("lockbox carry", v["sleeve"][("carry", "Lockbox")]["sharpe"], 0.881, 0.0005),
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
    ]:
        check(label, got, exp, tol, failures)
    if not (v["untilted_c"]["Dev"] > v["seas_c"]["Dev"] and v["seas_c"]["Gate"] > v["untilted_c"]["Gate"]):
        failures.append("  section 5.1 text assumes the charged overlay loses on dev and wins on the gate")
    if not (abs(tw.loc["SeasonalMomentum", "sharpe"]) < 0.3 and tw.loc["OrderflowFollow", "sharpe"] < 0
            and tw.loc["FundingCarry", "sharpe"] > 0):
        failures.append("  section 6 text no longer matches the twsq results")
    if failures:
        raise AssertionError("Report numbers disagree with the notebooks; PDF not written.\n"
                             + "\n".join(failures))
    return v


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

FAMILY_LABEL = {"incumbent_momentum": "baseline", "incumbent_pairs": "baseline",
                "seasonality": "seasonality", "orderflow": "orderflow", "carry": "carry",
                "fastrev": "fast reversal", "portfolio": "book"}


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


def perf_rows(stats: dict, keys: list, labels: list) -> list:
    rows = [["", "Ann. return", "Ann. vol", "Sharpe", "Max DD", "Beta BTC", "Alpha"]]
    for key, lab in zip(keys, labels):
        s = stats[key]
        rows.append([lab, pct(s["ret"]), pct(s["vol"]), sr(s["sharpe"]), pct(s["mdd"]),
                     f"{s['beta']:+.2f}", pct(s["alpha"])])
    return rows


PERF_WIDTHS = (34, 16, 14, 13, 14, 14, 13)


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


def build_pdf(d: dict, v: dict, figs: dict) -> int:
    reg = d["registry"]
    book, bookc, sl, base = v["book"], v["book_c"], v["sleeve"], v["base"]
    lk_wf, lk_ew = book[("WF", "Lockbox")], book[("EW", "Lockbox")]
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
    lo, hi = v["ci"]
    lo_c, hi_c = v["ci_c"]
    uni_size = int(uni.loc[DEV_START:].sum(axis=1).median())
    n_hourly = price_1h.shape[1]
    of_cut = 1 - stress.loc[("orderflow", 20), "dev"] / stress.loc[("orderflow", 7), "dev"]

    pdf = Report(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 15, 18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 19)
    pdf.multi_cell(0, 9, "Statistical Arbitrage in Cryptocurrencies", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90)
    pdf.multi_cell(0, 6, "Momentum, reversal and funding carry "
                         "on Binance data, 2020 to mid-2026", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)

    pdf.h1("Summary")
    pdf.body(
        "I tested momentum and reversal ideas on daily and hourly Binance data, charged every "
        "backtest realistic costs (20 bps for market orders, 7 bps for limit orders) and "
        "combined the strategies that passed into one book. Parameters were chosen on a "
        "development window (2020 to mid-2024), checked on a separate gate window (August 2024 "
        "to June 2025) and then run once on a lockbox year (July 2025 to July 2026) that no "
        f"decision had looked at. Every configuration tried is logged, {N_TRIALS} in all, and "
        "the deflated Sharpe ratio charges the book for that search.\n\n"
        "The baseline time-series momentum sleeve worked well on the development window "
        f"(Sharpe {sr(base[('momentum', 'Dev')]['sharpe'])}) and much less well afterwards "
        f"({sr(base[('momentum', 'Gate')]['sharpe'])}). Pairs trading and plain short-horizon "
        "reversal lost money after costs. Three sleeves made the book: momentum with weekday "
        "exposure halved (Seasonality), a 10-day taker-imbalance follower (Orderflow) and a "
        "funding carry trade (Carry). On the lockbox the walk-forward book returned "
        f"{pct(lk_wf['ret'])} a year at {pct(lk_wf['vol'])} volatility, a Sharpe of "
        f"{lk_wf['sharpe']:.2f}, with a beta to BTC of {lk_wf['beta']:+.2f}; the equal-weight "
        f"book had a Sharpe of {lk_ew['sharpe']:.2f}. The deflated Sharpe on the lockbox is "
        f"{v['dsr_lk']:.2f}, well short of the usual 0.95 bar, so the result is positive but "
        "not statistically proven.\n\n"
        "An audit after the lockbox found that the Seasonality backtest never paid for the "
        "trades that resize its book twice a week. Charged properly, that sleeve would not have "
        "passed selection. Re-scoring the same frozen book with those trades charged moves the "
        f"lockbox Sharpe to {bookc[('WF', 'Lockbox')]['sharpe']:.2f} (walk-forward) and "
        f"{bookc[('EW', 'Lockbox')]['sharpe']:.2f} (equal weight). Section 5 has the details; "
        "the conclusion does not change."
    )

    # 1. Data and method
    pdf.h1("1. Data and method")
    pdf.bullets([
        f"Daily klines for the top 150 Binance USDT pairs by volume, 2018-01 to 2026-07 "
        f"({price.shape[1]} names returned data). Close, USD volume and taker-buy volume are kept.",
        f"Hourly klines for the {n_hourly} names with the highest 30-day volume at fetch time, "
        "2020-01 to 2026-07. The hourly research starts in 2020-06.",
        f"Perpetual funding rates for the same {n_hourly} names where a perp exists "
        f"({funding.shape[1]} names), summed per day, from {funding.index.min():%Y-%m}.",
        "The data was fetched on 2026-07-06, so that last day is a partial day in every panel.",
    ])
    pdf.body(
        "The tradable universe is point-in-time: on each day, the 100 coins with the highest "
        "30-day median dollar volume, using data up to the day before, with a $1M floor and at "
        "least 30 days of history. Stablecoins, wrapped coins and leveraged tokens are excluded. "
        f"The median universe since 2020 has {uni_size} names.\n\n"
        "The backtest applies weights set at the close of day t to the return of day t+1, so no "
        "position uses information it could not have had. Costs are charged on turnover every "
        "day: 20 bps per dollar traded for market orders (7 bps commission plus 13 bps "
        "slippage) and 7 bps for limit orders. The momentum sleeves take liquidity and pay "
        "20 bps; orderflow, carry and pairs rebalance passively and pay 7 bps. Backtests are "
        "unconstrained, and Sharpe ratios are annualised with 365 days."
    )
    pdf.table_block([
        ["Window", "Dates", "Used for"],
        ["Development", f"{DEV_START} to {DEV_END}", "choosing signals and parameters"],
        ["Gate", f"{GATE_START} to {GATE_END}", "a hold-out check before anything is kept"],
        ["Lockbox", f"{LOCKBOX_START} to {LOCKBOX_END}", "one final run of the frozen book"],
    ], title="Validation windows", widths=(22, 38, 60), align=("LEFT", "LEFT", "LEFT"))
    pdf.caption("The lockbox was read once, in notebook 06, after every selection decision had "
                "been made. Section 5 re-scores the same book after an audit and says so.")

    # 2. Baselines
    pdf.h1("2. Baselines: momentum and pairs")
    pdf.body(
        "Two standard strategies serve as baselines. Time-series momentum "
        "holds each coin long or short by the sign of its 30-day return (skipping the latest "
        "day) and scales the book to 15% volatility. The pairs sleeve re-selects up to 20 "
        "correlated pairs every 63 days, opens a spread when its z-score passes 2 and holds it "
        "until the z-score passes 2 on the other side."
    )
    pdf.table_block(perf_rows(base, [("momentum", "Dev"), ("momentum", "Gate"),
                                     ("reversal", "Dev"), ("reversal", "Gate")],
                              ["Momentum, dev", "Momentum, gate", "Pairs, dev", "Pairs, gate"]),
                    title="Baseline sleeves", widths=PERF_WIDTHS)
    pdf.body(
        f"Momentum's Sharpe falls from {sr(base[('momentum', 'Dev')]['sharpe'])} on dev to "
        f"{sr(base[('momentum', 'Gate')]['sharpe'])} on the gate. Pairs loses money on dev "
        f"({sr(base[('reversal', 'Dev')]['sharpe'])}), so it is dropped even though the gate "
        f"was positive ({sr(base[('reversal', 'Gate')]['sharpe'])}). Momentum stays as the "
        "starting point for the seasonality work."
    )

    # 3. Research families
    pdf.h1("3. Research families")
    pdf.body(
        "Each family starts from a simple trading idea and was tested on dev and gate "
        "only. A config survives if it is positive on both windows and passes its family's "
        "extra rule: beat the untilted sleeve (seasonality), beat the unfiltered baseline "
        "(conditioned reversals), or have a neighbouring parameter that is also positive. "
        "The appendix lists all 50 registry rows."
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
        "that split lost money on the gate."
    )
    pdf.table_block(family_rows(reg, "seasonality"), title="Seasonality configs", widths=(80, 20, 20))
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
    pdf.table_block(fr_rows, title="Fast reversal on the dev window (hourly, annualised with 8,760 hours)",
                    widths=(34, 16, 15, 14, 12, 14, 15))

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
                    title="Book performance by window (alpha and beta from a regression on BTC and "
                          "the equal-weight market)", widths=PERF_WIDTHS)
    pdf.caption(
        "The walk-forward book needs 756 days of history before its first weights, so it starts "
        "in 2021-10 and its dev figure covers only 2021-10 to 2024-07. Equal weight beats "
        "walk-forward on every window except the lockbox: with three nearly uncorrelated sleeves "
        "there is little for mean-variance weights to add."
    )
    names = ["seasonality", "orderflow", "carry"]
    pdf.table_block(perf_rows(sl, [(n, w) for n in names for w in ["Dev", "Gate", "Lockbox"]],
                              [f"{n.capitalize()}, {w.lower()}" for n in names
                               for w in ["Dev", "Gate", "Lockbox"]]),
                    title="Sleeve performance by window", widths=PERF_WIDTHS)
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
        f"sample and {v['dsr_lk']:.2f} on the lockbox, both far below 0.95. One year is also "
        f"short: the bootstrap interval above is {hi - lo:.1f} Sharpe points wide. The book did "
        "make money with almost no market exposure, but this sample cannot separate that from "
        "luck."
    )

    # 5. Post-hoc checks
    pdf.h1("5. Checks added after the lockbox")
    pdf.body(
        "Notebook 07 was added in September 2026, after the lockbox had been opened. It changes "
        "nothing in the book and logs no trials; it measures three things the audit raised."
    )
    pdf.h2("5.1 The cost of resizing the Seasonality sleeve")
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
        f"{book[('WF', 'Full')]['sharpe']:.2f} to {bookc[('WF', 'Full')]['sharpe']:.2f}."
    )
    pdf.h2("5.2 Higher costs for Orderflow and Carry")
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
        "much of its P&L from funding, so it is less exposed to costs."
    )
    pdf.h2("5.3 Where the carry P&L comes from")
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
        "trade, like the twsq one in section 6, earns only the price leg."
    )

    # 6. twsq
    pdf.h1("6. The sleeves in twsq")
    pdf.body(
        "alphas/ holds the three sleeves written as alphas for twsq, an execution "
        "framework. twsq backtests them on Binance daily bars over 700 days (2024-08-06 to "
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
        "fits the charged Seasonality numbers in section 5. OrderflowFollow loses a little, so "
        "the order-flow effect in the research seems to come from the wider, less liquid part of "
        "the universe, which also limits how much money it could take. FundingCarry's price leg "
        "is slightly positive, without the funding income that the research sleeve earns."
    )

    # 7. Limitations
    pdf.h1("7. Limitations")
    pdf.bullets([
        "Survivorship. The universe is built from the pairs Binance listed on the fetch date, so "
        "coins delisted earlier are missing. The hourly and funding panels are worse: their 60 "
        "names were picked by volume at fetch time, which favours coins that did well. The "
        "point-in-time universe limits which names can trade on each day but cannot add back the "
        "missing ones.",
        "One exchange. Prices, volumes, taker flow and funding all come from Binance.",
        "Perp versus spot. The carry backtest adds funding payments to spot returns and ignores "
        "the basis between perp and spot prices.",
        "Limit-order fills. Orderflow and Carry assume every limit order fills at 7 bps. "
        "Section 5.2 shows what higher costs would do.",
        "Pairs baseline. The pairs engine does not charge for opening a position that a newly "
        "selected pair already has on, or for closing pairs dropped at a rebalance. Charging them "
        "would only lower a baseline that already loses money.",
        "Cost of resizing. Scaling returns by a volatility target or a calendar multiplier does "
        "not charge for the trades the resizing needs. For the volatility target this is small "
        f"(it moves momentum's dev Sharpe by {v['untilted']['Dev'] - unt_c['Dev']:.2f}); for the "
        "weekday overlay it is not, as section 5.1 shows.",
        "Capacity. Nothing here was sized for real money. The twsq results suggest the order-flow "
        "edge sits in smaller names, where large trades would move prices.",
        "Sample size. The lockbox is one year, and the deflated Sharpe is below 0.95 on every "
        "window.",
    ])

    # Appendix
    pdf.h1("Appendix: trial registry")
    pdf.body(
        "Every configuration tested, in the order it was run, with its dev "
        f"({DEV_START} to {DEV_END}) and gate ({GATE_START} to {GATE_END}) Sharpe. Rows 1 to "
        f"{N_TRIALS} are the research trials used for the deflated Sharpe; rows {N_TRIALS + 1} and "
        f"{N_TRIALS + 2} are the two combination rules for the book."
    )
    rows = [["#", "Family", "Config", "Dev", "Gate"]]
    for i, r in reg.iterrows():
        dev, gate = float(r["dev_sharpe"]), float(r["gate_sharpe"])
        rows.append([str(i + 1), FAMILY_LABEL[r["family"]], describe(r["family"], json.loads(r["config"])),
                     sr(dev) if np.isfinite(dev) else "n/a", sr(gate) if np.isfinite(gate) else "n/a"])
    pdf.table_block(rows, widths=(8, 22, 70, 12, 12), align=("RIGHT", "LEFT", "LEFT", "RIGHT", "RIGHT"))

    OUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB, {pdf.page} pages)")
    return pdf.page


def main():
    d = load_inputs()
    print("Checking report numbers against the notebooks...")
    v = compute(d)
    print("  all checks passed")
    with tempfile.TemporaryDirectory() as td:
        figs = make_figures(d, Path(td))
        build_pdf(d, v, figs)


if __name__ == "__main__":
    main()
