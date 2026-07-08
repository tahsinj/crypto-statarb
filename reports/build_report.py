"""Build the PDF research report, ``reports/REPORT.pdf``.

Tables and most numbers are computed from the cached panels in
data/processed/ with quantlib.metrics. The fast-reversal table and a few
sentences quote notebook outputs directly. The Section 4 headline numbers are
checked against a recomputation and the script stops if they disagree. Run
it after the notebooks:

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
from quantlib import metrics, plotting, robustness  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "REPORT.pdf"

# Validation windows
DEV_START = "2020-01-01"
DEV_END = "2024-07-31"
GATE_START = "2024-08-01"
GATE_END = "2025-06-30"
LOCKBOX_START = "2025-07-01"
LOCKBOX_END = "2026-07-06"

DEV = slice(DEV_START, DEV_END)
GATE = slice(GATE_START, GATE_END)
LOCKBOX = slice(LOCKBOX_START, LOCKBOX_END)
FULL = slice(DEV_START, None)

# Trial count used for deflated Sharpe: all pre-portfolio trials enumerated in the registry.
# Rows 0-47 are the 48 alpha trials; rows 48-49 are portfolio combination entries.
N_ALPHA_TRIALS = 48

# Unicode substitutions so fpdf2 core Latin-1 font handles everything.
_SUBS = {
    "–": "-", "—": "-", "−": "-", "≈": "~", "×": "x",
    "²": "^2", "·": ".", "→": "->", "‘": "'", "’": "'",
    "“": '"', "”": '"', "•": "-", "σ": "sigma",
    "Σ": "Sum", "α": "alpha", "β": "beta",
    "≥": ">=", "≤": "<=", "Δ": "d",
}


def _ascii(text: str) -> str:
    for k, v in _SUBS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _twsq_sharpe(alpha_name: str) -> float:
    """Compute daily Sharpe from a committed twsq pos_pnl.csv (same formula as run_twsq._stats)."""
    path = ROOT / "alphas" / alpha_name / "backtest" / "pos_pnl.csv"
    pnl = pd.read_csv(path)["pnl"].dropna()
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(365)) if sd else float("nan")


def _survivor_sharpes(reg: pd.DataFrame, family: str, match: dict) -> tuple[float, float]:
    """Return (dev_sharpe, gate_sharpe) for the first row in reg where family matches
    and config JSON contains all key-value pairs in match."""
    for _, row in reg[reg["family"] == family].iterrows():
        cfg = json.loads(row["config"])
        if all(cfg.get(k) == v for k, v in match.items()):
            return float(row["dev_sharpe"]), float(row["gate_sharpe"])
    raise ValueError(f"Survivor not found: family={family!r}, match={match!r}")


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_inputs():
    need = ["combined_portfolio", "benchmarks", "sleeve_baseline_momentum",
            "sleeve_baseline_reversal", "sleeve_seasonality", "sleeve_orderflow",
            "sleeve_carry"]
    missing = [n for n in need if not (PROC / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing panels {missing} in {PROC}. Run notebooks 00-06 first."
        )
    cp = pd.read_parquet(PROC / "combined_portfolio.parquet")
    bm = pd.read_parquet(PROC / "benchmarks.parquet")
    mom_base = pd.read_parquet(PROC / "sleeve_baseline_momentum.parquet")["momentum"]
    rev_base = pd.read_parquet(PROC / "sleeve_baseline_reversal.parquet")["reversal"]
    seas = pd.read_parquet(PROC / "sleeve_seasonality.parquet")["seasonality"]
    of = pd.read_parquet(PROC / "sleeve_orderflow.parquet")["orderflow"]
    carry = pd.read_parquet(PROC / "sleeve_carry.parquet")["carry"]
    reg = pd.read_csv(PROC / "trial_registry.csv")
    return cp, bm, mom_base, rev_base, seas, of, carry, reg


# ---------------------------------------------------------------------------
# Assert Section 4 numbers against combined_portfolio.parquet
# ---------------------------------------------------------------------------

def assert_section4(cp: pd.DataFrame, btc: pd.Series, mkt: pd.Series) -> dict:
    """Recompute the Section 4 headline numbers and stop if they disagree with
    the values quoted in the text.

    Returns the computed values so the report body can use them directly.
    """
    wf = cp["walk_forward"]
    ew = cp["equal_weight"]

    lk_wf = wf.loc[LOCKBOX].dropna()
    lk_ew = ew.loc[LOCKBOX].dropna()
    full_wf = wf.loc[FULL].dropna()

    lk_wf_sr = metrics.sharpe(lk_wf)
    lk_ew_sr = metrics.sharpe(lk_ew)
    dsr_full = metrics.deflated_sharpe(full_wf, n_trials=N_ALPHA_TRIALS)
    dsr_lk = metrics.deflated_sharpe(lk_wf, n_trials=N_ALPHA_TRIALS)

    # Alpha / beta uses both BTC and equal-weight market as regressors.
    ab = metrics.alpha_beta(
        lk_wf,
        btc.reindex(lk_wf.index),
        mkt.reindex(lk_wf.index),
        names=["BTC", "MKT"],
    )

    TOL_SR = 0.005   # Sharpe tolerance (rounding)
    TOL_BETA = 0.002
    TOL_ALPHA = 0.002
    TOL_T = 0.02
    TOL_DSR = 0.005

    checks = [
        ("lockbox WF Sharpe", lk_wf_sr, 1.458, TOL_SR),
        ("lockbox EW Sharpe", lk_ew_sr, 1.233, TOL_SR),
        ("lockbox beta_BTC", ab["beta_BTC"], -0.020, TOL_BETA),
        ("lockbox alpha_ann", ab["alpha_ann"], 0.0948, TOL_ALPHA),
        ("lockbox alpha_tstat", ab["alpha_tstat"], 1.25, TOL_T),
        ("deflated_sharpe_full", dsr_full, 0.490, TOL_DSR),
        ("deflated_sharpe_lockbox", dsr_lk, 0.205, TOL_DSR),
    ]
    failures = []
    for name, got, expected, tol in checks:
        if abs(got - expected) > tol:
            failures.append(f"  {name}: got {got:.4f}, expected {expected:.4f} (tol {tol})")
    if failures:
        raise AssertionError(
            "Section 4 number mismatch, PDF not written.\n" + "\n".join(failures)
        )

    return {
        "lk_wf_sr": lk_wf_sr,
        "lk_ew_sr": lk_ew_sr,
        "beta_BTC": ab["beta_BTC"],
        "beta_MKT": ab["beta_MKT"],
        "alpha_ann": ab["alpha_ann"],
        "alpha_tstat": ab["alpha_tstat"],
        "r_squared": ab["r_squared"],
        "dsr_full": dsr_full,
        "dsr_lk": dsr_lk,
        "full_wf_sr": metrics.sharpe(full_wf),
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def validation_protocol_table():
    return [
        ["Window", "Dates", "Purpose", "Opened"],
        ["Development", f"{DEV_START} to {DEV_END}", "Parameter search, model selection", "Yes"],
        ["Regime gate", f"{GATE_START} to {GATE_END}", "Hold-out during development", "Yes"],
        ["Lockbox", f"{LOCKBOX_START} to {LOCKBOX_END}", "Final evaluation", "Once"],
    ]


def incumbent_decay_table(mom: pd.Series, rev: pd.Series) -> list:
    rows = [["Sleeve", "Dev Sharpe", "Gate Sharpe", "Dev n", "Gate n"]]
    for name, s in [("Momentum (baseline)", mom), ("Pairs (baseline)", rev)]:
        d = s.loc[DEV].dropna()
        g = s.loc[GATE].dropna()
        rows.append([name,
                     f"{metrics.sharpe(d):+.2f}", f"{metrics.sharpe(g):+.2f}",
                     str(len(d)), str(len(g))])
    return rows


def family_table(reg: pd.DataFrame, family: str) -> list:
    sub = reg[reg["family"] == family].copy()
    sub["dev_sharpe"] = sub["dev_sharpe"].astype(float)
    sub["gate_sharpe"] = sub["gate_sharpe"].astype(float)
    if sub.empty:
        return [["config", "dev", "gate"]]
    rows = [["Config", "Dev Sharpe", "Gate Sharpe", "Note"]]
    for _, r in sub.iterrows():
        rows.append([
            str(r["config"])[:60],
            f"{r['dev_sharpe']:+.2f}",
            f"{r['gate_sharpe']:+.2f}",
            str(r.get("note", ""))[:40],
        ])
    return rows


def fastrev_highlight_table() -> list:
    """Gross vs cost table for the 4-hour and 4-hour/6-rebal configs (from nb 05)."""
    # Computed during notebook 05 execution; gross returns are not separately saved.
    return [
        ["Config (lookback, rebal)", "Gross SR (hrly)", "Gross Ann.", "Cost drag Ann.", "Ratio", "Pass 3x?"],
        ["4h, 1h",  "+4.83", "+272%", "+461%", "0.59", "No"],
        ["4h, 6h",  "+0.86",  "+44%", "+146%", "0.30", "No"],
        ["12h, 1h", "+1.80", "+103%", "+267%", "0.39", "No"],
        ["24h, 1h", "+0.63",  "+37%", "+187%", "0.20", "No"],
        ["48h, 1h", "+0.01",   "+1%", "+130%", "0.01", "No"],
    ]


def sleeve_corr_table(seas: pd.Series, of: pd.Series, carry: pd.Series) -> list:
    X = pd.concat(
        [seas.rename("Seasonality"), of.rename("Orderflow"), carry.rename("Carry")],
        axis=1,
    ).loc[DEV_START:DEV_END].dropna()
    c = X.corr().round(3)
    rows = [[""] + list(c.columns)]
    for idx in c.index:
        rows.append([idx] + [f"{v:.3f}" for v in c.loc[idx]])
    return rows


def windows_table(cp: pd.DataFrame) -> list:
    wf = cp["walk_forward"]
    ew = cp["equal_weight"]
    rows = [["Window", "WF Sharpe", "WF Ann.Ret", "WF MaxDD", "EW Sharpe", "EW Ann.Ret", "EW MaxDD"]]
    for label, slc in [
        ("Development (2020-2024-07)", DEV),
        ("Gate (2024-08 to 2025-06)", GATE),
        ("Lockbox (2025-07 to 2026-07)", LOCKBOX),
    ]:
        w = wf.loc[slc].dropna()
        e = ew.loc[slc].dropna()
        rows.append([
            label,
            f"{metrics.sharpe(w):+.2f}",
            f"{metrics.ann_return(w):.1%}",
            f"{metrics.max_drawdown(w):.1%}",
            f"{metrics.sharpe(e):+.2f}",
            f"{metrics.ann_return(e):.1%}",
            f"{metrics.max_drawdown(e):.1%}",
        ])
    return rows


def lockbox_alpha_table(vals: dict) -> list:
    return [
        ["Quantity (HAC, BTC + market regressors)", "Lockbox only"],
        ["Annualised alpha", f"{vals['alpha_ann']:.2%}"],
        ["Alpha t-stat (Newey-West)", f"{vals['alpha_tstat']:.2f}"],
        ["Beta to BTC", f"{vals['beta_BTC']:+.3f}"],
        ["Beta to equal-weight market", f"{vals['beta_MKT']:+.3f}"],
        ["R-squared", f"{vals['r_squared']:.3f}"],
    ]


def significance_table(cp: pd.DataFrame, vals: dict, boot_ci: tuple[float, float]) -> list:
    lo, hi = boot_ci
    return [
        ["Quantity", "Full sample", "Lockbox only"],
        ["Sharpe", f"{vals['full_wf_sr']:.2f}", f"{vals['lk_wf_sr']:.2f}"],
        ["Block-bootstrap 95% CI (lockbox)", "n/a", f"[{lo:.2f}, {hi:.2f}]"],
        [f"Deflated Sharpe ({N_ALPHA_TRIALS} trials)", f"{vals['dsr_full']:.3f}", f"{vals['dsr_lk']:.3f}"],
    ]


def registry_table(reg: pd.DataFrame) -> list:
    reg2 = reg.copy()
    reg2["dev_sharpe"] = reg2["dev_sharpe"].astype(float)
    reg2["gate_sharpe"] = reg2["gate_sharpe"].astype(float)
    rows = [["#", "Family", "Config (truncated)", "Dev SR", "Gate SR", "Note"]]
    for i, r in reg2.iterrows():
        rows.append([
            str(i + 1),
            str(r["family"]),
            str(r["config"])[:45],
            f"{r['dev_sharpe']:+.2f}",
            f"{r['gate_sharpe']:+.2f}",
            str(r.get("note", ""))[:30],
        ])
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(cp: pd.DataFrame, btc: pd.Series,
                 seas: pd.Series, of: pd.Series, carry: pd.Series,
                 tmp: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    # Figure 1: Equity curves (wf vs ew vs BTC, log scale)
    wf = cp["walk_forward"].loc[DEV_START:].dropna()
    ew = cp["equal_weight"].loc[DEV_START:].dropna()
    btc_s = btc.loc[DEV_START:].dropna()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    ax = axes[0]
    for ser, label, color in [
        (wf, f"Walk-forward (SR={metrics.sharpe(wf):.2f})", "steelblue"),
        (ew, f"Equal-weight (SR={metrics.sharpe(ew):.2f})", "darkorange"),
        (btc_s, f"BTC (SR={metrics.sharpe(btc_s):.2f})", "gray"),
    ]:
        eq = (1.0 + ser.fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=label,
                color=color, linewidth=1.2 if "BTC" not in label else 0.9,
                linestyle="--" if "BTC" in label else "-")
    ax.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0, label="Gate start")
    ax.axvline(pd.Timestamp(LOCKBOX_START), color="red", ls="--", lw=1.2, label="Lockbox open")
    ax.set_yscale("log")
    ax.set_title("Equity curves (log scale), 2020 to 2026-07")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("growth of $1")

    ax2 = axes[1]
    dd = metrics.drawdown_curve(wf)
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.35)
    ax2.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0)
    ax2.axvline(pd.Timestamp(LOCKBOX_START), color="red", ls="--", lw=1.2)
    ax2.set_title("Walk-forward drawdown")
    ax2.set_ylabel("drawdown")

    ax3 = axes[2]
    roll = wf.rolling(180).mean() / wf.rolling(180).std() * np.sqrt(metrics.TRADING_DAYS)
    ax3.plot(roll.index, roll.values, color="steelblue", lw=0.9)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0)
    ax3.axvline(pd.Timestamp(LOCKBOX_START), color="red", ls="--", lw=1.2)
    ax3.set_title("Rolling 180d Sharpe (walk-forward)")

    fig.tight_layout()
    p = tmp / "equity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["equity"] = p

    # Figure 2: Walk-forward weight path (regenerated from the 3 sleeves)
    X_book = pd.concat(
        [seas.rename("Seasonality"), of.rename("Orderflow"), carry.rename("Carry")],
        axis=1,
    ).loc[DEV_START:GATE_END].dropna()
    _, weight_path = robustness.walk_forward_weights(
        X_book, train_days=756, step_days=63, min_train=252
    )

    fig2, ax4 = plt.subplots(figsize=(10, 3.5))
    colors2 = {"Seasonality": "goldenrod", "Orderflow": "steelblue", "Carry": "forestgreen"}
    for col in weight_path.columns:
        ax4.plot(weight_path.index, weight_path[col].values,
                 label=col, color=colors2.get(col, None), linewidth=1.0)
    ax4.axhline(0, color="k", lw=0.6)
    ax4.axvline(pd.Timestamp(GATE_START), color="navy", ls=":", lw=1.0, label="Gate start")
    ax4.set_title("Walk-forward sleeve weights (max-Sharpe, fit on trailing 756 days)")
    ax4.set_ylabel("weight")
    ax4.legend(fontsize=8)
    fig2.tight_layout()
    p2 = tmp / "weights.png"
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    paths["weights"] = p2

    return paths


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, _ascii("Statistical Arbitrage in Cryptocurrencies: research report"),
                  align="R", new_x="LMARGIN", new_y="NEXT")
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
        self.multi_cell(0, 7, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 6, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(0)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def caption(self, text: str):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90)
        self.multi_cell(0, 4, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(1)

    def table_block(self, rows: list, title: str | None = None):
        if title:
            self.set_font("Helvetica", "B", 10)
            self.multi_cell(0, 5, _ascii(title), new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        with self.table(first_row_as_headings=True, line_height=5.0,
                        text_align="LEFT", padding=1) as table:
            for row in rows:
                tr = table.row()
                for cell in row:
                    tr.cell(_ascii(str(cell)))
        self.ln(2)

    def figure(self, path: Path, w: int = 170):
        self.image(str(path), w=w)
        self.ln(2)


def build_pdf(cp, btc, mkt, mom_base, rev_base, seas, of, carry, reg, vals, figs):
    # ---------------------------------------------------------------------------
    # Derive all prose values from committed artifacts
    # ---------------------------------------------------------------------------
    # Finding 1: twsq Sharpes from committed pos_pnl.csv files
    twsq_seas_sr = _twsq_sharpe("SeasonalMomentum")
    twsq_of_sr = _twsq_sharpe("OrderflowFollow")
    twsq_carry_sr = _twsq_sharpe("FundingCarry")

    # Finding 2: survivor config Sharpes from trial registry
    seas_dev_sr, seas_gate_sr = _survivor_sharpes(
        reg, "seasonality", {"weekday": 0.5, "weekend": 1.0}
    )
    of_dev_sr, of_gate_sr = _survivor_sharpes(
        reg, "orderflow", {"dir": "follow", "probe": "P3", "smooth": 10}
    )
    carry_dev_sr, carry_gate_sr = _survivor_sharpes(
        reg, "carry", {"probe": "P1", "smooth": 7}
    )

    # Finding 3: baseline Sharpes, same computation as incumbent_decay_table
    mom_dev_sr = metrics.sharpe(mom_base.loc[DEV].dropna())
    mom_gate_sr = metrics.sharpe(mom_base.loc[GATE].dropna())
    rev_dev_sr = metrics.sharpe(rev_base.loc[DEV].dropna())
    rev_gate_sr = metrics.sharpe(rev_base.loc[GATE].dropna())

    # Finding 4: bootstrap CI, computed once and used in both table and text
    lk_wf = cp["walk_forward"].loc[LOCKBOX].dropna()
    boot_lo, boot_hi = metrics.bootstrap_sharpe_ci(lk_wf, n_boot=3000, method="block", seed=42)
    _ci_txt = (
        f"[{boot_lo:.2f}, {boot_hi:.2f}], which includes zero"
        if boot_lo < 0 < boot_hi
        else f"[{boot_lo:.2f}, {boot_hi:.2f}], which excludes zero"
    )

    # ---------------------------------------------------------------------------
    pdf = Report(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(18, 15, 18)
    pdf.add_page()

    # Title block
    pdf.set_font("Helvetica", "B", 19)
    pdf.multi_cell(0, 9, _ascii("Statistical Arbitrage in Cryptocurrencies"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90)
    pdf.multi_cell(0, 6,
                   _ascii("Momentum, reversal and funding carry "
                           "in a low-beta book built on Binance data."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(3)

    # Abstract
    pdf.h1("Abstract")
    pdf.body(
        "We test momentum and reversal strategies on live Binance daily price-volume data, "
        "backtest them unconstrained and net of realistic costs (20 bps market / 7 bps limit), "
        "and combine the survivors into a low-beta book. "
        "The baseline time-series momentum sleeve has a clear edge in the development window "
        "that weakened after 2024-08. Three new sleeves make up the book: calendar-scaled "
        "momentum (Seasonality), a taker-imbalance follower (Orderflow) and a funding-carry "
        "short (Carry). Walk-forward weighting gives a lockbox Sharpe of +1.458 and equal "
        "weighting +1.233. The book's beta to BTC is close to zero. "
        "Annualised alpha is 9.48% (HAC t = 1.25). "
        "The deflated Sharpe is 0.490 on the full sample and 0.205 on the lockbox against 48 "
        "enumerated trials: positive, but not statistically decisive."
    )

    # Section 1: Project and data
    pdf.h1("1. Project and data")
    pdf.body(
        "Data is fetched live from Binance public APIs and cached locally:\n"
        "- Daily klines, top-150 USDT pairs, 2018-01 to 2026-07 (149 names with data).\n"
        "- Hourly klines, top-60 pairs by trailing ADV, used from 2020-06.\n"
        "- Perpetual funding rates from Binance fapi, 2019-09 onward.\n"
        "- Taker-buy volume from the daily klines.\n\n"
        "The tradable universe is point-in-time: a coin qualifies on date t if its "
        "trailing 30-day median dollar-volume clears the bar, computed on t-1 data only. "
        "Stablecoins are excluded. Evaluation starts 2020-01-01; data before then is "
        "too sparse for stable cross-sectional signals.\n\n"
        "Cost model: 20 bps per unit of turnover for market orders (momentum sleeves, "
        "7 bps commission + 13 bps slippage) and 7 bps for limit orders "
        "(pairs, carry and orderflow). Costs are charged on turnover each day."
    )
    pdf.table_block(validation_protocol_table(),
                    title="Validation protocol: three non-overlapping windows")
    pdf.caption(
        "The lockbox window was opened once, in notebook 06, after every selection "
        "decision had been made."
    )

    # Section 2: The decay chapter
    pdf.h1("2. Baselines")
    pdf.body(
        "The two baselines are a time-series momentum sleeve (30-day lookback, 15% vol "
        "target) and a pairs reversal sleeve (correlation and half-life selection, top 20 "
        "pairs, entry at 2 sigma). Run on the Binance "
        "panel through 2025-06-30:"
    )
    pdf.table_block(incumbent_decay_table(mom_base, rev_base),
                    title="Baseline sleeves: dev (2020-01 to 2024-07) vs gate (2024-08 to 2025-06)")
    pdf.body(
        f"Momentum is strong on dev ({mom_dev_sr:+.2f}) and much weaker on the gate ({mom_gate_sr:+.2f}). "
        f"Pairs is negative on dev ({rev_dev_sr:+.2f}) and positive on the gate ({rev_gate_sr:+.2f}), "
        "which is not enough to keep it.\n\n"
        "twsq backtests of the three final sleeves (Binance daily bars, 2024-08 to 2026-07, 701 days): "
        f"SeasonalMomentum Sharpe {twsq_seas_sr:+.2f}, OrderflowFollow {twsq_of_sr:+.2f}, FundingCarry {twsq_carry_sr:+.2f}. "
        "The carry figure is low partly because the twsq backtest holds spot, so it earns "
        "none of the funding the research sleeve collects."
    )

    # Section 3: The alpha hunt
    pdf.h1("3. Research families")
    pdf.body(
        "Four families were tested on the dev and gate windows. A config survives only if "
        "it is positive on both windows and passes its family's extra rule (beating the "
        "baseline, or a positive neighbouring parameter); the notebooks state each rule. "
        "The registry in the appendix lists all 50 rows."
    )

    # 3.1 Seasonality
    pdf.h2("3.1 Seasonality (calendar tilts on momentum)")
    pdf.body(
        "Idea: institutions trade on weekdays and in US hours, retail at night and at "
        "weekends, so trend signals may behave differently by calendar bucket. We tested "
        "12 configs: weekday/weekend scaling of the momentum sleeve, long-flat market timing "
        "by bucket, an hourly off-hours test and a turn-of-month tilt.\n\n"
        f"Survivor: scaling the momentum sleeve to 0.5x on weekdays and 1.0x on weekends "
        f"(weekday=0.5, weekend=1.0) gives dev {seas_dev_sr:+.2f}, gate {seas_gate_sr:+.2f}. "
        "This is the Seasonality sleeve in the final book."
    )
    pdf.table_block(family_table(reg, "seasonality"), title="Seasonality trials")

    # 3.2 Orderflow
    pdf.h2("3.2 Orderflow (taker imbalance signal)")
    pdf.body(
        "Idea: sustained taker-buy pressure predicts short-term continuation. "
        "Signal: cross-sectional z-score of the 10-day mean of (taker share - 0.5). "
        "Reversal on quiet names and on flow-unconfirmed moves was tested as well.\n\n"
        f"Survivor: 10-day smoothing, follow direction (dev {of_dev_sr:+.2f}, gate {of_gate_sr:+.2f}). "
        "This is the Orderflow sleeve. The unsmoothed (1-day) version loses on both windows, "
        "and every reversal variant is negative on dev."
    )
    pdf.table_block(family_table(reg, "orderflow"), title="Orderflow trials")

    # 3.3 Funding carry
    pdf.h2("3.3 Funding carry (short high-funding perps)")
    pdf.body(
        "Idea: persistently high perp funding marks crowded longs, and the short side "
        "collects the funding. Signal: minus the z-score of 7-day mean funding across the "
        "names with a perp. Reversal in crowded names and the change in funding were tested "
        "as well.\n\n"
        f"Survivor: 7-day smoothing (dev {carry_dev_sr:+.2f}, gate {carry_gate_sr:+.2f}). "
        "This is the Carry sleeve. Its twsq version (spot only, so no funding income) shows "
        f"a Sharpe of {twsq_carry_sr:+.2f} over 2024-08 to 2026-07."
    )
    pdf.table_block(family_table(reg, "carry"), title="Funding carry trials")

    # 3.4 Fast reversal
    pdf.h2("3.4 Fast reversal (hourly cross-sectional, no survivor)")
    pdf.body(
        "Idea: daily reversal has a gross edge that costs remove. If the edge comes from "
        "liquidity provision it should be larger at intraday horizons. Rule set in advance: "
        "a config needs a gross return of at least 3x its cost drag before any tuning.\n\n"
        "The 4-hour lookback has the highest gross Sharpe (+4.83), so a faster reversal "
        "edge does exist. But its cost drag (461% a year) is about 1.7 times the gross "
        "return (272% a year), and no config passes the 3x screen."
    )
    pdf.table_block(fastrev_highlight_table(), title="Fast reversal: gross edge vs cost drag (notebook 05)")
    pdf.caption(
        "Annualised with 8,760 hours a year. Five of the eight configs are shown (the ones "
        "with the largest gross edge); the gross-to-drag ratio is below 1 for all eight."
    )

    # Section 4: The book and the lockbox
    pdf.h1("4. The book and the lockbox")
    pdf.body(
        "The final book combines Seasonality, Orderflow and Carry with walk-forward "
        "max-Sharpe weights, re-fitted every 63 days on the previous 756 days. An "
        "equal-weight book is shown alongside."
    )

    pdf.h2("4.1 Sleeve correlations")
    pdf.body(
        "The three sleeves are nearly uncorrelated over the dev window, which limits "
        "the benefit of mean-variance weighting over simple equal-weighting."
    )
    pdf.table_block(sleeve_corr_table(seas, of, carry), title="Dev-window sleeve pairwise correlation")

    pdf.h2("4.2 Performance across windows")
    pdf.table_block(windows_table(cp), title="Walk-forward and equal-weight book by window")
    pdf.caption(
        "The walk-forward book needs 756 days of training, so its dev figure (+0.40) only "
        "covers 2021-10 to 2024-07; equal weight uses the whole dev window (1,674 days). "
        "Equal weight beats walk-forward on every window except the lockbox."
    )

    pdf.h2("4.3 Equity curve and weight path")
    pdf.figure(figs["equity"])
    pdf.caption(
        "Top: cumulative return (log scale). Dotted navy line = gate start (2024-08-01); "
        "dashed red line = lockbox open (2025-07-01). "
        "Middle: walk-forward drawdown. Bottom: rolling 180-day Sharpe."
    )
    pdf.figure(figs["weights"])
    pdf.caption(
        "Walk-forward sleeve weights, re-estimated every 63 days on the previous 756 days."
    )

    pdf.h2("4.4 Lockbox verdict")
    pdf.body(
        "The lockbox (2025-07-01 to 2026-07-06, 371 days, opened once) gives:\n"
        "- Walk-forward Sharpe: +1.458  |  Equal-weight Sharpe: +1.233\n"
        "- BTC beta: -0.020  (via joint BTC + market regression)\n"
        "- Annualised alpha: 9.48%  |  Alpha t-stat: 1.25  (HAC, not significant)\n"
        f"- Block-bootstrap 95% CI for the walk-forward Sharpe: {_ci_txt}.\n\n"
        "Sleeves on the lockbox: Orderflow +1.95 and Carry +0.88 held up; Seasonality came "
        "in at -0.57, so its gate outperformance did not last, which is the regime risk "
        "flagged in notebook 02.\n\n"
        "Performance is positive with almost no market exposure, but at this sample size "
        "and trial count it cannot be told apart from selection luck."
    )
    pdf.table_block(lockbox_alpha_table(vals), title="Lockbox alpha / beta (HAC Newey-West)")
    pdf.table_block(significance_table(cp, vals, (boot_lo, boot_hi)), title="Significance summary")
    pdf.body(
        f"The deflated Sharpe accounts for all {N_ALPHA_TRIALS} research trials in the "
        "registry (baselines, seasonality, orderflow, carry, fast reversal). It is 0.490 on "
        "the full sample and 0.205 on the lockbox, both below the usual 0.95 bar. The result "
        "is positive but not statistically proven."
    )

    # Section 5: Limitations
    pdf.h1("5. Limitations")
    pdf.body(
        "Survivorship bias. The daily universe is rebuilt from the pairs Binance listed "
        "when the data was fetched (July 2026), so coins delisted earlier are missing. The "
        "point-in-time liquidity screen and the three-window split reduce this but cannot "
        "remove it.\n\n"
        "Single exchange. Volume data comes from Binance only. Taker imbalance and "
        "funding rates are Binance-specific; the signal on other venues may differ.\n\n"
        "Spot prices for a perp trade. The carry sleeve adds Binance funding payments to "
        "spot returns. A real perp position would also carry the basis between perp and "
        "spot, which the backtest ignores.\n\n"
        "Seasonality on the lockbox. The Seasonality sleeve lost money on the lockbox "
        "(-0.57 against +1.24 on the gate). One year cannot tell whether the weekend effect "
        "has faded or was never there.\n\n"
        "Capacity and higher costs untested. The book has not been sized for real money and "
        "has not been rerun at higher costs. Carry and Orderflow trade top-100 names, but "
        "large positions in the smaller ones would move prices.\n\n"
        "Thin early hourly panel. The hourly work starts in 2020-06 because the hourly panel "
        "is thin before that.\n\n"
        "Statistical significance. Against 48 enumerated trials the deflated Sharpe stays "
        "below 0.95 on both the full sample and the lockbox."
    )

    # Section 6: Appendix (trial registry)
    pdf.h1("6. Appendix: full trial registry (50 rows)")
    pdf.body(
        "Every configuration evaluated during the research is listed here in chronological "
        "order. Dev = 2020-01-01 to 2024-07-31; Gate = 2024-08-01 to 2025-06-30. "
        "Rows 49-50 are the portfolio combination entries (not alpha trials)."
    )
    pdf.table_block(registry_table(reg))

    # Write PDF
    OUT.parent.mkdir(exist_ok=True)
    pdf.output(str(OUT))
    pages = pdf.page
    size_kb = OUT.stat().st_size // 1024
    print(f"Wrote {OUT}  ({size_kb} KB, {pages} pages)")
    return pages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cp, bm, mom_base, rev_base, seas, of, carry, reg = load_inputs()
    btc = bm["BTC"]
    mkt = bm["MKT"]

    print("Asserting Section 4 numbers...")
    vals = assert_section4(cp, btc, mkt)
    print("  All assertions passed.")

    with tempfile.TemporaryDirectory() as td:
        figs = make_figures(cp, btc, seas, of, carry, Path(td))
        pages = build_pdf(
            cp, btc, mkt, mom_base, rev_base, seas, of, carry, reg, vals, figs
        )

    if pages < 8:
        raise RuntimeError(f"PDF has only {pages} pages, expected at least 8.")
    print(f"Done. {pages} pages.")


if __name__ == "__main__":
    main()
