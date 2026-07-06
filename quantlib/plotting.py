"""Plot style plus a few chart helpers. Importing the module sets the matplotlib
style that reports/build_report.py uses."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics

plt.rcParams.update({
    "figure.figsize": (11, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def plot_equity(returns_dict: dict[str, pd.Series], title: str = "Cumulative net return", logy: bool = True, ax=None):
    """Overlay equity curves for several strategies (key = label)."""
    if ax is None:
        _, ax = plt.subplots()
    for label, r in returns_dict.items():
        eq = (1.0 + r.fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=f"{label} (SR={metrics.sharpe(r):.2f})")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("growth of $1")
    return ax


def plot_drawdown(returns: pd.Series, title: str = "Drawdown", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.6))
    dd = metrics.drawdown_curve(returns)
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.35)
    ax.set_title(title)
    ax.set_ylabel("drawdown")
    return ax


def plot_rolling_sharpe(returns: pd.Series, window: int = 180, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.8))
    roll = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(metrics.TRADING_DAYS)
    ax.plot(roll.index, roll.values)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Rolling {window}d Sharpe")
    return ax


def plot_param_heatmap(table: pd.DataFrame, title: str = "In-sample Sharpe", ax=None, fmt="{:.2f}"):
    """Heatmap of a metric over a 2-D parameter grid (rows x cols = params)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * table.shape[1] + 2, 0.6 * table.shape[0] + 2))
    im = ax.imshow(table.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel(table.columns.name or "param 2")
    ax.set_ylabel(table.index.name or "param 1")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def show_summary(summaries: list[pd.Series]) -> pd.DataFrame:
    """Concatenate metric Series into a single formatted comparison table."""
    return pd.concat(summaries, axis=1)
