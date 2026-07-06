"""quantlib: a small toolkit for the crypto stat-arb research.

Data handling, signal construction, the backtest engine, and performance
metrics, imported by every notebook so the calculations live in one place.
"""
from . import backtest, data, fetch, metrics, pairs, plotting, robustness, signals, strategies, trials

__all__ = [
    "data", "fetch", "signals", "backtest", "metrics", "plotting", "pairs",
    "strategies", "robustness", "trials",
]
