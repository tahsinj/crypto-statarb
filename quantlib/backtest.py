"""Vectorised backtest engine.

The weight lag in run() is what keeps the backtest look-ahead free: weights
formed at the close of day t are applied to the t->t+1 return, so trading always
happens after the signal is observable.

Costs are charged on turnover: 20 bps for market orders (7 commission + 13
slippage), 7 bps for limit orders (commission only). An optional borrow/funding
carry can be charged on short notional held (crypto shorts really pay this).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 365  # crypto trades every day


@dataclass
class BacktestResult:
    gross_returns: pd.Series      # before costs, per day
    net_returns: pd.Series        # after costs, per day
    turnover: pd.Series           # sum |Δweight| per day (one-way)
    costs: pd.Series              # turnover cost drag per day
    weights: pd.Series | pd.DataFrame  # the (pre-lag) target weights used
    cost_bps: float
    carry: pd.Series              # borrow/funding drag on shorts per day
    borrow_bps_annual: float = 0.0

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net_returns.fillna(0.0)).cumprod()

    def slice(self, start=None, end=None) -> "BacktestResult":
        """Restrict the result to a date window (for IS/OOS splits)."""
        sl = slice(start, end)
        return BacktestResult(
            gross_returns=self.gross_returns.loc[sl],
            net_returns=self.net_returns.loc[sl],
            turnover=self.turnover.loc[sl],
            costs=self.costs.loc[sl],
            weights=self.weights.loc[sl] if hasattr(self.weights, "loc") else self.weights,
            cost_bps=self.cost_bps,
            carry=self.carry.loc[sl],
            borrow_bps_annual=self.borrow_bps_annual,
        )


def run(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 20.0,
    borrow_bps_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestResult:
    """Backtest target weights against a return panel.

    weights and returns are both dates x symbols. cost_bps is per unit of
    turnover (20 for market orders, 7 for limit). borrow_bps_annual charges a
    daily borrow/funding carry on the short notional held (default 0 = off).
    """
    w = weights.reindex_like(returns).fillna(0.0)

    held = w.shift(1)                       # yesterday's target -> no look-ahead
    gross = (held * returns).sum(axis=1)

    turnover = (w - w.shift(1)).abs().sum(axis=1)
    costs = turnover * (cost_bps / 1e4)
    # Carry on short notional held that day, prorated from an annual borrow rate.
    short_notional = held.clip(upper=0.0).abs().sum(axis=1)
    carry = short_notional * (borrow_bps_annual / 1e4) / periods_per_year
    net = gross - costs - carry

    # Drop the leading warm-up day that has no prior weights.
    valid = held.notna().any(axis=1)
    gross, net, turnover, costs, carry = (
        s[valid] for s in (gross, net, turnover, costs, carry)
    )

    return BacktestResult(
        gross_returns=gross.rename("gross"),
        net_returns=net.rename("net"),
        turnover=turnover.rename("turnover"),
        costs=costs.rename("costs"),
        weights=weights,
        cost_bps=cost_bps,
        carry=carry.rename("carry"),
        borrow_bps_annual=borrow_bps_annual,
    )


def vol_target_scale(returns: pd.Series, target_vol: float = 0.10, halflife: int = 30,
                     periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Leverage that scales a return stream to target_vol, using vol through t-1."""
    realised = returns.ewm(halflife=halflife, min_periods=halflife).std().shift(1)
    realised_ann = realised * np.sqrt(periods_per_year)
    return (target_vol / realised_ann).clip(upper=5.0)  # cap leverage


def vol_target(returns: pd.Series, target_vol: float = 0.10, halflife: int = 30,
               periods_per_year: int = TRADING_DAYS) -> pd.Series:
    """Scale a return stream to a constant annualised target vol.

    The scale at t uses vol estimated through t-1 only, so the result stays
    tradable. Scaling the returns does not charge for the trades that changing
    leverage would need; for a slowly moving scale that cost is small.
    """
    scale = vol_target_scale(returns, target_vol, halflife, periods_per_year)
    return (returns * scale).rename(returns.name)
