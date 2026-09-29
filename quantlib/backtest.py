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


def fill_gap_starts(returns: pd.DataFrame, fallback: pd.DataFrame,
                    price: pd.DataFrame | None = None) -> pd.DataFrame:
    """``returns`` with the first day of each run of missing values taken from ``fallback``.

    Made for perp returns, which the every-pair panels drop on days a contract
    trades more than 20% away from spot or stops trading (data.archive_panels).
    A position held into such a day still moved, on a crash day or into a
    delisting, and the coin's spot move stands in for it. Later days in the
    same gap stay missing, so a position that cannot be closed (run_limit_fills
    needs a bar to fill) earns nothing while it waits; run_limit_fills settles
    it if the gap lasts, which is what happens to a delisted perp.

    With ``price`` (the same panel's prices), a day the contract trades again
    but has no return yet, because its previous close was dropped, takes the
    fallback too: a position still open then moved with it.
    """
    fill = returns.isna() & returns.shift(1).notna()
    if price is not None:
        fill |= returns.isna() & price.reindex_like(returns).notna()
    return returns.fillna(fallback.reindex_like(returns).where(fill))


def run_limit_fills(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cost_bps: float = 7.0,
    settle_after: int = 3,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Backtest where every trade is a limit order at the close it was decided on.

    run() assumes each day's trades fill at the close. Here an order to buy at
    the close of day t fills only if day t+1 trades below that price, and an
    order to sell only if it trades above it. A filled order earns the move
    from its limit price, so its P&L matches run(); an unfilled one is
    cancelled, the old position is kept for the day, and the next close sends
    a new order toward the new target. Orders miss exactly when the price runs
    away from them, which run() cannot show. Costs are charged on filled
    trades, on the day they fill. A coin with no bar cannot fill. A position
    whose contract has had no bar for more than ``settle_after`` days is
    settled at its last price and dropped, as an exchange does with a
    delisted contract, so it cannot come back to life on a later contract
    under the same name (LUNA and the relaunched LUNA2); after a shorter gap
    it carries on. Returns the result and the positions held.
    """
    idx, cols = returns.index, returns.columns
    w = weights.reindex(index=idx, columns=cols).fillna(0.0).to_numpy()
    c = close.reindex(index=idx, columns=cols).to_numpy()
    hi = high.reindex(index=idx, columns=cols).to_numpy()
    lo = low.reindex(index=idx, columns=cols).to_numpy()
    held = np.zeros_like(w)
    traded = np.zeros(len(idx))
    q = np.zeros(len(cols))
    gone = np.zeros(len(cols), dtype=int)                  # days in a row without a bar
    with np.errstate(invalid="ignore"):
        for t in range(1, len(idx)):
            gone = np.where(np.isnan(c[t]), gone + 1, 0)
            q = np.where(gone > settle_after, 0.0, q)      # settled at its last price
            order = w[t - 1] - q
            fill = ((order > 0) & (lo[t] < c[t - 1])) | ((order < 0) & (hi[t] > c[t - 1]))
            done = np.where(fill, order, 0.0)
            q = q + done
            held[t] = q
            traded[t] = np.abs(done).sum()
    held = pd.DataFrame(held, index=idx, columns=cols)
    gross = (held * returns).sum(axis=1).iloc[1:]
    turnover = pd.Series(traded, index=idx).iloc[1:]
    costs = turnover * (cost_bps / 1e4)
    res = BacktestResult(
        gross_returns=gross.rename("gross"),
        net_returns=(gross - costs).rename("net"),
        turnover=turnover.rename("turnover"),
        costs=costs.rename("costs"),
        weights=weights,
        cost_bps=cost_bps,
        carry=pd.Series(0.0, index=gross.index, name="carry"),
    )
    return res, held


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
