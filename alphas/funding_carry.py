"""twsq version of the carry sleeve (quantlib.strategies.carry_sleeve).

Signal: minus the cross-sectional z-score of the 7-day mean perp funding rate.
Short the names where longs pay the most funding, long the names where they
pay the least. Dollar-neutral, limit orders at 7 bps.

Known gap: the research sleeve holds perps and collects funding on its shorts,
which is a large part of its P&L. twsq backtests spot positions only, so this
version earns the price leg alone and understates the sleeve.

Funding history comes from Binance's public /fapi/v1/fundingRate endpoint.
prepare() downloads it once and drops symbols without a perp.
"""
from twsq.alpha import Alpha
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _binance_data import funding_history


SMOOTH = 7   # rolling window (days) for the funding signal


class FundingCarry(Alpha):
    def prepare(self, symbols, **kwargs):
        self.smooth = kwargs.get("smooth", SMOOTH)
        self.capital = kwargs.get("capital", 1_000_000)
        hist_days = kwargs.get("hist_days", 730)

        # Download the whole window once; drop symbols without a perp.
        print(f"funding_carry: fetching {hist_days}d funding history from Binance")
        raw = funding_history(symbols, days=hist_days)
        # all-NaN column = no perp on Binance
        raw = raw.dropna(axis=1, how="all")
        raw.index = pd.to_datetime(raw.index).normalize()
        self._fund = raw
        self.symbols = list(self._fund.columns)
        print(f"funding_carry: {len(self.symbols)} symbols with perp data "
              f"({self._fund.index.min().date()} to {self._fund.index.max().date()}): "
              f"{self.symbols}")

    def _zscore(self, series: pd.Series) -> pd.Series:
        mu, sd = series.mean(), series.std()
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sd

    def rebalance(self):
        ts = pd.Timestamp(self.ts).normalize()

        # Downloaded data up to the rebalance date.
        hist = self._fund[self._fund.index <= ts].dropna(axis=1, how="all")
        if len(hist) < self.smooth:
            return

        # Signal: minus the z-score of the smoothed funding (short high funding).
        mean_fund = hist.tail(self.smooth).mean()
        valid = mean_fund.dropna()
        if len(valid) < 2:
            return
        z = -self._zscore(valid)   # negative: short high-funding

        # Dollar-neutral, gross 1.
        pos = z.clip(lower=0)
        neg = (-z).clip(lower=0)
        pos_sum = pos.sum()
        neg_sum = neg.sum()
        weights = pd.Series(0.0, index=z.index)
        if pos_sum > 0:
            weights += pos / pos_sum * 0.5
        if neg_sum > 0:
            weights -= neg / neg_sum * 0.5

        target = {}
        for symbol, w in weights.items():
            if abs(w) < 1e-8:
                continue
            try:
                px = self.get_current_price(symbol + "/USD")
            except Exception:  # noqa: BLE001
                continue
            if px and px > 0:
                target[symbol] = w * self.capital / px

        self._trade_limit(target)

    def _trade_limit(self, target: dict):
        pos = self.get_pos()
        for symbol, tgt_qty in target.items():
            delta = tgt_qty - pos.get(symbol, 0.0)
            if abs(delta) < 1e-8:
                continue
            side = "buy" if delta > 0 else "sell"
            try:
                px = self.get_current_price(symbol + "/USD")
                self.create_order(symbol + "/USD", abs(delta), side,
                                  limit_price=px, route=True)
            except Exception as e:  # noqa: BLE001
                print(f"funding_carry: order error {symbol}: {e}")
        # Close positions no longer in target (skip USDT cash balance).
        for symbol, qty in pos.items():
            if symbol == "USDT":
                continue
            if symbol not in target and abs(qty) > 1e-8:
                side = "sell" if qty > 0 else "buy"
                try:
                    px = self.get_current_price(symbol + "/USD")
                    self.create_order(symbol + "/USD", abs(qty), side,
                                      limit_price=px, route=True)
                except Exception as e:  # noqa: BLE001
                    print(f"funding_carry: close error {symbol}: {e}")

    def on_exit(self):
        self.cancel_all_orders()


# Same 20 names; prepare() drops any without a Binance perp.
symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
           "LTC", "BCH", "ATOM", "XLM", "ETC", "FIL", "APT", "NEAR", "ARB", "OP"]

if __name__ == "__main__":
    result = FundingCarry.run_backtest(
        start_ts="20200101", freq="1d",
        maker_fee=7e-4, slip=0.0,
        symbols=symbols, smooth=7, hist_days=730,
    )
    print(result.pos_pnl.tail())
