"""twsq version of the orderflow sleeve (quantlib.strategies.orderflow_sleeve).

Signal: cross-sectional z-score of the 10-day mean of (taker-buy share - 0.5).
Long the names with more aggressive buying, short the names with more
aggressive selling. Dollar-neutral, limit orders at 7 bps.

The taker-buy share comes from Binance spot klines (taker quote volume over
total quote volume). prepare() downloads the whole history once, so
rebalance() only has to slice a DataFrame.
"""
from twsq.alpha import Alpha
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _binance_data import taker_imbalance_history


SMOOTH = 10          # rolling mean window for the imbalance signal


class OrderflowFollow(Alpha):
    def prepare(self, symbols, **kwargs):
        self.symbols = symbols
        self.smooth = kwargs.get("smooth", SMOOTH)
        self.capital = kwargs.get("capital", 1_000_000)
        # days of taker history to download
        hist_days = kwargs.get("hist_days", 730)

        # Download the whole window once; rebalance() only slices it.
        print(f"orderflow: fetching {hist_days}d taker history from Binance")
        raw = taker_imbalance_history(self.symbols, days=hist_days)
        # Index by calendar date.
        raw.index = pd.to_datetime(raw.index).normalize()
        self._imb = raw
        print(f"orderflow: got {len(self._imb)} days x {len(self._imb.columns)} symbols "
              f"({self._imb.index.min().date()} to {self._imb.index.max().date()})")

    def _zscore(self, series: pd.Series) -> pd.Series:
        mu, sd = series.mean(), series.std()
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sd

    def rebalance(self):
        ts = pd.Timestamp(self.ts).normalize()

        # Downloaded data up to the rebalance date.
        hist = self._imb[self._imb.index <= ts]
        if len(hist) < self.smooth:
            return

        # Signal: cross-sectional z-score of smooth-day mean (taker_share - 0.5).
        centered = (hist.tail(self.smooth) - 0.5).mean()
        valid = centered.dropna()
        if len(valid) < 2:
            return
        z = self._zscore(valid)

        # Dollar-neutral: long positives, short negatives. Gross = 1.
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

        # Limit orders at the last close (7 bps tier).
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
                print(f"orderflow: order error {symbol}: {e}")
        # Close any positions in symbols no longer in target (skip USDT cash balance).
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
                    print(f"orderflow: close error {symbol}: {e}")

    def on_exit(self):
        self.cancel_all_orders()


# Same 20 names as SeasonalMomentum.
symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
           "LTC", "BCH", "ATOM", "XLM", "ETC", "FIL", "APT", "NEAR", "ARB", "OP"]

if __name__ == "__main__":
    result = OrderflowFollow.run_backtest(
        start_ts="20200101", freq="1d",
        maker_fee=7e-4, slip=0.0,
        symbols=symbols, smooth=10, hist_days=730,
    )
    print(result.pos_pnl.tail())
