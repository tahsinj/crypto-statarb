"""twsq version of the seasonality sleeve (quantlib.strategies.seasonal_momentum_sleeve).

Time-series momentum: each coin is held long or short by the sign of its
30-day return, skipping the latest day, all at the same size. The book is
scaled to 15% annualised vol from an EWMA (30-day half-life) of its own daily
returns, capped at 5x, and a weekday/weekend overlay then holds it at full
size on Saturday and Sunday (UTC) and half size Monday to Friday.

Market orders, charged 7 bps commission plus 13 bps slippage (20 bps in
total). twsq charges every trade, including the twice-weekly resizing, so
this matches the fully charged version in notebook 07 rather than the frozen
research sleeve, which skips the resizing cost.
"""
from twsq.alpha import Alpha
import numpy as np
import pandas as pd

TRADING_DAYS = 365


class SeasonalMomentum(Alpha):
    def prepare(self, symbols, **kwargs):
        self.symbols = symbols
        self.lookback = kwargs.get("lookback", 30)          # trend horizon (days)
        self.halflife = kwargs.get("halflife", 30)          # vol estimate half-life (days)
        self.history = kwargs.get("history", 180)           # days of book returns for the vol estimate
        self.target_vol = kwargs.get("target_vol", 0.15)    # annualised book vol
        self.max_leverage = kwargs.get("max_leverage", 5.0)
        self.capital = kwargs.get("capital", 1_000_000)
        self.weekend_scale = kwargs.get("weekend_scale", 1.0)
        self.weekday_scale = kwargs.get("weekday_scale", 0.5)

    def _closes(self) -> pd.DataFrame:
        """Daily closes up to the last completed day, one column per symbol."""
        n = self.history + self.lookback + 2
        closes = {}
        for symbol in self.symbols:
            try:
                bars = self.get_lastn_bars(symbol + "/USD", n, "1d")
            except Exception as e:  # noqa: BLE001
                print(f"seasonal_momentum: skipping {symbol}: {e}")
                continue
            if len(bars):
                closes[symbol] = bars["close"]
        return pd.DataFrame(closes)

    def rebalance(self):
        px = self._closes()
        if len(px) < self.lookback + self.halflife + 2:
            return

        # Weight for the day after row t: sign of the return from t-lookback-1
        # to t-1 (the latest day is skipped), equal size across names.
        trail = px.shift(1) / px.shift(self.lookback + 1) - 1.0
        sign = np.sign(trail)
        w = sign.div(sign.abs().sum(axis=1).replace(0, np.nan), axis=0)

        # The unscaled book's daily returns (gross of costs) give the vol estimate.
        book = (w.shift(1) * px.pct_change(fill_method=None)).sum(axis=1, min_count=1).dropna()
        vol = book.ewm(halflife=self.halflife, min_periods=self.halflife).std().iloc[-1]
        vol *= np.sqrt(TRADING_DAYS)
        if not np.isfinite(vol) or vol <= 0:
            return
        scale = min(self.target_vol / vol, self.max_leverage)

        # Weekday/weekend overlay; self.ts is the rebalance time in UTC and the
        # position is held through that day.
        dow = self.ts.dayofweek  # 0 = Monday, 6 = Sunday
        season = self.weekend_scale if dow >= 5 else self.weekday_scale

        weights = w.iloc[-1].fillna(0.0) * scale * season
        target = {}
        for symbol, weight in weights.items():
            px_now = self.get_current_price(symbol + "/USD")
            if px_now and px_now > 0:
                target[symbol] = weight * self.capital / px_now
        self.trade_to_target(target, route=True)

    def on_exit(self):
        self.cancel_all_orders()


# Fixed universe: 20 large coins, all listed on Binance for the whole backtest.
symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
           "LTC", "BCH", "ATOM", "XLM", "ETC", "FIL", "APT", "NEAR", "ARB", "OP"]

if __name__ == "__main__":
    result = SeasonalMomentum.run_backtest(
        start_ts="20240806", end_ts="20260707", freq="1d",
        taker_fee=7e-4, slip=13e-4,
        symbols=symbols, lookback=30, target_vol=0.15,
        weekend_scale=1.0, weekday_scale=0.5,
    )
    print(result.pos_pnl.tail())
