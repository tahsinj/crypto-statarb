"""twsq version of the seasonality sleeve (quantlib.strategies.seasonal_momentum_sleeve).

Vol-targeted time-series momentum with a weekday/weekend overlay: full size on
Saturday and Sunday (UTC), half size Monday to Friday. Market orders, charged
7 bps commission plus 13 bps slippage, 20 bps in total.
"""
from twsq.alpha import Alpha
import numpy as np
import pandas as pd

TRADING_DAYS = 365


class SeasonalMomentum(Alpha):
    def prepare(self, symbols, **kwargs):
        self.symbols = symbols
        self.lookback = kwargs.get("lookback", 30)        # trend horizon (days)
        self.vol_window = kwargs.get("vol_window", 30)    # vol-estimate window
        self.target_vol = kwargs.get("target_vol", 0.15)  # annualised book vol
        self.capital = kwargs.get("capital", 1_000_000)
        self.max_leverage = kwargs.get("max_leverage", 3.0)
        self.weekend_scale = kwargs.get("weekend_scale", 1.0)
        self.weekday_scale = kwargs.get("weekday_scale", 0.5)

    def _bars(self, symbol, n):
        # daily bars up to the last completed day
        return self.get_lastn_bars(symbol + "/USD", n, "1d")

    def rebalance(self):
        n = max(self.lookback, self.vol_window) + 5
        raw = {}
        for symbol in self.symbols:
            try:
                close = self._bars(symbol, n)["close"]
                if close.notna().sum() < n - 2:
                    continue
                trail_ret = close.iloc[-1] / close.iloc[-self.lookback - 1] - 1.0
                daily_vol = close.pct_change().iloc[-self.vol_window:].std()
                if daily_vol <= 0 or np.isnan(daily_vol) or np.isnan(trail_ret):
                    continue
                raw[symbol] = np.sign(trail_ret) / daily_vol
            except Exception as e:  # noqa: BLE001
                print(f"seasonal_momentum: skipping {symbol}: {e}")

        if not raw:
            return

        s = pd.Series(raw)
        w = s / s.abs().sum()
        port_daily_vol = (w.abs() * (1.0 / s.abs())).sum()
        scale = self.target_vol / (port_daily_vol * np.sqrt(TRADING_DAYS) + 1e-12)
        scale = float(np.clip(scale, 0, self.max_leverage))
        w = w * scale

        # Weekday/weekend overlay; self.ts is the rebalance time in UTC.
        dow = self.ts.dayofweek  # 0 = Monday, 6 = Sunday
        season_scale = self.weekend_scale if dow >= 5 else self.weekday_scale
        w = w * season_scale

        target = {}
        for symbol, weight in w.items():
            px = self.get_current_price(symbol + "/USD")
            if px and px > 0:
                target[symbol] = weight * self.capital / px
        self.trade_to_target(target, route=True)

    def on_exit(self):
        self.cancel_all_orders()


# Fixed universe: 20 large coins, all listed on Binance for the whole backtest.
symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
           "LTC", "BCH", "ATOM", "XLM", "ETC", "FIL", "APT", "NEAR", "ARB", "OP"]

if __name__ == "__main__":
    result = SeasonalMomentum.run_backtest(
        start_ts="20200101", freq="1d",
        taker_fee=7e-4, slip=13e-4,
        symbols=symbols, lookback=30, vol_window=30, target_vol=0.15,
        weekend_scale=1.0, weekday_scale=0.5,
    )
    print(result.pos_pnl.tail())
