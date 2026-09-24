"""Frozen survivor sleeves: construction invariants on toy panels."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import strategies  # noqa: E402


def _toy(n=400, k=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    cols = [f"C{i}" for i in range(k)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n, k)), index=idx, columns=cols)
    price = 100 * (1 + rets).cumprod()
    uni = pd.DataFrame(True, index=idx, columns=cols)
    imb = pd.DataFrame(rng.uniform(0.3, 0.7, (n, k)), index=idx, columns=cols)
    fund = pd.DataFrame(rng.normal(1e-4, 3e-4, (n, k)), index=idx, columns=cols)
    return price, price.pct_change(), uni, imb, fund


def test_seasonal_momentum_is_scaled_momentum():
    price, rets, uni, _, _ = _toy()
    base = strategies.momentum_sleeve(price, rets, uni)
    seas = strategies.seasonal_momentum_sleeve(price, rets, uni, weekend=1.0, weekday=0.5)
    both = pd.concat([base, seas], axis=1).dropna()
    wd = both.index.dayofweek < 5
    assert np.allclose(both.loc[wd].iloc[:, 1], 0.5 * both.loc[wd].iloc[:, 0])
    assert np.allclose(both.loc[~wd].iloc[:, 1], both.loc[~wd].iloc[:, 0])
    assert seas.name == "seasonality"


def test_charge_resizing_matches_frozen_at_zero_cost():
    # With no costs the two versions hold the same positions, so they must agree.
    price, rets, uni, _, _ = _toy()
    frozen = strategies.seasonal_momentum_sleeve(price, rets, uni, cost_bps=0.0)
    charged = strategies.seasonal_momentum_sleeve(price, rets, uni, cost_bps=0.0,
                                                  charge_resizing=True)
    both = pd.concat([frozen, charged], axis=1).dropna()
    assert len(both) > 300
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-12)


def test_charge_resizing_costs_more_when_the_book_is_resized():
    # Halving the book on weekdays means extra trades twice a week, so the
    # fully charged version must earn less than the frozen one.
    price, rets, uni, _, _ = _toy()
    frozen = strategies.seasonal_momentum_sleeve(price, rets, uni, cost_bps=20.0)
    charged = strategies.seasonal_momentum_sleeve(price, rets, uni, cost_bps=20.0,
                                                  charge_resizing=True)
    both = pd.concat([frozen, charged], axis=1).dropna()
    assert both.iloc[:, 1].sum() < both.iloc[:, 0].sum()
    assert charged.name == "seasonality"


def test_orderflow_sleeve_runs_and_named():
    price, rets, uni, imb, _ = _toy()
    s = strategies.orderflow_sleeve(imb, rets, uni)
    assert s.name == "orderflow" and s.dropna().abs().sum() > 0


def test_carry_sleeve_receives_funding_on_shorts():
    n, k, seed = 400, 12, 3
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    cols = [f"C{i}" for i in range(k)]
    # Constant funding level per asset, so there is real cross-sectional
    # carry to collect.
    fund = pd.DataFrame(
        np.tile(rng.normal(1e-4, 3e-4, k), (n, 1)),
        index=idx, columns=cols,
    )
    uni = pd.DataFrame(True, index=idx, columns=cols)
    zero = pd.DataFrame(0.0, index=idx, columns=cols)
    # Zero price returns isolate the funding leg: shorting high-funding names
    # should receive funding, so mean P&L before costs is positive.
    s = strategies.carry_sleeve(fund, zero, uni, cost_bps=0.0)
    assert s.name == "carry"
    assert s.dropna().mean() > 0, "carry sleeve should collect funding on shorts"


if __name__ == "__main__":
    for fn in [test_seasonal_momentum_is_scaled_momentum,
               test_charge_resizing_matches_frozen_at_zero_cost,
               test_charge_resizing_costs_more_when_the_book_is_resized,
               test_orderflow_sleeve_runs_and_named,
               test_carry_sleeve_receives_funding_on_shorts]:
        fn()
        print(f"ok {fn.__name__}")
