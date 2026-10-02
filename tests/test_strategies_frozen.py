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
    return price, price.pct_change(fill_method=None), uni, imb, fund


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


def test_carry_pnl_funding_pays_a_held_coins_dropped_day():
    """Funding dropped from the signal's data still counts for a coin held into that day."""
    price, rets, uni, _, fund = _toy()
    held = strategies.carry_weights(fund, uni).shift(1)
    day = rets.index[200]
    coin = held.loc[day].abs().idxmax()                    # the largest position held that day
    dropped = fund.copy()
    dropped.loc[day, coin] = np.nan                        # the 20% rule dropped its data that day
    as_run = strategies.carry_sleeve(dropped, rets, uni)
    paid = strategies.carry_sleeve(dropped, rets, uni, pnl_funding=fund)
    assert np.isclose(as_run[day] - paid[day], held.loc[day, coin] * fund.loc[day, coin])
    assert np.allclose(as_run.drop(day).dropna(), paid.drop(day).dropna())   # same signal, same trades


def test_v2_is_as_registered_by_default():
    price, rets, uni, imb, fund = _toy()
    perp = rets.copy()
    perp.iloc[[150, 170, 190], :3] = np.nan                # perp data dropped on a few days
    registered = strategies.v2_sleeves(imb, rets, perp, fund, uni)
    no_fill = strategies.carry_sleeve(fund, perp, uni, weighting="rank")
    assert np.array_equal(registered["carry"].to_numpy(), no_fill.to_numpy(), equal_nan=True)
    tested = strategies.v2_sleeves(imb, rets, rets, fund, uni, pnl_funding=2 * fund)
    ref = strategies.carry_sleeve(fund, rets, uni, weighting="rank", pnl_funding=2 * fund)
    assert np.array_equal(tested["carry"].to_numpy(), ref.to_numpy(), equal_nan=True)
    assert tested["orderflow"].equals(registered["orderflow"])


def test_rank_weights_keep_one_coin_from_taking_a_side():
    """One coin with extreme funding takes the whole long side under z-scores, not under ranks."""
    idx = pd.date_range("2022-01-01", periods=30)
    cols = [f"C{i}" for i in range(40)]
    fund = pd.DataFrame(np.tile(np.linspace(0, 1e-3, 40), (30, 1)), index=idx, columns=cols)
    fund["C0"] = -0.2                                      # the crash coin: shorts pay 20% a day
    uni = pd.DataFrame(True, index=idx, columns=cols)
    wz = strategies.carry_weights(fund, uni, weighting="zscore").iloc[-1]
    wr = strategies.carry_weights(fund, uni, weighting="rank").iloc[-1]
    assert wz["C0"] > 0.45 and wr["C0"] < 0.05
    assert np.isclose(wr.abs().sum(), 1.0) and abs(wr.sum()) < 1e-12


def test_carry_default_is_the_frozen_zscore_rule():
    price, rets, uni, _, fund = _toy()
    a = strategies.carry_sleeve(fund, rets, uni)
    b = strategies.carry_sleeve(fund, rets, uni, weighting="zscore")
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_v2_book_is_the_mean_of_its_sleeves():
    price, rets, uni, imb, fund = _toy()
    sl = strategies.v2_sleeves(imb, rets, rets, fund, uni)
    assert list(sl.columns) == ["orderflow", "carry"]
    book = strategies.v2_book(sl)
    assert np.allclose(book, sl.dropna().mean(axis=1)) and book.name == "v2"


if __name__ == "__main__":
    for fn in [test_seasonal_momentum_is_scaled_momentum,
               test_charge_resizing_matches_frozen_at_zero_cost,
               test_charge_resizing_costs_more_when_the_book_is_resized,
               test_orderflow_sleeve_runs_and_named,
               test_carry_sleeve_receives_funding_on_shorts,
               test_carry_pnl_funding_pays_a_held_coins_dropped_day,
               test_v2_is_as_registered_by_default,
               test_rank_weights_keep_one_coin_from_taking_a_side,
               test_carry_default_is_the_frozen_zscore_rule,
               test_v2_book_is_the_mean_of_its_sleeves]:
        fn()
        print(f"ok {fn.__name__}")
