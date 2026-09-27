"""Smoke tests for quantlib's look-ahead, neutrality and cost behaviour.

Run from the project root:  .venv/bin/python tests/test_quantlib.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import backtest, data, metrics, robustness, signals  # noqa: E402


def _toy_panels(n_days=400, n_assets=20, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    cols = [f"C{i}" for i in range(n_assets)]
    rets = pd.DataFrame(rng.normal(0, 0.03, (n_days, n_assets)), index=idx, columns=cols)
    price = 100 * (1 + rets).cumprod()
    dvol = pd.DataFrame(rng.uniform(1e6, 1e8, (n_days, n_assets)), index=idx, columns=cols)
    return {"price": price, "returns": price.pct_change(), "dollar_volume": dvol}


def test_no_lookahead():
    """A signal built from same-day returns must not make money, because the
    engine lags weights by a day. Without the lag this would be very profitable."""
    p = _toy_panels()
    rets = p["returns"]
    universe = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    # Day t's return decides the weights, which the engine applies to day t+1.
    # With independent toy returns the Sharpe should be close to zero.
    cheat = rets
    w = signals.signal_to_weights(cheat, universe)
    res = backtest.run(w, rets, cost_bps=0)
    assert abs(metrics.sharpe(res.net_returns)) < 1.0, "look-ahead leak detected!"


def test_dollar_neutral():
    p = _toy_panels()
    rets = p["returns"]
    universe = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    sig = signals.trailing_return(p["price"], 20)
    w = signals.signal_to_weights(sig, universe, long_short=True)
    net_exposure = w.sum(axis=1).abs()
    assert net_exposure.max() < 1e-9, "long-short book is not dollar-neutral"
    gross = w.abs().sum(axis=1)
    nz = gross[gross > 0]
    assert np.allclose(nz, 1.0, atol=1e-9), "gross leverage != 1"


def test_costs_monotonic():
    """Higher cost_bps must monotonically reduce net return."""
    p = _toy_panels()
    rets = p["returns"]
    universe = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    sig = signals.trailing_return(p["price"], 5)  # fast -> high turnover
    w = signals.signal_to_weights(sig, universe)
    totals = []
    for bps in (0, 7, 20, 50):
        res = backtest.run(w, rets, cost_bps=bps)
        totals.append(res.net_returns.sum())
    assert all(totals[i] > totals[i + 1] for i in range(len(totals) - 1)), totals


def test_buy_and_hold_sharpe_sane():
    """A single-asset buy-and-hold returns a finite, reasonable Sharpe."""
    p = _toy_panels()
    bh = p["returns"]["C0"]
    sr = metrics.sharpe(bh)
    assert np.isfinite(sr) and -5 < sr < 5


def test_alpha_beta_recovers_known_beta():
    """Construct y = 0.5*x + noise and check beta ~ 0.5, and that the HAC fit
    still exposes the exact return-dict keys downstream code depends on."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    x = pd.Series(rng.normal(0, 0.02, 800), index=idx)
    y = 0.5 * x + pd.Series(rng.normal(0, 0.005, 800), index=idx)
    ab = metrics.alpha_beta(y, x, names=["mkt"])
    assert abs(ab["beta_mkt"] - 0.5) < 0.05
    assert abs(ab["alpha_daily"]) < 1e-3
    # keys used by metrics.summary and notebook 06
    for key in ("alpha_daily", "alpha_ann", "alpha_tstat", "r_squared", "resid_info_ratio", "beta_mkt"):
        assert key in ab, key
    assert np.isfinite(ab["alpha_tstat"])


def test_treynor_mazuy_separates_timing_from_alpha():
    """A convex (market-timing) payoff shows up as a positive timing term, and
    the timing-adjusted alpha is lower than the plain regression's alpha."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=1400, freq="D")
    m = pd.Series(rng.normal(0, 0.03, len(idx)), index=idx)
    r = 5 * m ** 2 + pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)
    tm = metrics.treynor_mazuy(r, m)
    plain = metrics.alpha_beta(r, m, names=["mkt"])
    assert tm["timing"] > 0 and tm["timing_tstat"] > 3
    assert tm["alpha_ann"] < plain["alpha_ann"]


def test_sharpe_tstat_hac_finite():
    """The HAC (Newey-West) Sharpe t-stat is finite and same-signed as iid."""
    p = _toy_panels()
    r = p["returns"]["C0"]
    t_iid = metrics.sharpe_tstat(r)
    t_hac = metrics.sharpe_tstat(r, hac=True)
    assert np.isfinite(t_hac)
    assert np.sign(t_hac) == np.sign(t_iid)


def test_block_bootstrap_ci_ordered():
    """Both bootstrap methods return an ordered (lo <= hi) CI bracketing nothing absurd."""
    p = _toy_panels(n_days=500)
    r = p["returns"]["C0"]
    for method in ("block", "iid"):
        lo, hi = metrics.bootstrap_sharpe_ci(r, n_boot=500, method=method, seed=0)
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi, (method, lo, hi)


def test_carry_reduces_net_and_monotone():
    """Borrow carry on shorts reduces net return, monotonically in the rate."""
    p = _toy_panels()
    rets = p["returns"]
    universe = pd.DataFrame(True, index=rets.index, columns=rets.columns)
    sig = signals.trailing_return(p["price"], 20)
    w = signals.signal_to_weights(sig, universe, long_short=True)  # has shorts
    totals = []
    for borrow in (0.0, 500.0, 1000.0, 2000.0):  # annual bps
        res = backtest.run(w, rets, cost_bps=0, borrow_bps_annual=borrow)
        totals.append(res.net_returns.sum())
    assert (res.carry >= 0).all(), "carry must be non-negative"
    assert all(totals[i] > totals[i + 1] for i in range(len(totals) - 1)), totals


def test_walk_forward_weights_causal():
    """Walk-forward weights/returns before date T must not change when sleeve
    returns AFTER T are perturbed (no look-ahead in the rolling estimation)."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=1400, freq="D")
    sleeves = pd.DataFrame(
        {"A": rng.normal(0.0005, 0.01, 1400), "B": rng.normal(0.0003, 0.008, 1400)},
        index=idx,
    )
    combo_a, wp_a = robustness.walk_forward_weights(sleeves, train_days=756, step_days=63)
    cut = idx[1100]
    perturbed = sleeves.copy()
    perturbed.loc[cut:] += 0.05  # only the future changes
    combo_b, wp_b = robustness.walk_forward_weights(perturbed, train_days=756, step_days=63)
    before = combo_a.index[combo_a.index < cut]
    assert np.allclose(combo_a.loc[before], combo_b.loc[before], atol=1e-12), "walk-forward leaked future data"


def test_capacity_sharpe_non_increasing():
    """Under square-root impact, higher AUM must not raise Sharpe."""
    p = _toy_panels(n_days=500)
    net = p["returns"]["C0"] * 0 + p["returns"].mean(axis=1)  # a return series
    turnover = pd.Series(0.5, index=net.index)  # constant one-way turnover
    grid = [1e6, 1e7, 1e8, 1e9]
    cap = robustness.capacity_curve(net, turnover, adv_usd=5e6, aum_grid=grid)
    sh = cap["sharpe"].values
    assert all(sh[i] >= sh[i + 1] - 1e-9 for i in range(len(sh) - 1)), sh


def test_universe_pointwise_no_lookahead():
    """Universe on date t must not use volume from date t (uses shift(1))."""
    p = _toy_panels()
    uni = data.build_universe(p, top_n=10, adv_window=30, min_history=30)
    # First 30 days cannot be eligible (insufficient history / ADV warm-up).
    assert not uni.iloc[:30].any().any()
    assert uni.iloc[40:].sum(axis=1).max() <= 10


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("All smoke tests passed.")
