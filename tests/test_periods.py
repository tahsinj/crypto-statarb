"""periods_per_year must scale annualized stats by sqrt(k) and leave defaults unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import backtest, metrics  # noqa: E402


def _r(n=800, seed=0, mu=2e-4, sd=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.date_range("2023-01-01", periods=n, freq="h"))


def test_sharpe_scaling():
    r = _r()
    # 4*365 periods/year -> sqrt(4) = 2x the default-365 Sharpe
    assert np.isclose(metrics.sharpe(r, periods_per_year=4 * 365), 2 * metrics.sharpe(r))


def test_vol_scaling_and_defaults():
    r = _r(seed=1)
    assert np.isclose(metrics.ann_vol(r, periods_per_year=4 * 365), 2 * metrics.ann_vol(r))
    # the default annualisation is 365 periods a year
    assert np.isclose(metrics.sharpe(r), r.mean() / r.std() * np.sqrt(365))
    assert np.isclose(metrics.ann_vol(r), r.std() * np.sqrt(365))


def test_ann_return_periods():
    r = _r(seed=2)
    k = 24 * 365
    expected = (1 + r).prod() ** (k / len(r)) - 1
    assert np.isclose(metrics.ann_return(r, periods_per_year=k), expected)


def test_summary_threads_periods():
    r = _r(seed=3)
    s365 = metrics.summary(r)
    s_h = metrics.summary(r, periods_per_year=24 * 365)
    assert np.isclose(s_h["Sharpe"], metrics.sharpe(r, periods_per_year=24 * 365))
    assert not np.isclose(s365["Sharpe"], s_h["Sharpe"])


def test_vol_target_periods():
    r = _r(seed=4)
    out365 = backtest.vol_target(r, 0.10)
    out_h = backtest.vol_target(r, 0.10, periods_per_year=24 * 365)
    # hourly annualization means realized ann vol is larger -> scale is smaller
    assert out_h.abs().sum() < out365.abs().sum()


if __name__ == "__main__":
    for fn in [test_sharpe_scaling, test_vol_scaling_and_defaults, test_ann_return_periods,
               test_summary_threads_periods, test_vol_target_periods]:
        fn()
        print(f"ok {fn.__name__}")
