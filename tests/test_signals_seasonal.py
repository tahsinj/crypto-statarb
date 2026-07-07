"""Calendar masks and rolling z-scores."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import signals  # noqa: E402


def test_weekday_weekend_partition():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")  # Mon 2024-01-01
    wd = signals.calendar_mask(idx, "weekday")
    we = signals.calendar_mask(idx, "weekend")
    assert (wd ^ we).all()                      # exact partition
    assert we[idx.dayofweek >= 5].all() and not we[idx.dayofweek < 5].any()


def test_us_hours_bucket():
    idx = pd.date_range("2024-01-02 00:00", periods=48, freq="h")  # Tue-Wed
    m = signals.calendar_mask(idx, "us_hours")
    assert m[idx.hour == 14].all() and m[idx.hour == 20].all()
    assert not m[idx.hour == 21].any() and not m[idx.hour == 13].any()
    sat = pd.date_range("2024-01-06 14:00", periods=3, freq="h")  # Saturday
    assert not signals.calendar_mask(sat, "us_hours").any()
    off = signals.calendar_mask(idx, "off_hours")
    assert (off ^ signals.calendar_mask(idx, "us_hours")).all()


def test_turn_of_month():
    idx = pd.date_range("2024-01-28", "2024-02-06", freq="D")
    m = signals.calendar_mask(idx, "turn_of_month")
    assert m[idx.day == 31].all() and m[idx.day == 1].all() and m[idx.day == 3].all()
    assert not m[idx.day == 28].any() and not m[idx.day == 6].any()


def test_unknown_bucket_raises():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    try:
        signals.calendar_mask(idx, "lunar_phase")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "lunar_phase" in str(e)


def test_rolling_zscore_shift_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"A": np.arange(100, dtype=float)}, index=idx)
    z = signals.rolling_zscore(df, window=20, shift=True)
    # With shift=True the stats for date t come from data through t-1, so for
    # a strictly increasing series each score is larger than the unshifted one
    # (x_t sits above a mean that excludes it).
    z0 = signals.rolling_zscore(df, window=20, shift=False)
    both = pd.concat([z["A"], z0["A"]], axis=1).dropna()
    assert (both.iloc[:, 0] > both.iloc[:, 1]).all()


def test_rolling_zscore_values():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 10.0]}, index=idx)
    z = signals.rolling_zscore(df, window=3, min_periods=3, shift=True)
    # Date 5: stats from [2,3,4] (shifted window) -> mean 3, std 1 -> (10-3)/1 = 7
    assert np.isclose(z["A"].iloc[-1], 7.0)
    assert z["A"].iloc[:3].isna().all()  # not enough shifted history


if __name__ == "__main__":
    for fn in [test_weekday_weekend_partition, test_us_hours_bucket,
               test_turn_of_month, test_unknown_bucket_raises,
               test_rolling_zscore_shift_no_lookahead, test_rolling_zscore_values]:
        fn()
        print(f"ok {fn.__name__}")
