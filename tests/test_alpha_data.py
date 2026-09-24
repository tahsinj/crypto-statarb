"""Timing guard for the twsq alphas' signal data (alphas/_binance_data.py)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alphas"))
from _binance_data import rows_before  # noqa: E402


def test_rows_before_excludes_the_rebalance_day():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    df = pd.DataFrame({"x": range(5)}, index=idx)
    # A rebalance at the start of Jan 4 may only see Jan 1-3; Jan 4 has not closed.
    assert list(rows_before(df, pd.Timestamp("2025-01-04")).index) == list(idx[:3])
    # Same answer later in the day and with a timezone-aware timestamp.
    assert list(rows_before(df, pd.Timestamp("2025-01-04 13:30", tz="UTC")).index) == list(idx[:3])


if __name__ == "__main__":
    for fn in [test_rows_before_excludes_the_rebalance_day]:
        fn()
        print(f"ok {fn.__name__}")
