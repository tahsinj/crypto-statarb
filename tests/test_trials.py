"""Append-only trial registry: every config ever evaluated gets one row."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantlib import trials  # noqa: E402


def test_log_and_count():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "reg.csv"
        assert trials.n_trials(p) == 0
        trials.log_trial(p, "seasonality", {"bucket": "us_hours", "lb": 30}, 0.8)
        trials.log_trial(p, "seasonality", {"lb": 60, "bucket": "weekend"}, -0.1, gate_sharpe=0.2)
        assert trials.n_trials(p) == 2
        df = trials.load_registry(p)
        assert list(df.columns) == ["timestamp", "family", "config", "dev_sharpe", "gate_sharpe", "note"]
        assert df["family"].tolist() == ["seasonality", "seasonality"]
        # config is JSON with sorted keys -> stable, parseable
        import json
        assert json.loads(df["config"].iloc[1]) == {"bucket": "weekend", "lb": 60}
        assert df["gate_sharpe"].iloc[1] == 0.2


def test_same_config_is_not_logged_twice():
    # Rerunning a notebook must not add trials: same family and config -> skipped.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "reg.csv"
        assert trials.log_trial(p, "carry", {"probe": "P1", "smooth": 7}, 1.0, 2.0)
        assert not trials.log_trial(p, "carry", {"smooth": 7, "probe": "P1"}, 1.0, 2.0)
        assert trials.log_trial(p, "carry", {"probe": "P1", "smooth": 30}, 0.3, 0.7)
        assert trials.log_trial(p, "orderflow", {"probe": "P1", "smooth": 7}, 0.1, 0.2)
        assert trials.n_trials(p) == 3


def test_load_missing_is_empty():
    with tempfile.TemporaryDirectory() as td:
        df = trials.load_registry(Path(td) / "nope.csv")
        assert len(df) == 0


if __name__ == "__main__":
    for fn in [test_log_and_count, test_same_config_is_not_logged_twice, test_load_missing_is_empty]:
        fn()
        print(f"ok {fn.__name__}")
