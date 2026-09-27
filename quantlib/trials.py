"""Append-only registry of every strategy configuration evaluated.

The deflated Sharpe in notebook 06 needs the number of configurations tried,
including the ones that were dropped, so every research notebook calls
log_trial for each config it backtests. A config that is already in the
registry is not added again, so rerunning a notebook does not inflate the
count.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNS = ["timestamp", "family", "config", "dev_sharpe", "gate_sharpe", "note"]


def log_trial(
    path: str | Path,
    family: str,
    config: dict,
    dev_sharpe: float,
    gate_sharpe: float | None = None,
    note: str = "",
) -> bool:
    """Append one tested config; returns False if it was already logged."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.dumps(config, sort_keys=True)
    if path.exists():
        reg = pd.read_csv(path)
        if ((reg["family"] == family) & (reg["config"] == cfg)).any():
            return False
    row = pd.DataFrame(
        [[
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            family,
            cfg,
            dev_sharpe,
            gate_sharpe,
            note,
        ]],
        columns=COLUMNS,
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    return True


def load_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def n_trials(path: str | Path) -> int:
    return len(load_registry(path))
