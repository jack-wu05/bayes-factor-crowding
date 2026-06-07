"""Load market and factor return data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_factor_returns_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load factor returns from a CSV indexed by date."""

    frame = pd.read_csv(path, parse_dates=[date_column])
    frame = frame.set_index(date_column).sort_index()
    frame.index.name = "date"
    return frame.astype(float)
