"""Data-quality checks that guard against accidental leakage or bad inputs."""

from __future__ import annotations

import pandas as pd


def assert_datetime_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Expected dates to be sorted ascending.")


def assert_no_missing_returns(frame: pd.DataFrame) -> None:
    if frame.isna().any().any():
        raise ValueError("Factor returns contain missing values.")
