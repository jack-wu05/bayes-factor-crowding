"""Volatility-targeting baseline."""

from __future__ import annotations

import pandas as pd


def volatility_target_multiplier(
    returns: pd.Series,
    target_vol: float = 0.10,
    window: int = 63,
    max_leverage: float = 1.0,
) -> pd.Series:
    realized_vol = returns.rolling(window, min_periods=window).std() * (252**0.5)
    multiplier = target_vol / realized_vol
    return multiplier.clip(lower=0.0, upper=max_leverage).fillna(1.0)
