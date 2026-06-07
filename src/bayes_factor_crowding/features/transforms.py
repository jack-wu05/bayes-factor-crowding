"""Convert raw returns into model-ready features."""

from __future__ import annotations

import pandas as pd


def rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualized rolling volatility using only historical observations."""

    return returns.rolling(window=window, min_periods=window).std() * (252**0.5)


def make_feature_frame(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Initial feature set for the regime model."""

    vol = rolling_volatility(factor_returns).add_suffix("_vol_21d")
    drawdown_proxy = factor_returns.rolling(21, min_periods=21).sum().add_suffix("_ret_21d")
    return pd.concat([vol, drawdown_proxy], axis=1).dropna()
