"""Build rolling observation windows for walk-forward evaluation."""

from __future__ import annotations

import pandas as pd

from bayes_factor_crowding.schema import ObservationWindow


def build_observation_window(
    factor_returns: pd.DataFrame,
    features: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> ObservationWindow:
    """Slice data ending at as_of_date, excluding future observations."""

    returns_window = factor_returns.loc[:as_of_date].tail(lookback_days)
    feature_window = features.loc[:as_of_date].tail(lookback_days)
    if returns_window.empty:
        raise ValueError(f"No factor returns available through {as_of_date}.")
    return ObservationWindow(
        start_date=returns_window.index[0],
        end_date=returns_window.index[-1],
        factor_returns=returns_window,
        features=feature_window,
        metadata={"lookback_days": lookback_days},
    )
