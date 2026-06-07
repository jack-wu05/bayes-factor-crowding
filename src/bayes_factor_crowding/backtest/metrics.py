"""Performance metrics for walk-forward experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    annual_return = float(returns.mean() * 252)
    annual_vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = annual_return / annual_vol if annual_vol else 0.0
    cumulative = (1.0 + returns).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0
    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
