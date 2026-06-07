"""Run a small synthetic walk-forward demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayes_factor_crowding.backtest.walkforward import run_walkforward_backtest
from bayes_factor_crowding.inference.dummy import VolatilityHeuristicInference
from bayes_factor_crowding.signals.risk_overlay import PosteriorRiskOverlay


def make_synthetic_factor_returns(rows: int = 900, factors: int = 4, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    calm_rows = int(rows * 0.40)
    crowded_rows = int(rows * 0.30)
    stress_rows = rows - calm_rows - crowded_rows
    regimes = np.r_[
        np.full(calm_rows, 0.006),
        np.full(crowded_rows, 0.014),
        np.full(stress_rows, 0.028),
    ]
    values = rng.normal(0.0002, regimes[:, None], size=(rows, factors))
    return pd.DataFrame(values, index=dates, columns=[f"factor_{i + 1}" for i in range(factors)])


def main() -> None:
    returns = make_synthetic_factor_returns()
    result = run_walkforward_backtest(
        factor_returns=returns,
        engine=VolatilityHeuristicInference(),
        overlay=PosteriorRiskOverlay(),
        lookback_days=252,
    )
    print(result.metrics)
    print(result.decisions.tail())


if __name__ == "__main__":
    main()
