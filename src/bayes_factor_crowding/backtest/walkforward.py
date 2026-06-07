"""Leakage-free walk-forward evaluation loop."""

from __future__ import annotations

import pandas as pd

from bayes_factor_crowding.backtest.costs import linear_turnover_cost
from bayes_factor_crowding.backtest.metrics import summarize_returns
from bayes_factor_crowding.features.transforms import make_feature_frame
from bayes_factor_crowding.features.windows import build_observation_window
from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.schema import BacktestResult
from bayes_factor_crowding.signals.risk_overlay import PosteriorRiskOverlay


def run_walkforward_backtest(
    factor_returns: pd.DataFrame,
    engine: RegimeInferenceEngine,
    overlay: PosteriorRiskOverlay,
    lookback_days: int = 252 * 3,
    cost_bps: float = 1.0,
) -> BacktestResult:
    """Infer regimes through time and apply next-period risk multipliers."""

    features = make_feature_frame(factor_returns)
    test_dates = factor_returns.index[lookback_days:-1]
    previous_multiplier = 1.0
    decisions: list[dict[str, float | str | pd.Timestamp]] = []
    realized_returns: list[tuple[pd.Timestamp, float]] = []

    for as_of_date in test_dates:
        window = build_observation_window(factor_returns, features, as_of_date, lookback_days)
        posterior = engine.fit_predict(window)
        signal = overlay.generate(posterior)
        next_date = factor_returns.index[factor_returns.index.get_loc(as_of_date) + 1]
        base_return = float(factor_returns.loc[next_date].mean())
        cost = linear_turnover_cost(previous_multiplier, signal.risk_multiplier, cost_bps)
        strategy_return = signal.risk_multiplier * base_return - cost

        row = {
            "date": as_of_date,
            "next_date": next_date,
            "risk_multiplier": signal.risk_multiplier,
            "realized_return": strategy_return,
            "normal_probability": float(posterior.regime_probabilities.get("normal", 0.0)),
            "crowded_probability": float(posterior.regime_probabilities.get("crowded", 0.0)),
            "stress_probability": float(posterior.regime_probabilities.get("stress", 0.0)),
        }
        decisions.append(row)
        realized_returns.append((next_date, strategy_return))
        previous_multiplier = signal.risk_multiplier

    returns = pd.Series(
        [value for _, value in realized_returns],
        index=[date for date, _ in realized_returns],
        name="strategy_return",
    )
    return BacktestResult(
        decisions=pd.DataFrame(decisions).set_index("date"),
        returns=returns,
        metrics=summarize_returns(returns),
    )
