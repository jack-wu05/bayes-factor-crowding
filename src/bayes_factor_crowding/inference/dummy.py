"""Deterministic placeholder inference for wiring tests and demos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


class VolatilityHeuristicInference(RegimeInferenceEngine):
    """Toy engine that maps recent volatility into regime probabilities."""

    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        recent_vol = window.factor_returns.tail(21).std().mean() * np.sqrt(252)
        stress = float(np.clip((recent_vol - 0.20) / 0.25, 0.0, 1.0))
        crowded = float(np.clip((recent_vol - 0.12) / 0.18, 0.0, 1.0)) * (1.0 - stress)
        normal = max(0.0, 1.0 - crowded - stress)
        probabilities = pd.Series(
            {"normal": normal, "crowded": crowded, "stress": stress},
            name="probability",
        )
        probabilities = probabilities / probabilities.sum()
        return PosteriorState(
            as_of_date=window.end_date,
            regime_probabilities=probabilities,
            diagnostics={"engine": "volatility_heuristic", "recent_vol": recent_vol},
        )
