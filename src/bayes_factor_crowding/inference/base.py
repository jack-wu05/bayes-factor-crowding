"""Stable inference interface used by backtests and experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


class RegimeInferenceEngine(ABC):
    """Any engine that maps an observation window to posterior beliefs."""

    @abstractmethod
    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        """Fit or update the model and infer the current regime posterior."""
