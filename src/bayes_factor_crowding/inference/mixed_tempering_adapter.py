"""Adapter boundary for the custom mixed-tempering sampler."""

from __future__ import annotations

from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.models.latent_regime import LatentRegimeModelConfig
from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


class MixedTemperingRegimeInference(RegimeInferenceEngine):
    """Planned integration point for tree-referenced parallel tempering."""

    def __init__(self, config: LatentRegimeModelConfig | None = None) -> None:
        self.config = config or LatentRegimeModelConfig()

    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        raise NotImplementedError(
            "Connect this adapter to the mixed-tempering sampler once the "
            "model log density and state representation are implemented."
        )
