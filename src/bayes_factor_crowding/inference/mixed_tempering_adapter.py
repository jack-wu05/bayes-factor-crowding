"""Adapter boundary for local mixed-tempering inference."""

from __future__ import annotations

from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.models.latent_regime import LatentRegimeModelConfig
from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


class MixedTemperingRegimeInference(RegimeInferenceEngine):
    """Placeholder for local sampler integration.

    The core sampler implementation is intentionally kept local and ignored by
    Git. Use this class as the public integration point once the local sampler
    wrapper is wired to the regime model state.
    """

    def __init__(self, config: LatentRegimeModelConfig | None = None) -> None:
        self.config = config or LatentRegimeModelConfig()

    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        _ = window
        raise NotImplementedError(
            "Local mixed-tempering inference is not wired yet. The sampler code "
            "can live locally under src/bayes_factor_crowding/samplers/, which "
            "is ignored by Git."
        )
