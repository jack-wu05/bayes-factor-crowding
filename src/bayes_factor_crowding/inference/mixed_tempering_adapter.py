"""Adapter boundary for local mixed-tempering inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.models.latent_regime import LatentRegimeModelConfig
from bayes_factor_crowding.models.student_t_regime import (
    StudentTRegimePriors,
    StudentTRegimeState,
    log_posterior,
    make_initial_state,
)
from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


class MixedTemperingRegimeInference(RegimeInferenceEngine):
    """Run local sampler code against the multivariate Student-t regime target.

    The core sampler implementation is intentionally kept local and ignored by
    Git under ``src/bayes_factor_crowding/samplers/``.
    """

    def __init__(
        self,
        config: LatentRegimeModelConfig | None = None,
        priors: StudentTRegimePriors | None = None,
        num_chains: int = 4,
        num_tuning_rounds: int = 7,
        random_seed: int | None = None,
    ) -> None:
        self.config = config or LatentRegimeModelConfig()
        self.priors = priors
        self.num_chains = num_chains
        self.num_tuning_rounds = num_tuning_rounds
        self.random_seed = random_seed

    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        try:
            from bayes_factor_crowding.samplers import tree_PT_with_RWMH
        except ImportError as exc:
            raise RuntimeError(
                "Local sampler code was not found. Put the private sampler under "
                "src/bayes_factor_crowding/samplers/, which is ignored by Git."
            ) from exc

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        observations = window.factor_returns.to_numpy(dtype=float)
        n_observations, n_factors = observations.shape
        n_regimes = len(self.config.regimes)
        priors = self.priors or StudentTRegimePriors.from_observations(observations, n_regimes)

        initial_vectors = _make_initial_vectors(
            observations=observations,
            n_regimes=n_regimes,
            num_chains=self.num_chains,
            sticky_probability=self.config.sticky_probability,
        )
        supports = {i: list(range(n_regimes)) for i in range(n_observations)}
        disc_indices = list(range(n_observations))
        cont_indices = list(range(n_observations, len(initial_vectors[0])))

        def log_target(vector):
            state = _decode_state_vector(vector, n_observations, n_regimes, n_factors)
            return log_posterior(observations, state, priors)

        result = tree_PT_with_RWMH(
            initial_vectors,
            self.num_chains,
            self.num_tuning_rounds,
            log_target,
            supports,
            disc_indices,
            cont_indices,
        )
        samples = result["samples"]
        probabilities = _terminal_regime_probabilities(samples, n_observations, self.config.regimes)
        final_state = _decode_state_vector(samples[-1], n_observations, n_regimes, n_factors) if samples else None

        return PosteriorState(
            as_of_date=window.end_date,
            regime_probabilities=probabilities,
            diagnostics={
                "engine": "local_mixed_tree_variational_parallel_tempering",
                "num_samples": len(samples),
                "reject_rates": result.get("reject_rates"),
                "schedule": result.get("schedule"),
                "num_restarts": result.get("num_restarts"),
            },
            parameter_samples=_state_summary(final_state, self.config.regimes),
        )


def _make_initial_vectors(
    observations: np.ndarray,
    n_regimes: int,
    num_chains: int,
    sticky_probability: float,
) -> list[np.ndarray]:
    base_state = make_initial_state(observations, n_regimes, sticky_probability)
    base_vector = _encode_state_vector(base_state)
    n_observations = len(observations)
    n_factors = observations.shape[1]
    covariance_offset = n_observations + n_regimes * n_regimes + n_regimes * n_factors
    initial_vectors = []

    for _ in range(num_chains):
        vector = base_vector.copy()
        vector[:n_observations] = np.random.randint(0, n_regimes, size=n_observations)
        vector[n_observations:covariance_offset] += np.random.normal(
            0.0,
            0.03,
            size=covariance_offset - n_observations,
        )
        initial_vectors.append(vector)

    return initial_vectors


def _encode_state_vector(state: StudentTRegimeState) -> np.ndarray:
    transition_logits = np.log(np.clip(state.transition_matrix, 1e-12, 1.0)).ravel()
    cholesky_params = np.concatenate([_cholesky_to_unconstrained(covariance) for covariance in state.covariances])
    return np.r_[
        state.regimes.astype(float),
        transition_logits,
        state.means.ravel(),
        cholesky_params,
    ]


def _decode_state_vector(
    vector: np.ndarray,
    n_observations: int,
    n_regimes: int,
    n_factors: int,
) -> StudentTRegimeState:
    vector = np.asarray(vector, dtype=float)
    regimes = vector[:n_observations].astype(int)
    offset = n_observations

    transition_logits = vector[offset : offset + n_regimes * n_regimes].reshape(n_regimes, n_regimes)
    offset += n_regimes * n_regimes

    means = vector[offset : offset + n_regimes * n_factors].reshape(n_regimes, n_factors)
    offset += n_regimes * n_factors

    chol_size = n_factors * (n_factors + 1) // 2
    covariances = np.empty((n_regimes, n_factors, n_factors))
    for regime in range(n_regimes):
        params = vector[offset : offset + chol_size]
        offset += chol_size
        cholesky = _unconstrained_to_cholesky(params, n_factors)
        covariances[regime] = cholesky @ cholesky.T

    return StudentTRegimeState(
        regimes=regimes,
        transition_matrix=_row_softmax(transition_logits),
        means=means,
        covariances=covariances,
    )


def _row_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _cholesky_to_unconstrained(covariance: np.ndarray) -> np.ndarray:
    cholesky = np.linalg.cholesky(covariance)
    params = []
    for row in range(cholesky.shape[0]):
        for col in range(row + 1):
            value = cholesky[row, col]
            params.append(np.log(value) if row == col else value)
    return np.asarray(params, dtype=float)


def _unconstrained_to_cholesky(params: np.ndarray, n_factors: int) -> np.ndarray:
    cholesky = np.zeros((n_factors, n_factors))
    idx = 0
    for row in range(n_factors):
        for col in range(row + 1):
            value = params[idx]
            cholesky[row, col] = np.exp(value) if row == col else value
            idx += 1
    return cholesky


def _terminal_regime_probabilities(samples, n_observations: int, regime_names: tuple[str, ...]) -> pd.Series:
    if not samples:
        probabilities = np.ones(len(regime_names)) / len(regime_names)
    else:
        terminal_regimes = np.asarray(samples)[:, n_observations - 1].astype(int)
        counts = np.bincount(terminal_regimes, minlength=len(regime_names))
        probabilities = counts / counts.sum()
    return pd.Series(probabilities, index=regime_names, name="probability")


def _state_summary(state: StudentTRegimeState | None, regime_names: tuple[str, ...]) -> dict[str, Any]:
    if state is None:
        return {}
    return {
        "transition_matrix": state.transition_matrix,
        "means": {
            regime: state.means[index]
            for index, regime in enumerate(regime_names)
        },
        "covariances": {
            regime: state.covariances[index]
            for index, regime in enumerate(regime_names)
        },
    }
