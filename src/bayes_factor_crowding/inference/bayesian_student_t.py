"""Bayesian Student-t latent-regime inference.

This is the first concrete statistical model in the repo. It uses a
Normal-inverse-gamma posterior predictive distribution for each regime and a
sticky Markov transition matrix to infer current regime probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi, sqrt
from typing import Mapping

import numpy as np
import pandas as pd

from bayes_factor_crowding.inference.base import RegimeInferenceEngine
from bayes_factor_crowding.models.latent_regime import LatentRegimeModelConfig
from bayes_factor_crowding.models.priors import PriorConfig
from bayes_factor_crowding.schema import ObservationWindow, PosteriorState


@dataclass(frozen=True)
class RegimePredictiveParams:
    regime: str
    location: float
    scale: float
    degrees_of_freedom: float
    assigned_observations: int


class BayesianStudentTRegimeInference(RegimeInferenceEngine):
    """Infer normal/crowded/stress probabilities from portfolio returns.

    The implementation is deliberately lightweight:
    - collapse factor returns into a portfolio return using static weights;
    - assign historical observations to ordered volatility buckets;
    - fit a Bayesian Student-t posterior predictive distribution per bucket;
    - run a sticky HMM filter and return the final-day regime posterior.

    It is a practical first model and a clean stepping stone toward the custom
    mixed discrete-continuous sampler.
    """

    def __init__(
        self,
        model_config: LatentRegimeModelConfig | None = None,
        prior_config: PriorConfig | None = None,
        factor_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.model_config = model_config or LatentRegimeModelConfig()
        self.prior_config = prior_config or PriorConfig()
        self.factor_weights = dict(factor_weights) if factor_weights is not None else None
        self._validate_config()

    def fit_predict(self, window: ObservationWindow) -> PosteriorState:
        returns = self._portfolio_returns(window.factor_returns)
        values = returns.to_numpy(dtype=float)
        if len(values) < 30:
            raise ValueError("BayesianStudentTRegimeInference needs at least 30 observations.")

        params = self._fit_regime_predictives(values)
        log_emissions = np.column_stack(
            [
                _student_t_logpdf(values, param.location, param.scale, param.degrees_of_freedom)
                for param in params
            ]
        )
        transition = self._transition_matrix()
        initial = self._initial_probabilities()
        filtered, log_likelihood = _forward_filter(log_emissions, transition, initial)

        probabilities = pd.Series(
            filtered[-1],
            index=self.model_config.regimes,
            name="probability",
        )
        predictive_summary = {
            param.regime: {
                "location": param.location,
                "scale": param.scale,
                "degrees_of_freedom": param.degrees_of_freedom,
                "assigned_observations": param.assigned_observations,
            }
            for param in params
        }
        return PosteriorState(
            as_of_date=window.end_date,
            regime_probabilities=probabilities,
            diagnostics={
                "engine": "bayesian_student_t_hmm_filter",
                "log_likelihood": log_likelihood,
                "latest_portfolio_return": float(values[-1]),
                "transition_matrix": pd.DataFrame(
                    transition,
                    index=self.model_config.regimes,
                    columns=self.model_config.regimes,
                ),
                "predictive_parameters": predictive_summary,
            },
            parameter_samples=predictive_summary,
        )

    def _portfolio_returns(self, factor_returns: pd.DataFrame) -> pd.Series:
        if self.factor_weights is None:
            return factor_returns.mean(axis=1).rename("portfolio_return")

        weights = pd.Series(self.factor_weights, dtype=float)
        missing = set(weights.index) - set(factor_returns.columns)
        if missing:
            raise ValueError(f"Missing factor return columns for weights: {sorted(missing)}")
        weights = weights.reindex(factor_returns.columns).fillna(0.0)
        weight_sum = weights.sum()
        if abs(weight_sum) < 1e-12:
            raise ValueError("Factor weights must have a nonzero net sum.")
        weights = weights / weight_sum
        return (factor_returns * weights).sum(axis=1).rename("portfolio_return")

    def _fit_regime_predictives(self, values: np.ndarray) -> list[RegimePredictiveParams]:
        centered_abs = np.abs(values - np.median(values))
        thresholds = np.quantile(centered_abs, self.model_config.regime_quantiles)
        assignments = np.searchsorted(thresholds, centered_abs, side="right")
        global_mean = float(np.mean(values))
        global_var = float(np.var(values, ddof=1))
        global_var = max(global_var, self.prior_config.prior_volatility_floor**2)

        params: list[RegimePredictiveParams] = []
        for regime_index, regime in enumerate(self.model_config.regimes):
            group = values[assignments == regime_index]
            params.append(self._fit_predictive_for_group(regime, group, global_mean, global_var))
        return params

    def _fit_predictive_for_group(
        self,
        regime: str,
        values: np.ndarray,
        prior_mean: float,
        prior_variance: float,
    ) -> RegimePredictiveParams:
        n = len(values)
        kappa_0 = self.prior_config.prior_observations
        alpha_0 = self.prior_config.prior_alpha
        beta_0 = prior_variance * (alpha_0 - 1.0)

        if n:
            sample_mean = float(np.mean(values))
            sum_squares = float(np.sum((values - sample_mean) ** 2))
        else:
            sample_mean = prior_mean
            sum_squares = 0.0

        kappa_n = kappa_0 + n
        alpha_n = alpha_0 + n / 2.0
        mean_gap = sample_mean - prior_mean
        beta_n = (
            beta_0
            + 0.5 * sum_squares
            + (kappa_0 * n * mean_gap**2) / (2.0 * kappa_n)
        )
        location = (kappa_0 * prior_mean + n * sample_mean) / kappa_n
        scale = sqrt(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n))
        scale = max(scale, self.prior_config.prior_volatility_floor)
        return RegimePredictiveParams(
            regime=regime,
            location=float(location),
            scale=float(scale),
            degrees_of_freedom=float(2.0 * alpha_n),
            assigned_observations=n,
        )

    def _transition_matrix(self) -> np.ndarray:
        regimes = self.model_config.regimes
        n = len(regimes)
        sticky = self.model_config.sticky_probability
        off_diagonal = (1.0 - sticky) / (n - 1)
        matrix = np.full((n, n), off_diagonal)
        np.fill_diagonal(matrix, sticky)
        return matrix

    def _initial_probabilities(self) -> np.ndarray:
        configured = np.asarray(self.prior_config.initial_regime_probabilities, dtype=float)
        n = len(self.model_config.regimes)
        if len(configured) != n:
            configured = np.ones(n, dtype=float)
        return configured / configured.sum()

    def _validate_config(self) -> None:
        regimes = self.model_config.regimes
        if len(regimes) < 2:
            raise ValueError("At least two regimes are required.")
        if len(self.model_config.regime_quantiles) != len(regimes) - 1:
            raise ValueError("regime_quantiles must have one fewer value than regimes.")
        if not 0.0 < self.model_config.sticky_probability < 1.0:
            raise ValueError("sticky_probability must be between 0 and 1.")
        if any(q <= 0.0 or q >= 1.0 for q in self.model_config.regime_quantiles):
            raise ValueError("regime_quantiles must be between 0 and 1.")
        if tuple(sorted(self.model_config.regime_quantiles)) != self.model_config.regime_quantiles:
            raise ValueError("regime_quantiles must be sorted ascending.")


def _student_t_logpdf(values: np.ndarray, location: float, scale: float, df: float) -> np.ndarray:
    standardized = (values - location) / scale
    normalizer = lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) - log(scale) - 0.5 * log(df * pi)
    kernel = -((df + 1.0) / 2.0) * np.log1p((standardized**2) / df)
    return normalizer + kernel


def _forward_filter(
    log_emissions: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_transition = np.log(transition)
    alpha = np.empty_like(log_emissions)
    log_likelihood = 0.0

    alpha[0] = np.log(initial) + log_emissions[0]
    normalizer = _logsumexp(alpha[0])
    alpha[0] -= normalizer
    log_likelihood += normalizer

    for t in range(1, len(log_emissions)):
        predicted = _logsumexp(alpha[t - 1][:, None] + log_transition, axis=0)
        alpha[t] = log_emissions[t] + predicted
        normalizer = _logsumexp(alpha[t])
        alpha[t] -= normalizer
        log_likelihood += normalizer

    return np.exp(alpha), float(log_likelihood)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    max_value = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - max_value), axis=axis, keepdims=True)
    result = max_value + np.log(summed)
    if axis is None:
        return float(np.squeeze(result))
    return np.squeeze(result, axis=axis)
