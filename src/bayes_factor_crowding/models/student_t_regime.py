"""Explicit Student-t latent-regime model target for mixed samplers."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np


@dataclass(frozen=True)
class StudentTRegimePriors:
    """Hyperparameters for the full mixed discrete-continuous model."""

    transition_alpha: np.ndarray
    initial_probabilities: np.ndarray
    inverse_gamma_shape: float
    inverse_gamma_scale: float
    normal_mean: float
    normal_kappa: float
    student_t_dof: float

    @classmethod
    def default(cls, n_regimes: int) -> "StudentTRegimePriors":
        transition_alpha = np.full((n_regimes, n_regimes), 2.0)
        np.fill_diagonal(transition_alpha, 20.0)
        initial = np.ones(n_regimes) / n_regimes
        return cls(
            transition_alpha=transition_alpha,
            initial_probabilities=initial,
            inverse_gamma_shape=3.0,
            inverse_gamma_scale=1e-4,
            normal_mean=0.0,
            normal_kappa=5.0,
            student_t_dof=7.0,
        )


@dataclass(frozen=True)
class StudentTRegimeState:
    """A complete point in the posterior state space."""

    regimes: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def log_posterior(
    observations: np.ndarray,
    state: StudentTRegimeState,
    priors: StudentTRegimePriors,
) -> float:
    """Evaluate the joint log posterior up to an additive constant.

    # A_k ~ Dirichlet(α_k)
    # z_1 ~ Categorical(π_0)
    # z_t | z_{t-1}, A ~ Categorical(A[z_{t-1}])
    # σ_k² ~ InverseGamma(a_0, b_0)
    # μ_k | σ_k² ~ Normal(m_0, σ_k² / κ_0)
    # y_t | z_t = k, μ_k, σ_k² ~ StudentT(ν, μ_k, σ_k)
    """

    y = np.asarray(observations, dtype=float)
    z = np.asarray(state.regimes, dtype=int)
    transition = np.asarray(state.transition_matrix, dtype=float)
    means = np.asarray(state.means, dtype=float)
    variances = np.asarray(state.variances, dtype=float)

    if not _state_is_valid(y, z, transition, means, variances, priors):
        return float("-inf")

    total = 0.0
    total += _log_transition_prior(transition, priors.transition_alpha)
    total += _log_categorical(priors.initial_probabilities, z[0])
    total += _log_regime_path(z, transition)

    for variance, mean in zip(variances, means):
        total += _log_inverse_gamma(variance, priors.inverse_gamma_shape, priors.inverse_gamma_scale)
        total += _log_normal(mean, priors.normal_mean, variance / priors.normal_kappa)

    sigma = np.sqrt(variances)
    for observation, regime in zip(y, z):
        total += _log_student_t(
            observation,
            priors.student_t_dof,
            means[regime],
            sigma[regime],
        )

    return float(total)


def make_initial_state(
    observations: np.ndarray,
    n_regimes: int = 3,
    sticky_probability: float = 0.90,
) -> StudentTRegimeState:
    """Create a simple valid starting point for a sampler."""

    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("observations must be a non-empty one-dimensional array.")
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if not 0.0 < sticky_probability < 1.0:
        raise ValueError("sticky_probability must be between 0 and 1.")

    centered_abs = np.abs(y - np.median(y))
    quantiles = np.linspace(0.0, 1.0, n_regimes + 1)[1:-1]
    thresholds = np.quantile(centered_abs, quantiles)
    regimes = np.searchsorted(thresholds, centered_abs, side="right")

    off_diagonal = (1.0 - sticky_probability) / (n_regimes - 1)
    transition = np.full((n_regimes, n_regimes), off_diagonal)
    np.fill_diagonal(transition, sticky_probability)

    global_mean = float(np.mean(y))
    global_variance = max(float(np.var(y, ddof=1)), 1e-8) if len(y) > 1 else 1e-8
    means = np.empty(n_regimes)
    variances = np.empty(n_regimes)
    for regime in range(n_regimes):
        group = y[regimes == regime]
        means[regime] = float(np.mean(group)) if len(group) else global_mean
        variances[regime] = max(float(np.var(group, ddof=1)), 1e-8) if len(group) > 1 else global_variance

    return StudentTRegimeState(
        regimes=regimes,
        transition_matrix=transition,
        means=means,
        variances=variances,
    )


def _state_is_valid(
    observations: np.ndarray,
    regimes: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    priors: StudentTRegimePriors,
) -> bool:
    n_regimes = len(means)
    if observations.ndim != 1 or regimes.ndim != 1:
        return False
    if len(observations) != len(regimes) or len(observations) == 0:
        return False
    if transition.shape != (n_regimes, n_regimes):
        return False
    if priors.transition_alpha.shape != transition.shape:
        return False
    if len(priors.initial_probabilities) != n_regimes:
        return False
    if len(variances) != n_regimes:
        return False
    if np.any(regimes < 0) or np.any(regimes >= n_regimes):
        return False
    if np.any(transition <= 0.0) or np.any(variances <= 0.0):
        return False
    if np.any(priors.transition_alpha <= 0.0) or np.any(priors.initial_probabilities <= 0.0):
        return False
    if priors.inverse_gamma_shape <= 0.0 or priors.inverse_gamma_scale <= 0.0:
        return False
    if priors.normal_kappa <= 0.0 or priors.student_t_dof <= 0.0:
        return False
    return bool(
        np.allclose(transition.sum(axis=1), 1.0)
        and np.isclose(priors.initial_probabilities.sum(), 1.0)
    )


def _log_transition_prior(transition: np.ndarray, alpha: np.ndarray) -> float:
    return float(sum(_log_dirichlet(row, row_alpha) for row, row_alpha in zip(transition, alpha)))


def _log_regime_path(regimes: np.ndarray, transition: np.ndarray) -> float:
    if len(regimes) <= 1:
        return 0.0
    previous = regimes[:-1]
    current = regimes[1:]
    return float(np.log(transition[previous, current]).sum())


def _log_dirichlet(values: np.ndarray, alpha: np.ndarray) -> float:
    normalizer = lgamma(float(alpha.sum())) - sum(lgamma(float(a)) for a in alpha)
    kernel = float(np.sum((alpha - 1.0) * np.log(values)))
    return normalizer + kernel


def _log_categorical(probabilities: np.ndarray, category: int) -> float:
    return float(log(float(probabilities[category])))


def _log_inverse_gamma(value: float, shape: float, scale: float) -> float:
    return shape * log(scale) - lgamma(shape) - (shape + 1.0) * log(value) - scale / value


def _log_normal(value: float, mean: float, variance: float) -> float:
    return -0.5 * (log(2.0 * pi * variance) + ((value - mean) ** 2) / variance)


def _log_student_t(value: float, dof: float, location: float, scale: float) -> float:
    standardized = (value - location) / scale
    normalizer = lgamma((dof + 1.0) / 2.0) - lgamma(dof / 2.0) - log(scale) - 0.5 * log(dof * pi)
    kernel = -((dof + 1.0) / 2.0) * log(1.0 + (standardized**2) / dof)
    return normalizer + kernel
