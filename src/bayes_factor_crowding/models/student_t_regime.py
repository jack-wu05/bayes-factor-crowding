"""Explicit multivariate Student-t latent-regime target for mixed samplers."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, pi

import numpy as np


@dataclass(frozen=True)
class StudentTRegimePriors:
    """Hyperparameters for the full mixed discrete-continuous model."""

    transition_alpha: np.ndarray
    initial_probabilities: np.ndarray
    inverse_wishart_df: float
    inverse_wishart_scale: np.ndarray
    normal_mean: np.ndarray
    normal_kappa: float
    student_t_dof: float
    enforce_covariance_ordering: bool = True

    @classmethod
    def default(
        cls,
        n_regimes: int,
        n_factors: int = 1,
        covariance_prior_strength: float = 20.0,
    ) -> "StudentTRegimePriors":
        target_covariance = 1e-4 * np.eye(n_factors)
        return cls.from_empirical_moments(
            n_regimes=n_regimes,
            mean=np.zeros(n_factors),
            covariance=target_covariance,
            covariance_prior_strength=covariance_prior_strength,
        )

    @classmethod
    def from_observations(
        cls,
        observations: np.ndarray,
        n_regimes: int,
        covariance_prior_strength: float = 20.0,
        transition_stickiness: float = 20.0,
        transition_off_diagonal: float = 2.0,
        normal_kappa: float = 5.0,
        student_t_dof: float = 7.0,
        enforce_covariance_ordering: bool = True,
    ) -> "StudentTRegimePriors":
        returns = _as_2d(observations)
        mean = np.mean(returns, axis=0)
        covariance = _regularized_covariance(returns, returns.shape[1])
        return cls.from_empirical_moments(
            n_regimes=n_regimes,
            mean=mean,
            covariance=covariance,
            covariance_prior_strength=covariance_prior_strength,
            transition_stickiness=transition_stickiness,
            transition_off_diagonal=transition_off_diagonal,
            normal_kappa=normal_kappa,
            student_t_dof=student_t_dof,
            enforce_covariance_ordering=enforce_covariance_ordering,
        )

    @classmethod
    def from_empirical_moments(
        cls,
        n_regimes: int,
        mean: np.ndarray,
        covariance: np.ndarray,
        covariance_prior_strength: float = 20.0,
        transition_stickiness: float = 20.0,
        transition_off_diagonal: float = 2.0,
        normal_kappa: float = 5.0,
        student_t_dof: float = 7.0,
        enforce_covariance_ordering: bool = True,
    ) -> "StudentTRegimePriors":
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        n_factors = len(mean)
        transition_alpha = np.full((n_regimes, n_regimes), transition_off_diagonal)
        np.fill_diagonal(transition_alpha, transition_stickiness)
        initial = _default_initial_probabilities(n_regimes)
        degrees_of_freedom = n_factors + 1.0 + covariance_prior_strength
        scale = covariance * covariance_prior_strength
        return cls(
            transition_alpha=transition_alpha,
            initial_probabilities=initial,
            inverse_wishart_df=degrees_of_freedom,
            inverse_wishart_scale=scale,
            normal_mean=mean,
            normal_kappa=normal_kappa,
            student_t_dof=student_t_dof,
            enforce_covariance_ordering=enforce_covariance_ordering,
        )


@dataclass(frozen=True)
class StudentTRegimeState:
    """A complete point in the posterior state space."""

    regimes: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


def log_posterior(
    observations: np.ndarray,
    state: StudentTRegimeState,
    priors: StudentTRegimePriors,
) -> float:
    """Evaluate the joint log posterior up to an additive constant.

    # For k = 1, ..., K:
    #   A_k ~ Dirichlet(α_k)
    #   Σ_k ~ InverseWishart(ν_0, Ψ_0)
    #   μ_k | Σ_k ~ MultivariateNormal(m_0, Σ_k / κ_0)
    #
    # Initial state:
    #   z_1 ~ Categorical(π_0)
    #
    # For t = 2, ..., T:
    #   z_t | z_{t-1}, A ~ Categorical(A[z_{t-1}])
    #
    # For t = 1, ..., T:
    #   r_t | z_t = k, μ_k, Σ_k
    #     ~ MultivariateStudentT(ν, μ_k, Σ_k)
    #
    # Constraint:
    #   trace(Σ_1) <= trace(Σ_2) <= ... <= trace(Σ_K)
    """

    returns = _as_2d(observations)
    regimes = np.asarray(state.regimes, dtype=int)
    transition = np.asarray(state.transition_matrix, dtype=float)
    means = np.asarray(state.means, dtype=float)
    covariances = np.asarray(state.covariances, dtype=float)

    if not _state_is_valid(returns, regimes, transition, means, covariances, priors):
        return float("-inf")

    total = 0.0
    total += _log_transition_prior(transition, priors.transition_alpha)
    total += _log_categorical(priors.initial_probabilities, regimes[0])
    total += _log_regime_path(regimes, transition)

    for mean, covariance in zip(means, covariances):
        total += _log_inverse_wishart(covariance, priors.inverse_wishart_df, priors.inverse_wishart_scale)
        total += _log_multivariate_normal(mean, priors.normal_mean, covariance / priors.normal_kappa)

    for observation, regime in zip(returns, regimes):
        total += _log_multivariate_student_t(
            observation,
            priors.student_t_dof,
            means[regime],
            covariances[regime],
        )

    return float(total)


def make_initial_state(
    observations: np.ndarray,
    n_regimes: int = 3,
    sticky_probability: float = 0.90,
) -> StudentTRegimeState:
    """Create a simple valid starting point for a sampler."""

    returns = _as_2d(observations)
    if len(returns) == 0:
        raise ValueError("observations must be non-empty.")
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if not 0.0 < sticky_probability < 1.0:
        raise ValueError("sticky_probability must be between 0 and 1.")

    n_factors = returns.shape[1]
    centered_norm = np.linalg.norm(returns - np.median(returns, axis=0), axis=1)
    quantiles = np.linspace(0.0, 1.0, n_regimes + 1)[1:-1]
    thresholds = np.quantile(centered_norm, quantiles)
    regimes = np.searchsorted(thresholds, centered_norm, side="right")

    off_diagonal = (1.0 - sticky_probability) / (n_regimes - 1)
    transition = np.full((n_regimes, n_regimes), off_diagonal)
    np.fill_diagonal(transition, sticky_probability)

    global_mean = np.mean(returns, axis=0)
    global_covariance = _regularized_covariance(returns, n_factors)
    means = np.empty((n_regimes, n_factors))
    covariances = np.empty((n_regimes, n_factors, n_factors))
    for regime in range(n_regimes):
        group = returns[regimes == regime]
        means[regime] = np.mean(group, axis=0) if len(group) else global_mean
        covariances[regime] = _regularized_covariance(group, n_factors) if len(group) > 1 else global_covariance

    return StudentTRegimeState(
        regimes=regimes,
        transition_matrix=transition,
        means=means,
        covariances=covariances,
    )


def _as_2d(observations: np.ndarray) -> np.ndarray:
    returns = np.asarray(observations, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]
    if returns.ndim != 2:
        raise ValueError("observations must be a one- or two-dimensional array.")
    return returns


def _regularized_covariance(values: np.ndarray, n_factors: int, jitter: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return jitter * np.eye(n_factors)
    covariance = np.cov(values, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.array([[float(covariance)]])
    return covariance + jitter * np.eye(n_factors)


def _state_is_valid(
    observations: np.ndarray,
    regimes: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    priors: StudentTRegimePriors,
) -> bool:
    n_observations, n_factors = observations.shape
    n_regimes = len(means)
    if regimes.ndim != 1 or len(regimes) != n_observations or n_observations == 0:
        return False
    if transition.shape != (n_regimes, n_regimes):
        return False
    if means.shape != (n_regimes, n_factors):
        return False
    if covariances.shape != (n_regimes, n_factors, n_factors):
        return False
    if priors.transition_alpha.shape != transition.shape:
        return False
    if len(priors.initial_probabilities) != n_regimes:
        return False
    if priors.normal_mean.shape != (n_factors,):
        return False
    if priors.inverse_wishart_scale.shape != (n_factors, n_factors):
        return False
    if np.any(regimes < 0) or np.any(regimes >= n_regimes):
        return False
    if np.any(transition <= 0.0):
        return False
    if np.any(priors.transition_alpha <= 0.0) or np.any(priors.initial_probabilities <= 0.0):
        return False
    if priors.inverse_wishart_df <= n_factors - 1:
        return False
    if priors.normal_kappa <= 0.0 or priors.student_t_dof <= 0.0:
        return False
    if not np.allclose(transition.sum(axis=1), 1.0):
        return False
    if not np.isclose(priors.initial_probabilities.sum(), 1.0):
        return False
    if not _is_positive_definite(priors.inverse_wishart_scale):
        return False
    if priors.enforce_covariance_ordering and not _covariances_are_ordered(covariances):
        return False
    return all(_is_positive_definite(covariance) for covariance in covariances)


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _covariances_are_ordered(covariances: np.ndarray) -> bool:
    traces = np.array([np.trace(covariance) for covariance in covariances])
    return bool(np.all(np.diff(traces) >= -1e-12))


def _default_initial_probabilities(n_regimes: int) -> np.ndarray:
    if n_regimes == 3:
        return np.array([0.80, 0.15, 0.05])
    if n_regimes == 2:
        return np.array([0.90, 0.10])
    weights = np.linspace(n_regimes, 1, n_regimes, dtype=float)
    return weights / weights.sum()


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


def _log_inverse_wishart(covariance: np.ndarray, degrees_of_freedom: float, scale: np.ndarray) -> float:
    dimension = covariance.shape[0]
    scale_logdet = _logdet_positive_definite(scale)
    covariance_logdet = _logdet_positive_definite(covariance)
    precision_trace = float(np.trace(scale @ np.linalg.inv(covariance)))
    return (
        0.5 * degrees_of_freedom * scale_logdet
        - 0.5 * degrees_of_freedom * dimension * log(2.0)
        - _log_multivariate_gamma(0.5 * degrees_of_freedom, dimension)
        - 0.5 * (degrees_of_freedom + dimension + 1.0) * covariance_logdet
        - 0.5 * precision_trace
    )


def _log_multivariate_normal(value: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    dimension = len(value)
    diff = value - mean
    solve = np.linalg.solve(covariance, diff)
    return -0.5 * (
        dimension * log(2.0 * pi)
        + _logdet_positive_definite(covariance)
        + float(diff @ solve)
    )


def _log_multivariate_student_t(value: np.ndarray, dof: float, location: np.ndarray, scale: np.ndarray) -> float:
    dimension = len(value)
    diff = value - location
    solve = np.linalg.solve(scale, diff)
    quadratic = float(diff @ solve)
    return (
        lgamma((dof + dimension) / 2.0)
        - lgamma(dof / 2.0)
        - 0.5 * dimension * log(dof * pi)
        - 0.5 * _logdet_positive_definite(scale)
        - 0.5 * (dof + dimension) * log(1.0 + quadratic / dof)
    )


def _log_multivariate_gamma(value: float, dimension: int) -> float:
    return (
        dimension * (dimension - 1.0) * log(pi) / 4.0
        + sum(lgamma(value + (1.0 - i) / 2.0) for i in range(1, dimension + 1))
    )


def _logdet_positive_definite(matrix: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        return float("nan")
    return float(logdet)
