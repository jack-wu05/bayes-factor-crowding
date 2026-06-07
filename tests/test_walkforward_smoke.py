import numpy as np

from bayes_factor_crowding.backtest.walkforward import run_walkforward_backtest
from bayes_factor_crowding.experiments.run_demo import make_synthetic_factor_returns
from bayes_factor_crowding.inference.mixed_tempering_adapter import (
    _decode_state_vector,
    _encode_state_vector,
)
from bayes_factor_crowding.inference.dummy import VolatilityHeuristicInference
from bayes_factor_crowding.models.student_t_regime import (
    StudentTRegimePriors,
    log_posterior,
    make_initial_state,
)
from bayes_factor_crowding.signals.risk_overlay import PosteriorRiskOverlay


def test_walkforward_smoke() -> None:
    returns = make_synthetic_factor_returns(rows=320, factors=3)
    result = run_walkforward_backtest(
        factor_returns=returns,
        engine=VolatilityHeuristicInference(),
        overlay=PosteriorRiskOverlay(),
        lookback_days=252,
    )

    assert not result.decisions.empty
    assert not result.returns.empty
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown"} <= set(result.metrics)
    assert result.decisions["risk_multiplier"].between(0.0, 1.0).all()


def test_student_t_regime_log_posterior_is_finite() -> None:
    returns = make_synthetic_factor_returns(rows=260, factors=3)
    observations = returns.to_numpy()
    state = make_initial_state(observations, n_regimes=3)
    priors = StudentTRegimePriors.default(n_regimes=3, n_factors=3)

    value = log_posterior(observations, state, priors)

    assert np.isfinite(value)


def test_student_t_regime_log_posterior_rejects_invalid_state() -> None:
    observations = np.array([[0.01, 0.02], [-0.02, 0.01], [0.005, -0.01], [0.004, 0.002]])
    state = make_initial_state(observations, n_regimes=3)
    bad_state = state.__class__(
        regimes=np.array([0, 1, 2, 3]),
        transition_matrix=state.transition_matrix,
        means=state.means,
        covariances=state.covariances,
    )
    priors = StudentTRegimePriors.default(n_regimes=3, n_factors=2)

    assert log_posterior(observations, bad_state, priors) == float("-inf")


def test_multivariate_state_vector_roundtrip() -> None:
    observations = make_synthetic_factor_returns(rows=80, factors=3).to_numpy()
    state = make_initial_state(observations, n_regimes=3)
    vector = _encode_state_vector(state)
    decoded = _decode_state_vector(vector, n_observations=len(observations), n_regimes=3, n_factors=3)

    assert np.array_equal(decoded.regimes, state.regimes)
    assert np.allclose(decoded.transition_matrix, state.transition_matrix)
    assert np.allclose(decoded.means, state.means)
    assert np.allclose(decoded.covariances, state.covariances)


def test_covariance_ordering_identifies_regime_labels() -> None:
    observations = make_synthetic_factor_returns(rows=80, factors=2).to_numpy()
    state = make_initial_state(observations, n_regimes=3)
    priors = StudentTRegimePriors.from_observations(observations, n_regimes=3)
    unordered_state = state.__class__(
        regimes=state.regimes,
        transition_matrix=state.transition_matrix,
        means=state.means,
        covariances=state.covariances[::-1],
    )

    assert np.isfinite(log_posterior(observations, state, priors))
    assert log_posterior(observations, unordered_state, priors) == float("-inf")
