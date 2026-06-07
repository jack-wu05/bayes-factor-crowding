from bayes_factor_crowding.backtest.walkforward import run_walkforward_backtest
from bayes_factor_crowding.experiments.run_demo import make_synthetic_factor_returns
from bayes_factor_crowding.features.windows import build_observation_window
from bayes_factor_crowding.inference.bayesian_student_t import BayesianStudentTRegimeInference
from bayes_factor_crowding.signals.risk_overlay import PosteriorRiskOverlay


def test_walkforward_smoke() -> None:
    returns = make_synthetic_factor_returns(rows=320, factors=3)
    result = run_walkforward_backtest(
        factor_returns=returns,
        engine=BayesianStudentTRegimeInference(),
        overlay=PosteriorRiskOverlay(),
        lookback_days=252,
    )

    assert not result.decisions.empty
    assert not result.returns.empty
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown"} <= set(result.metrics)
    assert result.decisions["risk_multiplier"].between(0.0, 1.0).all()


def test_bayesian_student_t_regime_probabilities_are_normalized() -> None:
    returns = make_synthetic_factor_returns(rows=260, factors=3)
    window = build_observation_window(returns, returns, returns.index[-1], lookback_days=252)
    posterior = BayesianStudentTRegimeInference().fit_predict(window)

    assert set(posterior.regime_probabilities.index) == {"normal", "crowded", "stress"}
    assert abs(float(posterior.regime_probabilities.sum()) - 1.0) < 1e-9
    assert posterior.regime_probabilities.between(0.0, 1.0).all()
