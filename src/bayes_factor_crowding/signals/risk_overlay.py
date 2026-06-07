"""Risk-overlay policies based on posterior regime probabilities."""

from __future__ import annotations

from bayes_factor_crowding.schema import PosteriorState, RiskSignal


class PosteriorRiskOverlay:
    """Convert regime probabilities into a scalar exposure multiplier."""

    def __init__(self, crowded_multiplier: float = 0.70, stress_multiplier: float = 0.25) -> None:
        self.crowded_multiplier = crowded_multiplier
        self.stress_multiplier = stress_multiplier

    def generate(self, posterior: PosteriorState) -> RiskSignal:
        probs = posterior.regime_probabilities
        normal = float(probs.get("normal", 0.0))
        crowded = float(probs.get("crowded", 0.0))
        stress = float(probs.get("stress", 0.0))
        multiplier = (
            normal
            + self.crowded_multiplier * crowded
            + self.stress_multiplier * stress
        )
        return RiskSignal(
            as_of_date=posterior.as_of_date,
            risk_multiplier=float(max(0.0, min(1.0, multiplier))),
            reason="Posterior-weighted risk exposure.",
            inputs={"regime_probabilities": probs.to_dict()},
        )
