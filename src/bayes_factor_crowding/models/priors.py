"""Prior configuration for Bayesian regime models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """Normal-inverse-gamma prior for regime return distributions."""

    prior_observations: float = 8.0
    prior_alpha: float = 3.0
    prior_volatility_floor: float = 1e-4
    initial_regime_probabilities: tuple[float, ...] = (0.80, 0.15, 0.05)
