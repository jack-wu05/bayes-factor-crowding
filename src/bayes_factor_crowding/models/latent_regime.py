"""Regime model configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatentRegimeModelConfig:
    regimes: tuple[str, ...] = ("normal", "crowded", "stress")
    likelihood: str = "student_t"
    sticky_probability: float = 0.94
    regime_quantiles: tuple[float, ...] = (0.60, 0.85)
