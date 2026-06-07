"""Core objects passed between system layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ObservationWindow:
    """Leakage-free data available for a single model decision date."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    factor_returns: pd.DataFrame
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorState:
    """Posterior beliefs produced by a regime inference engine."""

    as_of_date: pd.Timestamp
    regime_probabilities: pd.Series
    diagnostics: dict[str, Any] = field(default_factory=dict)
    parameter_samples: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSignal:
    """Actionable risk decision derived from posterior beliefs."""

    as_of_date: pd.Timestamp
    risk_multiplier: float
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestResult:
    """Recorded walk-forward decisions and realized returns."""

    decisions: pd.DataFrame
    returns: pd.Series
    metrics: dict[str, float]
