"""Transaction-cost models."""

from __future__ import annotations


def linear_turnover_cost(previous_multiplier: float, next_multiplier: float, cost_bps: float) -> float:
    """Return cost as a decimal return drag."""

    turnover = abs(next_multiplier - previous_multiplier)
    return turnover * cost_bps / 10_000.0
