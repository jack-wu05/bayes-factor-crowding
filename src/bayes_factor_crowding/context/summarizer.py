"""Placeholder for AI-assisted explanations of posterior regime moves."""

from __future__ import annotations

import pandas as pd

from bayes_factor_crowding.schema import PosteriorState


def summarize_regime_move(
    previous: PosteriorState,
    current: PosteriorState,
    documents: pd.DataFrame,
) -> str:
    """Summarize regime changes using only timestamp-valid context documents."""

    _ = documents
    prev_stress = float(previous.regime_probabilities.get("stress", 0.0))
    curr_stress = float(current.regime_probabilities.get("stress", 0.0))
    return (
        f"Stress probability moved from {prev_stress:.1%} to {curr_stress:.1%}. "
        "Attach timestamp-aware retrieval here before using external context."
    )
