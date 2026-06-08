"""LLM-assisted explanations of posterior regime moves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from bayes_factor_crowding.schema import PosteriorState


@dataclass(frozen=True)
class MarketContextDocument:
    """Timestamped market context used by the summarizer."""

    published_at: pd.Timestamp
    source: str
    title: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class RegimeContextSummary:
    """LLM explanation and the documents it was allowed to use."""

    summary: str
    as_of_date: pd.Timestamp
    documents_used: int
    model: str


class OpenAIRegimeSummarizer:
    """Summarize regime shifts with the OpenAI Responses API."""

    def __init__(
        self,
        model: str = "gpt-5",
        client: Any | None = None,
        max_documents: int = 8,
    ) -> None:
        self.model = model
        self.client = client
        self.max_documents = max_documents

    def summarize(
        self,
        previous: PosteriorState,
        current: PosteriorState,
        documents: pd.DataFrame,
    ) -> RegimeContextSummary:
        valid_documents = _timestamp_valid_documents(documents, current.as_of_date).tail(self.max_documents)
        prompt = _build_prompt(previous, current, valid_documents)
        response_text = self._call_openai(prompt)
        return RegimeContextSummary(
            summary=response_text,
            as_of_date=current.as_of_date,
            documents_used=len(valid_documents),
            model=self.model,
        )

    def _call_openai(self, prompt: str) -> str:
        client = self.client or _default_openai_client()
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You summarize market context for a Bayesian factor-regime model. "
                        "Be concise, distinguish evidence from speculation, and avoid making "
                        "trading recommendations."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.output_text.strip()


def summarize_regime_move(
    previous: PosteriorState,
    current: PosteriorState,
    documents: pd.DataFrame,
    summarizer: OpenAIRegimeSummarizer | None = None,
) -> str:
    """Summarize regime changes using only timestamp-valid context documents."""

    engine = summarizer or OpenAIRegimeSummarizer()
    return engine.summarize(previous, current, documents).summary


def _default_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Install the optional LLM dependency with "
            "`python -m pip install -e '.[llm]'`."
        ) from exc

    return OpenAI()


def _timestamp_valid_documents(documents: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if documents.empty:
        return documents.copy()
    required = {"published_at", "source", "title", "text"}
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(f"Context documents are missing required columns: {sorted(missing)}")

    frame = documents.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"])
    return frame.loc[frame["published_at"] <= as_of_date].sort_values("published_at")


def _build_prompt(previous: PosteriorState, current: PosteriorState, documents: pd.DataFrame) -> str:
    previous_probs = _format_probabilities(previous)
    current_probs = _format_probabilities(current)
    document_block = _format_documents(documents)
    return f"""
As-of date: {current.as_of_date.date()}

Previous posterior regime probabilities:
{previous_probs}

Current posterior regime probabilities:
{current_probs}

Timestamp-valid macro, market, and news context:
{document_block}

Write a short explanation of what changed in regime probabilities and which context items may help explain the move. Mention uncertainty when the evidence is weak.
""".strip()


def _format_probabilities(state: PosteriorState) -> str:
    return "\n".join(
        f"- {regime}: {float(probability):.1%}"
        for regime, probability in state.regime_probabilities.items()
    )


def _format_documents(documents: pd.DataFrame) -> str:
    if documents.empty:
        return "No timestamp-valid context documents were provided."

    lines = []
    for row in documents.itertuples(index=False):
        url = f" ({row.url})" if hasattr(row, "url") and pd.notna(row.url) else ""
        text = str(row.text).replace("\n", " ").strip()
        lines.append(
            f"- {pd.Timestamp(row.published_at).date()} | {row.source} | {row.title}{url}: {text}"
        )
    return "\n".join(lines)
