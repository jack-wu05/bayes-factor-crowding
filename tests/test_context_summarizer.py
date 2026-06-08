import pandas as pd

from bayes_factor_crowding.context.summarizer import OpenAIRegimeSummarizer, summarize_regime_move
from bayes_factor_crowding.schema import PosteriorState


class _FakeResponses:
    def __init__(self) -> None:
        self.prompt = None

    def create(self, model, input):
        self.prompt = input[-1]["content"]

        class Response:
            output_text = "Stress probability rose as risk assets sold off and rate concerns intensified."

        return Response()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


def _posterior(date: str, normal: float, crowded: float, stress: float) -> PosteriorState:
    return PosteriorState(
        as_of_date=pd.Timestamp(date),
        regime_probabilities=pd.Series(
            {
                "normal": normal,
                "crowded": crowded,
                "stress": stress,
            }
        ),
    )


def test_openai_summarizer_filters_future_documents() -> None:
    fake_client = _FakeClient()
    summarizer = OpenAIRegimeSummarizer(client=fake_client, model="test-model")
    documents = pd.DataFrame(
        [
            {
                "published_at": "2026-01-02",
                "source": "Fed",
                "title": "Minutes",
                "text": "Policy remained restrictive.",
            },
            {
                "published_at": "2026-01-10",
                "source": "News",
                "title": "Future article",
                "text": "This should not be visible.",
            },
        ]
    )

    result = summarizer.summarize(
        previous=_posterior("2026-01-01", 0.70, 0.20, 0.10),
        current=_posterior("2026-01-05", 0.20, 0.30, 0.50),
        documents=documents,
    )

    assert result.documents_used == 1
    assert result.model == "test-model"
    assert "Policy remained restrictive" in fake_client.responses.prompt
    assert "This should not be visible" not in fake_client.responses.prompt


def test_summarize_regime_move_returns_text() -> None:
    fake_client = _FakeClient()
    summarizer = OpenAIRegimeSummarizer(client=fake_client)
    documents = pd.DataFrame(
        [
            {
                "published_at": "2026-01-02",
                "source": "Market",
                "title": "Volatility rose",
                "text": "Equity volatility moved higher.",
            }
        ]
    )

    text = summarize_regime_move(
        previous=_posterior("2026-01-01", 0.80, 0.10, 0.10),
        current=_posterior("2026-01-03", 0.25, 0.25, 0.50),
        documents=documents,
        summarizer=summarizer,
    )

    assert "Stress probability rose" in text
