"""Unit tests for `agents/subtopic_agent.py` (Task 2.2.2's `broad_topic_fetch`
and Task 2.2.1b's `propose_candidates`).

`sourcing_agent` is Phase 1's already-approved, already-tested module (see
`tests/test_sourcing_agent.py`) -- `broad_topic_fetch`'s tests mock it out
entirely rather than re-testing its internals, and only confirm
`broad_topic_fetch`'s own thin-wrapper behaviour: it derives keywords from
the topic string, forwards `lookback_days`/`pool`/`settings` through, and
unwraps `ScoredArticle`s to plain article dicts.

`propose_candidates`'s tests mock `get_chat_model` and
`get_langfuse_callback_handler` entirely (no real API key needed) and
confirm: the prompt template renders with `topic`/`n_candidates`, the chat
model is fetched for the `"subtopic"` stage and wrapped with
`with_structured_output(SubtopicCandidateList)`, the Langfuse callback +
`trace_metadata` are attached to the invocation config, and the model's
structured-output result is passed through unchanged.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.agents.subtopic_agent import broad_topic_fetch, propose_candidates
from newsresearch.llm.schemas import SubtopicCandidate, SubtopicCandidateList

ARTICLE_A = {
    "title": "Solar capacity hits record high",
    "url": "https://example-a.com/solar-record",
    "domain": "example-a.com",
    "published_at": datetime(2026, 7, 20, tzinfo=timezone.utc),
    "source_type": "gdelt",
}
ARTICLE_B = {
    "title": "Wind policy shake-up announced",
    "url": "https://example-b.com/wind-policy",
    "domain": "example-b.com",
    "published_at": datetime(2026, 7, 21, tzinfo=timezone.utc),
    "source_type": "rss",
}

MOCK_SCORED_ARTICLES = [
    ScoredArticle(article=ARTICLE_A, reputation_score=0.9, reputation_tier="major"),
    ScoredArticle(article=ARTICLE_B, reputation_score=0.85, reputation_tier="wire"),
]


@patch("newsresearch.agents.subtopic_agent.sourcing_agent")
def test_broad_topic_fetch_derives_keywords_and_calls_sourcing_agent(mock_sourcing_agent):
    mock_sourcing_agent.return_value = MOCK_SCORED_ARTICLES

    result = broad_topic_fetch("renewable energy", lookback_days=7)

    mock_sourcing_agent.assert_called_once_with(
        ["renewable energy"], 7, pool=None, settings=None
    )
    assert result == [ARTICLE_A, ARTICLE_B]
    assert all(isinstance(item, dict) for item in result)
    assert not any(isinstance(item, ScoredArticle) for item in result)


@patch("newsresearch.agents.subtopic_agent.sourcing_agent")
def test_broad_topic_fetch_forwards_pool_and_settings(mock_sourcing_agent):
    mock_sourcing_agent.return_value = []
    sentinel_pool = object()
    sentinel_settings = object()

    result = broad_topic_fetch(
        "renewable energy", lookback_days=14, pool=sentinel_pool, settings=sentinel_settings
    )

    mock_sourcing_agent.assert_called_once_with(
        ["renewable energy"], 14, pool=sentinel_pool, settings=sentinel_settings
    )
    assert result == []


MOCK_CANDIDATE_LIST = SubtopicCandidateList(
    candidates=[
        SubtopicCandidate(
            label="Senate battleground-state races",
            rationale="A distinct institutional layer of the midterms, separate from House races.",
        ),
        SubtopicCandidate(
            label="Ballot measures on abortion access",
            rationale="A cross-cutting policy theme distinct from any single race.",
        ),
    ]
)


class _RecordingStructuredModel:
    """`prompt | model` coerces a plain callable `model` into a
    `RunnableLambda` via `coerce_to_runnable` -- an explicit `config`
    parameter (unlike a bare `MagicMock`, whose call signature isn't
    introspectable) makes `RunnableLambda` actually forward the run config
    through, matching how a real chat model's `.invoke(input, config=...)`
    receives it.
    """

    def __init__(self, return_value):
        self.return_value = return_value
        self.calls: list[tuple[object, dict | None]] = []

    def __call__(self, prompt_value, config=None):
        self.calls.append((prompt_value, config))
        return self.return_value

    @property
    def call_args(self):
        prompt_value, config = self.calls[-1]
        return (prompt_value,), {"config": config}


def _make_mock_structured_model():
    return _RecordingStructuredModel(MOCK_CANDIDATE_LIST)


@patch("newsresearch.agents.subtopic_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.subtopic_agent.get_chat_model")
def test_propose_candidates_returns_structured_output(mock_get_chat_model, mock_get_langfuse):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_get_langfuse.return_value = MagicMock()

    result = propose_candidates("2026 US midterm elections", n_candidates=8, run_id="run-1")

    assert result == MOCK_CANDIDATE_LIST
    mock_get_chat_model.assert_called_once_with("subtopic")
    mock_chat_model.with_structured_output.assert_called_once_with(SubtopicCandidateList)


@patch("newsresearch.agents.subtopic_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.subtopic_agent.get_chat_model")
def test_propose_candidates_fills_prompt_variables(mock_get_chat_model, mock_get_langfuse):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_get_langfuse.return_value = MagicMock()

    propose_candidates("quantum computing breakthroughs", n_candidates=5, run_id="run-2")

    call_args, call_kwargs = mock_structured_model.call_args
    rendered_input = call_args[0]
    # Confirm the topic/n_candidates actually made it into the rendered
    # prompt value the chain passes downstream.
    rendered_text = rendered_input.to_string()
    assert "quantum computing breakthroughs" in rendered_text
    assert "5" in rendered_text


@patch("newsresearch.agents.subtopic_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.subtopic_agent.get_chat_model")
def test_propose_candidates_attaches_langfuse_callback_and_trace_metadata(
    mock_get_chat_model, mock_get_langfuse
):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    sentinel_callback = MagicMock()
    mock_get_langfuse.return_value = sentinel_callback

    propose_candidates("commercial airline industry mergers", run_id="run-3")

    call_args, call_kwargs = mock_structured_model.call_args
    config = call_kwargs.get("config")
    assert config is not None
    # LangChain normalizes the `callbacks` list into a `CallbackManager` by
    # the time it reaches this inner step -- inspect `.handlers` rather than
    # the raw list passed to `chain.invoke`.
    assert sentinel_callback in config["callbacks"].handlers
    assert config["metadata"]["run_id"] == "run-3"
    assert config["metadata"]["stage"] == "subtopic"


def test_propose_candidates_defaults_n_candidates_to_eight():
    import inspect

    signature = inspect.signature(propose_candidates)
    assert signature.parameters["n_candidates"].default == 8
