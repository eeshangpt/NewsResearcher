"""Unit tests for `agents/claim_extraction_agent.py` (Task 3.2.1b).

Mocks `get_chat_model` and `get_langfuse_callback_handler` entirely (no real
API key needed), matching `tests/test_subtopic_agent.py::propose_candidates`'s
established pattern -- confirms the prompt template renders with
`article_text`, the chat model is fetched for the `"claim_extraction"` stage
and wrapped with `with_structured_output(ClaimList)`, the Langfuse callback +
`trace_metadata` are attached to the invocation config, and the model's
structured-output result is passed through unchanged.

Langfuse trace *content* isn't asserted here -- that requires a real
Langfuse instance and API key (see CLAUDE.md's `tests/live/` convention);
confirming the trace is actually visible/inspectable in Langfuse per Story
3.2's acceptance line is a manual/live-smoke-test concern, not a mocked unit
assertion that would prove nothing about the real integration.
"""

from unittest.mock import MagicMock, patch

from newsresearch.agents.claim_extraction_agent import extract_claims
from newsresearch.llm.schemas import Claim, ClaimList

MOCK_CLAIM_LIST = ClaimList(
    claims=[
        Claim(
            claim_text="The Federal Reserve raised interest rates by 0.25 points.",
            subject="the Federal Reserve",
            attributed_source="the article's own reporting",
            attribution_type="reported",
            certainty="confirmed",
        ),
        Claim(
            claim_text="Some analysts expect further hikes this year.",
            subject="analysts",
            attributed_source="unnamed market analysts",
            attribution_type="alleged",
            certainty="developing",
        ),
    ]
)


class _RecordingStructuredModel:
    """See `tests/test_subtopic_agent.py`'s identical helper: an explicit
    `config` parameter makes `RunnableLambda` actually forward the run config
    through, unlike a bare `MagicMock`.
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
    return _RecordingStructuredModel(MOCK_CLAIM_LIST)


@patch("newsresearch.agents.claim_extraction_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.claim_extraction_agent.get_chat_model")
def test_extract_claims_returns_structured_output(mock_get_chat_model, mock_get_langfuse):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_get_langfuse.return_value = MagicMock()

    result = extract_claims("The Fed raised rates today.", run_id="run-1")

    assert result == MOCK_CLAIM_LIST
    mock_get_chat_model.assert_called_once_with("claim_extraction")
    mock_chat_model.with_structured_output.assert_called_once_with(ClaimList)


@patch("newsresearch.agents.claim_extraction_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.claim_extraction_agent.get_chat_model")
def test_extract_claims_fills_prompt_variable(mock_get_chat_model, mock_get_langfuse):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_get_langfuse.return_value = MagicMock()

    extract_claims("Distinctive article body text about widgets.", run_id="run-2")

    call_args, _ = mock_structured_model.call_args
    rendered_text = call_args[0].to_string()
    assert "Distinctive article body text about widgets." in rendered_text


@patch("newsresearch.agents.claim_extraction_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.claim_extraction_agent.get_chat_model")
def test_extract_claims_attaches_langfuse_callback_and_trace_metadata(
    mock_get_chat_model, mock_get_langfuse
):
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_handler = MagicMock()
    mock_get_langfuse.return_value = mock_handler

    extract_claims("Some article text.", run_id="run-3")

    _, call_kwargs = mock_structured_model.call_args
    config = call_kwargs["config"]
    # LangChain normalizes the `callbacks` list into a `CallbackManager` by
    # the time it reaches this inner step -- inspect `.handlers` rather than
    # the raw list passed to `chain.invoke`.
    assert mock_handler in config["callbacks"].handlers
    assert config["metadata"]["stage"] == "claim_extraction"
    assert config["metadata"]["run_id"] == "run-3"


@patch("newsresearch.agents.claim_extraction_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.claim_extraction_agent.get_chat_model")
def test_extract_claims_merges_ambient_config(mock_get_chat_model, mock_get_langfuse):
    """An ambient `config` (e.g. forwarded from a LangGraph node) keeps its
    own callbacks alongside the Langfuse handler this call attaches --
    proves `merge_configs` is used rather than a config being overwritten.
    """
    mock_structured_model = _make_mock_structured_model()
    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = mock_structured_model
    mock_get_chat_model.return_value = mock_chat_model
    mock_get_langfuse.return_value = MagicMock()

    ambient_callback = MagicMock()
    ambient_config = {"callbacks": [ambient_callback]}

    extract_claims("Some article text.", config=ambient_config)

    _, call_kwargs = mock_structured_model.call_args
    config = call_kwargs["config"]
    assert ambient_callback in config["callbacks"].handlers
