"""Unit tests for `agents/summarization_agent.py` (Task 3.6.1b).

Mocks `get_chat_model`, `get_langfuse_callback_handler`, and the persistence
read/write functions (no real DB, no real API key needed) -- matches
`tests/test_claim_extraction_agent.py`'s established pattern. Prompt/summary
*quality* is a manual spot-check per EXECUTION_PLAN.md Task 3.6.1's
acceptance line, not asserted here; this proves the wiring (prompt
construction from asserting claims only, model stage, config forwarding,
write-back) calls through correctly.
"""

from unittest.mock import MagicMock, patch

from newsresearch.agents.summarization_agent import summarize_cluster

MOCK_ROWS = [
    {
        "article_id": "a1",
        "relation": "asserts",
        "claim_text": "The model costs 80% less to run.",
        "sentiment_label": "neutral",
        "sentiment_compound": 0.0,
        "domain": "reuters.com",
    },
    {
        "article_id": "a2",
        "relation": "asserts",
        "claim_text": "The model is 80% cheaper to operate.",
        "sentiment_label": "neutral",
        "sentiment_compound": 0.0,
        "domain": "apnews.com",
    },
    {
        "article_id": "a3",
        "relation": "omits",
        "claim_text": None,
        "sentiment_label": None,
        "sentiment_compound": None,
        "domain": "bbc.com",
    },
]


class _RecordingChatModel:
    """See `tests/test_subtopic_agent.py`'s identical helper: an explicit
    `config` parameter makes `RunnableLambda` actually forward the run config
    through, unlike a bare `MagicMock`.
    """

    def __init__(self, content: str):
        self.content = content
        self.calls: list[tuple[object, dict | None]] = []

    def __call__(self, prompt_value, config=None):
        self.calls.append((prompt_value, config))
        return MagicMock(content=self.content)

    @property
    def call_args(self):
        prompt_value, config = self.calls[-1]
        return (prompt_value,), {"config": config}


@patch("newsresearch.agents.summarization_agent.write_cluster_summary")
@patch("newsresearch.agents.summarization_agent.read_cluster_article_relations")
@patch("newsresearch.agents.summarization_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.summarization_agent.get_chat_model")
def test_summarize_cluster_uses_summarization_stage_and_writes_back(
    mock_get_chat_model, mock_get_langfuse, mock_read_rows, mock_write_summary
):
    mock_model = _RecordingChatModel("Both sources report an 80% price cut.")
    mock_get_chat_model.return_value = mock_model
    mock_get_langfuse.return_value = MagicMock()
    mock_read_rows.return_value = MOCK_ROWS
    mock_pool = MagicMock()

    result = summarize_cluster(mock_pool, "sub1:0", run_id="run-1")

    assert result == "Both sources report an 80% price cut."
    mock_get_chat_model.assert_called_once_with("summarization")
    mock_write_summary.assert_called_once_with(mock_pool, "sub1:0", "Both sources report an 80% price cut.")


@patch("newsresearch.agents.summarization_agent.write_cluster_summary")
@patch("newsresearch.agents.summarization_agent.read_cluster_article_relations")
@patch("newsresearch.agents.summarization_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.summarization_agent.get_chat_model")
def test_summarize_cluster_excludes_omitting_rows_from_prompt(
    mock_get_chat_model, mock_get_langfuse, mock_read_rows, mock_write_summary
):
    mock_model = _RecordingChatModel("summary")
    mock_get_chat_model.return_value = mock_model
    mock_get_langfuse.return_value = MagicMock()
    mock_read_rows.return_value = MOCK_ROWS

    summarize_cluster(MagicMock(), "sub1:0")

    call_args, _ = mock_model.call_args
    rendered_text = call_args[0].to_string()
    assert "reuters.com: The model costs 80% less to run." in rendered_text
    assert "apnews.com: The model is 80% cheaper to operate." in rendered_text
    assert "bbc.com" not in rendered_text


@patch("newsresearch.agents.summarization_agent.write_cluster_summary")
@patch("newsresearch.agents.summarization_agent.read_cluster_article_relations")
@patch("newsresearch.agents.summarization_agent.get_langfuse_callback_handler")
@patch("newsresearch.agents.summarization_agent.get_chat_model")
def test_summarize_cluster_attaches_langfuse_and_merges_ambient_config(
    mock_get_chat_model, mock_get_langfuse, mock_read_rows, mock_write_summary
):
    mock_model = _RecordingChatModel("summary")
    mock_get_chat_model.return_value = mock_model
    mock_handler = MagicMock()
    mock_get_langfuse.return_value = mock_handler
    mock_read_rows.return_value = MOCK_ROWS

    ambient_callback = MagicMock()
    ambient_config = {"callbacks": [ambient_callback]}

    summarize_cluster(MagicMock(), "sub1:0", run_id="run-3", config=ambient_config)

    _, call_kwargs = mock_model.call_args
    config = call_kwargs["config"]
    assert mock_handler in config["callbacks"].handlers
    assert ambient_callback in config["callbacks"].handlers
    assert config["metadata"]["stage"] == "summarization"
    assert config["metadata"]["run_id"] == "run-3"