from langfuse.langchain import CallbackHandler

from newsresearch.config import Settings
from newsresearch.observability.langfuse_setup import (
    get_langfuse_callback_handler,
    trace_metadata,
)


def test_get_langfuse_callback_handler_returns_a_callback_handler(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-dummy")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    settings = Settings()

    handler = get_langfuse_callback_handler(settings)

    assert isinstance(handler, CallbackHandler)


def test_trace_metadata_tags_run_id_only():
    metadata = trace_metadata("run-1")

    assert metadata["run_id"] == "run-1"
    assert "subtopic_id" not in metadata
    assert metadata["langfuse_tags"] == ["run_id:run-1"]


def test_trace_metadata_tags_run_id_and_subtopic_id():
    metadata = trace_metadata("run-1", subtopic_id="sub-2")

    assert metadata["run_id"] == "run-1"
    assert metadata["subtopic_id"] == "sub-2"
    assert metadata["langfuse_tags"] == ["run_id:run-1", "subtopic_id:sub-2"]
