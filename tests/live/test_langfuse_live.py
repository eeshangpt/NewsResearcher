"""Real-API smoke test for `observability/langfuse_setup.py`.

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach). Requires a real Langfuse project's
API keys (LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY) against a reachable
LANGFUSE_HOST (self-hosted stack from Story 0.2, `docker compose up -d`).
Skipped automatically when those aren't configured, since CI/most local runs
won't have a provisioned Langfuse project.

Run explicitly with real keys, e.g.:

    LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-... \
        uv run pytest -m live tests/live/test_langfuse_live.py
"""

import os
import time

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from newsresearch.config import Settings
from newsresearch.observability.langfuse_setup import (
    get_langfuse_callback_handler,
    trace_metadata,
)

pytestmark = pytest.mark.live

_MISSING_CREDS = not (
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)


@pytest.mark.skipif(
    _MISSING_CREDS,
    reason="requires a real Langfuse project's LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY",
)
def test_stub_call_produces_a_trace_tagged_with_run_id_in_langfuse():
    settings = Settings()
    handler = get_langfuse_callback_handler(settings)

    chain = RunnableLambda(lambda x: x.upper())
    run_id = f"live-test-{int(time.time())}"
    chain.invoke(
        "hello live test",
        config={
            "callbacks": [handler],
            "metadata": trace_metadata(run_id, subtopic_id="sub-live"),
            "run_name": "live-test-chain",
        },
    )

    trace_id = handler.last_trace_id
    assert trace_id is not None

    from langfuse import get_client

    get_client().flush()
    time.sleep(2)  # give the async ingestion pipeline a moment to land the trace

    response = httpx.get(
        f"{settings.langfuse_host}/api/public/traces/{trace_id}",
        auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
        timeout=10,
    )
    response.raise_for_status()
    trace = response.json()

    assert f"run_id:{run_id}" in trace["tags"]
    assert "subtopic_id:sub-live" in trace["tags"]
    assert trace["metadata"]["run_id"] == run_id
