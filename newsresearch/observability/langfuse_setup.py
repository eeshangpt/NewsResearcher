"""Langfuse `CallbackHandler` factory (Cross-Cutting Concerns: LLM-call-level
tracing/debugging, distinct from `mlflow_setup.py`'s pipeline-level runs).

Every agent's prompts/responses/latency get traced automatically once this
handler is attached at the top-level `graph.invoke(..., config={"callbacks":
[...]})` call, the same way `cost_callback.py`'s handler is. Traces are
tagged with `run_id` (+ `subtopic_id` once Phase 2 fans out per subtopic) so
a Langfuse trace maps back to a specific pipeline run -- self-hosted per
Story 0.2, reachable at `Settings.langfuse_host` (default
`http://localhost:3000`).
"""

from __future__ import annotations

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from newsresearch.config import Settings


def get_langfuse_callback_handler(settings: Settings | None = None) -> CallbackHandler:
    """Return a Langfuse `CallbackHandler` wired to `Settings`'s credentials.

    Explicitly initializes the underlying `Langfuse` client from
    `Settings.langfuse_public_key`/`langfuse_secret_key`/`langfuse_host`
    rather than relying on the SDK's own ambient-env-var lookup, per every
    other module's convention of reading exclusively from `Settings`.
    """
    settings = settings or Settings()
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)


def trace_metadata(run_id: str, subtopic_id: str | None = None) -> dict[str, object]:
    """Build the `config["metadata"]` payload that tags a Langfuse trace.

    Pass this alongside the callback handler, e.g.:

        graph.invoke(state, config={
            "callbacks": [get_langfuse_callback_handler()],
            "metadata": trace_metadata(run_id, subtopic_id),
        })

    `langfuse_tags` (a key Langfuse's callback handler specifically reads,
    see `langfuse.langchain.CallbackHandler`) makes `run_id`/`subtopic_id`
    filterable as trace tags in the Langfuse UI; the plain `run_id`/
    `subtopic_id` keys also stay visible in the trace's own metadata panel.
    """
    tags = [f"run_id:{run_id}"]
    metadata: dict[str, object] = {"run_id": run_id}

    if subtopic_id is not None:
        tags.append(f"subtopic_id:{subtopic_id}")
        metadata["subtopic_id"] = subtopic_id

    metadata["langfuse_tags"] = tags
    return metadata
