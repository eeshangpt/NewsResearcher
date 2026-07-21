"""Per-LLM-call cost/token logging callback (Cross-Cutting Concerns, NFR-1).

Attach a `CostCallbackHandler` at every top-level `graph.invoke(state,
config={"callbacks": [...]})` call -- LangChain callbacks propagate
automatically through every nested LangChain call in every node, so no
per-agent instrumentation code is needed. Writes one row per LLM call to
`run_costs`, independent of whether Langfuse is reachable. Fails soft on any
DB-write error (same graceful-degradation pattern as the Google News RSS
backfill, NFR-3): observability must never crash a pipeline run.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# Placeholder USD-per-1K-token rates (input, output) for whichever exact
# model name `Settings.models.*` resolves to. This is bookkeeping
# arithmetic, not a model-choice decision -- extend/replace this table if
# pricing changes or new stage models are added. Unlisted models estimate
# at $0 rather than guessing.
_PRICING_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
}


def _estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = _PRICING_PER_1K_TOKENS.get(model or "", (0.0, 0.0))
    return (input_tokens / 1000) * input_rate + (output_tokens / 1000) * output_rate


def _extract_token_usage(response: LLMResult) -> tuple[int, int]:
    """Pull `(input_tokens, output_tokens)` out of an `LLMResult`.

    Prefers each generation's `usage_metadata` (the shape `ChatOpenAI` et al.
    populate on the returned `AIMessage`); falls back to the legacy
    `llm_output["token_usage"]` shape for providers that only populate that.
    """
    for generation_batch in response.generations:
        for generation in generation_batch:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message else None
            if usage:
                return usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0

    token_usage = (response.llm_output or {}).get("token_usage") or {}
    return token_usage.get("prompt_tokens", 0) or 0, token_usage.get("completion_tokens", 0) or 0


def _extract_model_name(response: LLMResult, fallback: str | None) -> str | None:
    model_name = (response.llm_output or {}).get("model_name")
    return model_name or fallback


class CostCallbackHandler(BaseCallbackHandler):
    """Writes one `run_costs` row per completed LLM call.

    `run_id`/`stage` are read from the `metadata` dict attached to the
    individual LLM invocation's config -- LangChain's callback machinery
    doesn't otherwise know about NewsResearch's own run/stage concepts, e.g.:

        model.invoke(prompt, config={
            "callbacks": [cost_callback],
            "metadata": {"run_id": run_id, "stage": "subtopic"},
        })
    """

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._pending: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, serialized, metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, serialized, metadata)

    def _start(
        self,
        run_id: UUID,
        serialized: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        model = (serialized or {}).get("kwargs", {}).get("model")
        self._pending[run_id] = {
            "started_at": time.monotonic(),
            "run_id": (metadata or {}).get("run_id"),
            "stage": (metadata or {}).get("stage"),
            "model": model,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        latency_ms = (time.monotonic() - pending["started_at"]) * 1000
        input_tokens, output_tokens = _extract_token_usage(response)
        model = _extract_model_name(response, pending["model"])
        estimated_cost = _estimate_cost(model, input_tokens, output_tokens)

        self._write_row(
            run_id=pending["run_id"],
            stage=pending["stage"],
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # No usable token/cost data on a failed call -- just drop the
        # pending entry, nothing to write.
        self._pending.pop(run_id, None)

    def _write_row(self, **fields: Any) -> None:
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO run_costs
                        (run_id, stage, model, input_tokens, output_tokens,
                         estimated_cost, latency_ms, created_at)
                    VALUES
                        (%(run_id)s, %(stage)s, %(model)s, %(input_tokens)s,
                         %(output_tokens)s, %(estimated_cost)s, %(latency_ms)s,
                         %(created_at)s)
                    """,
                    {**fields, "created_at": datetime.now(timezone.utc)},
                )
        except Exception:
            # NFR-3 soft-fail: observability must never crash a pipeline run,
            # regardless of the reason (DB down, FK violation, etc).
            logger.exception(
                "cost_callback: failed to write run_costs row; continuing"
            )
