"""Graph assembly (Phase 0 Task 0.5.2).

Wires the full TRD section 3.1 node topology as trivial passthrough nodes:

    Subtopic -> Gate1 -> FanOut -> Sourcing -> Clustering -> Gate2
             -> Claims -> Summarize -> Bias -> Briefing -> Snapshot -> Timeline

`FanOut` stands in for the `Send`-based concurrent per-subtopic fan-out that
Phase 2 will implement for real; here it is just another passthrough node so
the topology and compilation/checkpointing machinery exist end-to-end before
any real node logic does. Compiled with `PostgresSaver` (not an in-memory
checkpointer) so gate durability across process restarts — the entire point
of using Postgres here — actually holds.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.graph.state import GraphState

# TRD 3.1 pipeline order, Subtopic through Timeline, as no-op node names.
NODE_ORDER: list[str] = [
    "subtopic",
    "gate1",
    "fan_out",
    "sourcing",
    "clustering",
    "gate2",
    "claims",
    "summarize",
    "bias",
    "briefing",
    "snapshot",
    "timeline",
]


def _make_passthrough_node(name: str):
    """Build a trivial passthrough node function named `name`.

    Real logic lands in later phases (`graph/nodes/` per-module split); this
    story only needs the topology and checkpointing to work.
    """

    def _node(state: GraphState) -> dict[str, Any]:
        return {}

    _node.__name__ = f"{name}_node"
    return _node


class _StubSubtopicChatModel(BaseChatModel):
    """Deterministic stand-in chat model for the `subtopic` node (Task 0.7.4).

    The real Subtopic Agent (prompt, schema, `get_chat_model("subtopic")`)
    lands in Phase 2 Task 2.2.1. Until then this is the minimal chat-model
    call needed so the observability stack attached at the top-level
    `graph.invoke()` call (cost callback, Langfuse, MLflow) has an actual LLM
    invocation to capture end-to-end -- a real `ChatOpenAI` call would
    require a live `OPENAI_API_KEY`, which Phase 0 must not depend on.
    """

    model_name: str = "stub-subtopic-model"

    @property
    def _llm_type(self) -> str:
        return "stub-subtopic-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="acknowledged",
            usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"model_name": self.model_name},
        )


def _make_subtopic_stub_node():
    """The `subtopic` node's Phase 0 stand-in.

    Still a no-op with respect to graph state -- the real subtopic-proposal
    logic isn't built yet -- but exercises the observability path with one
    stub chat-model call, per Task 0.7.4. Accepts `config` so the callbacks/
    metadata attached at the top-level `graph.invoke()` call propagate to
    this nested LLM call, the same pattern `cost_callback.py` documents.
    """

    def _node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        _StubSubtopicChatModel().invoke(f"Acknowledge topic: {state['topic']}", config=config)
        return {}

    _node.__name__ = "subtopic_node"
    return _node


def build_state_graph() -> StateGraph:
    """Assemble the full no-op node topology, uncompiled."""
    builder = StateGraph(GraphState)
    for name in NODE_ORDER:
        node_fn = _make_subtopic_stub_node() if name == "subtopic" else _make_passthrough_node(name)
        builder.add_node(name, node_fn)

    builder.add_edge(START, NODE_ORDER[0])
    for upstream, downstream in zip(NODE_ORDER, NODE_ORDER[1:]):
        builder.add_edge(upstream, downstream)
    builder.add_edge(NODE_ORDER[-1], END)

    return builder


def build_checkpointer(database_url: str) -> PostgresSaver:
    """Build a `PostgresSaver` checkpointer against `database_url`.

    Uses its own `ConnectionPool` (autocommit, dict-row) per
    `langgraph-checkpoint-postgres`'s own connection-setup convention —
    separate from `persistence.db.init_db`'s pool, since the checkpointer
    owns its own `checkpoints`/`checkpoint_writes`/etc. tables, distinct
    from the app schema in `persistence/schema.sql`. `setup()` must be
    called once before first use to create/migrate those tables.
    """
    pool = ConnectionPool(
        conninfo=database_url,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer


def build_graph(database_url: str | None = None) -> CompiledStateGraph:
    """Compile the no-op pipeline graph with a durable `PostgresSaver`.

    `database_url` defaults to `Settings().database_url`
    (`NEWSRESEARCH_DATABASE_URL`) when not given explicitly.
    """
    settings = Settings()
    resolved_url = database_url or settings.database_url
    if not resolved_url:
        raise ValueError(
            "database_url is required to compile the graph (set "
            "NEWSRESEARCH_DATABASE_URL or pass database_url explicitly)"
        )

    checkpointer = build_checkpointer(resolved_url)
    return build_state_graph().compile(checkpointer=checkpointer)
