"""Graph assembly (Phase 0 Task 0.5.2, fan-out Task 2.4.1).

Wires the full TRD section 3.1 node topology as trivial passthrough nodes:

    Subtopic -> Gate1 -> FanOut -> Sourcing -> Clustering -> Gate2
             -> Claims -> Summarize -> Bias -> Briefing -> Snapshot -> Timeline

`FanOut` is a real node (Task 2.4.1: `fan_out` -> `sourcing` -> `clustering`
-> `gate2` are all `langgraph.types.Send`-based conditional edges, one `Send`
per approved `GraphState.candidates` entry out of `fan_out`, and one relay
`Send` per active branch at each subsequent hop -- see `_make_relay_router`
-- each carrying a `SubtopicState`-shaped identity, `run_id`/`subtopic_id`/
`label`, into its own concurrent branch). `Sourcing`/`Gate2` remain
passthrough for now; they just record their own `(node_name, subtopic_id,
label)` into `GraphState.fan_trace` when running inside a fanned branch, so
fan-out mechanics are provable without any real node logic existing yet.
`Clustering` is real (PR #32 tech-lead-rejected rework): it runs Task 2.5.1's
`topical_clustering_agent` + Task 2.6.1's `build_gate2_report` exactly once
per branch, upstream of `gate2`'s `interrupt()` -- see `_make_clustering_node`
for why this can't live inside `gate2_node` itself (LangGraph replays a node
function from the top on every resume, which would double real sourcing
calls and risk overwriting a human-reviewed report with a re-run, possibly
different, one). When `candidates` is empty (e.g. Phase 0's own
no-op-topology test), `fan_out` falls back to a single ordinary edge into
`sourcing`, so the pre-Phase-2 no-candidates path is unaffected. Compiled
with `PostgresSaver` (not an in-memory checkpointer) so gate durability
across process restarts — the entire point of using Postgres here — actually
holds.
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
from langgraph.types import Send
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from newsresearch.agents.topical_clustering_agent import topical_clustering_agent
from newsresearch.config import Settings
from newsresearch.graph.state import GraphState
from newsresearch.reports.gate2_report import build_gate2_report

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

# The three passthrough nodes immediately downstream of `fan_out` -- entered
# once per `Send`-fanned branch (or once, plainly, on the no-candidates
# fallback path). They record their own visit into `fan_trace` instead of
# staying a bare no-op, so fan-out mechanics are provable per Task 2.4.1's
# acceptance criterion without any real per-subtopic logic (Story 2.5/2.6)
# existing yet.
FAN_OUT_TARGET_NODES: frozenset[str] = frozenset({"sourcing", "clustering", "gate2"})


def _make_passthrough_node(name: str):
    """Build a trivial passthrough node function named `name`.

    Real logic lands in later phases (`graph/nodes/` per-module split); this
    story only needs the topology and checkpointing to work.
    """

    def _node(state: GraphState) -> dict[str, Any]:
        return {}

    _node.__name__ = f"{name}_node"
    return _node


def _make_fan_out_target_node(name: str):
    """Passthrough node that also records `(name, subtopic_id, label)` into
    `GraphState.fan_trace` when it's running inside a `Send`-fanned branch
    (i.e. `subtopic_id` is present on its input state). On the plain,
    no-candidates fallback path `subtopic_id` is absent and this behaves
    exactly like `_make_passthrough_node`.

    Deliberately does *not* echo `subtopic_id`/`label` back as plain state
    fields: concurrent branches in the same superstep would then all write
    different values to the same non-reducer channel, which LangGraph
    rejects (`InvalidUpdateError`, "can receive only one value per step").
    `fan_trace` (an `operator.add`-reduced accumulator) is the one channel
    safe for concurrent per-branch writes, so it's also how the *next* hop's
    routing (`_make_relay_router`) rediscovers each branch's identity,
    instead of reading it back off plain state.
    """

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        subtopic_id = state.get("subtopic_id")
        if subtopic_id is None:
            return {}
        return {"fan_trace": [(name, subtopic_id, state.get("label"))]}

    _node.__name__ = f"{name}_node"
    return _node


def _make_clustering_node(
    lookback_days: int, *, pool: ConnectionPool | None = None, settings: Settings | None = None
):
    """Real `clustering` node (Task 2.5.1 + Task 2.6.1, PR #32 rework).

    Runs `topical_clustering_agent` (per-subtopic sourcing + coarse
    clustering) then `build_gate2_report` (zero-LLM aggregation) for the
    branch's own `subtopic_id`/`label`. `clustering` is one of
    `FAN_OUT_TARGET_NODES`, entered exactly once per `Send`-fanned branch --
    never replayed by a downstream Gate 2 `interrupt()`/resume, unlike code
    that ran *inside* `gate2_node` itself would be.

    Writes the report into the reducer-safe `cluster_reports` accumulator
    (not a plain field: concurrent branches racing in the same superstep
    can't share one non-reducer channel) alongside the usual `fan_trace`
    entry; `_make_relay_router`'s clustering->gate2 hop then folds the
    branch's own entry into that branch's outgoing `Send` payload as a plain
    `cluster_report` field, so `gate2_node` (`graph/nodes/gate2.py`) can read
    it straight off its own per-branch state with no recomputation.

    On the plain, no-candidates fallback path `subtopic_id` is absent and
    this behaves like a no-op passthrough, same as `_make_fan_out_target_node`.
    """

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        subtopic_id = state.get("subtopic_id")
        if subtopic_id is None:
            return {}
        label = state.get("label", "")
        clustering_result = topical_clustering_agent(
            subtopic_id, label, lookback_days, pool=pool, settings=settings
        )
        report = build_gate2_report(clustering_result)
        return {
            "fan_trace": [("clustering", subtopic_id, label)],
            "cluster_reports": [(subtopic_id, report)],
        }

    _node.__name__ = "clustering_node"
    return _node


def fan_out_router(state: GraphState) -> list[Send] | str:
    """Conditional edge out of `fan_out`: Task 2.4.1's real `Send`-based
    fan-out.

    One `Send("sourcing", ...)` per approved `GraphState.candidates` entry,
    each carrying a `SubtopicState`-shaped sub-state (`run_id`,
    `subtopic_id`, `label`) into its own concurrent branch. Falls back to a
    plain edge into `sourcing` when there are no candidates yet (e.g. before
    Gate 1 populates them, or Phase 0's original no-candidates topology
    test), so this doesn't dead-end runs that never reach Gate 1 approval.
    """
    candidates = state.get("candidates") or []
    if not candidates:
        return "sourcing"

    return [
        Send(
            "sourcing",
            {
                "run_id": state["run_id"],
                "subtopic_id": f"{state['run_id']}-sub{i}",
                "label": candidate["label"],
            },
        )
        for i, candidate in enumerate(candidates)
    ]


def _make_relay_router(source_name: str, target_name: str, *, carry_field: str | None = None):
    """Conditional edge between two `FAN_OUT_TARGET_NODES`, keeping a
    `Send`-fanned run's N branches distinct across multiple hops.

    Re-derives each branch's `(subtopic_id, label)` from the `fan_trace`
    entries `source_name` just wrote (reducer-safe, so this is the one place
    concurrent branches' identities survive a superstep) and re-`Send`s one
    message per branch into `target_name`. Falls back to a plain edge when
    there's nothing to relay (`source_name` never fanned -- the no-candidates
    path), so it's a no-op unless fan-out is actually happening.

    `carry_field`, when given, names a `(subtopic_id, value)`-tuple reducer
    field (e.g. `cluster_reports`) whose per-branch entry gets folded into
    that branch's outgoing `Send` payload as a plain `cluster_report` field
    -- this is how `clustering`'s real per-branch output reaches `gate2`'s
    own state without ever being written back to a shared, collision-prone
    channel.
    """

    def _router(state: GraphState) -> list[Send] | str:
        branches = [
            (subtopic_id, label)
            for name, subtopic_id, label in state.get("fan_trace", [])
            if name == source_name
        ]
        if not branches:
            return target_name

        carried = dict(state.get(carry_field, [])) if carry_field else {}
        sends = []
        for subtopic_id, label in branches:
            payload = {"run_id": state["run_id"], "subtopic_id": subtopic_id, "label": label}
            if carry_field:
                payload["cluster_report"] = carried.get(subtopic_id, {})
            sends.append(Send(target_name, payload))
        return sends

    return _router


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


def build_state_graph(
    *, lookback_days: int = 7, pool: ConnectionPool | None = None, settings: Settings | None = None
) -> StateGraph:
    """Assemble the full node topology, uncompiled.

    `lookback_days`/`pool`/`settings` are forwarded to the real `clustering`
    node (Task 2.5.1's `topical_clustering_agent` call); `lookback_days`
    defaults to 7, matching `cli.py`'s own default -- no per-run tunable
    exists for this yet, and inventing one is out of this task's scope.
    """
    builder = StateGraph(GraphState)
    for name in NODE_ORDER:
        if name == "subtopic":
            node_fn = _make_subtopic_stub_node()
        elif name == "clustering":
            node_fn = _make_clustering_node(lookback_days, pool=pool, settings=settings)
        elif name in FAN_OUT_TARGET_NODES:
            node_fn = _make_fan_out_target_node(name)
        else:
            node_fn = _make_passthrough_node(name)
        builder.add_node(name, node_fn)

    builder.add_edge(START, NODE_ORDER[0])
    for upstream, downstream in zip(NODE_ORDER, NODE_ORDER[1:]):
        if upstream == "fan_out":
            # Real `Send`-based fan-out (Task 2.4.1): `downstream` here is
            # always "sourcing" per NODE_ORDER, the sole `path_map` target.
            builder.add_conditional_edges(upstream, fan_out_router, [downstream])
        elif upstream in FAN_OUT_TARGET_NODES and downstream in FAN_OUT_TARGET_NODES:
            # Relay hop between two fan-out branch nodes (sourcing ->
            # clustering, clustering -> gate2): keeps each branch's identity
            # distinct across the hop, see `_make_relay_router`. The
            # clustering->gate2 hop also carries the real `cluster_report`
            # `clustering` just computed forward into `gate2`'s own state.
            carry_field = "cluster_reports" if upstream == "clustering" else None
            builder.add_conditional_edges(
                upstream, _make_relay_router(upstream, downstream, carry_field=carry_field), [downstream]
            )
        else:
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


def build_graph(
    database_url: str | None = None,
    *,
    lookback_days: int = 7,
    pool: ConnectionPool | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Compile the pipeline graph with a durable `PostgresSaver`.

    `database_url` defaults to `Settings().database_url`
    (`NEWSRESEARCH_DATABASE_URL`) when not given explicitly. `lookback_days`/
    `pool`/`settings` are forwarded to `build_state_graph`'s real
    `clustering` node.
    """
    settings = settings or Settings()
    resolved_url = database_url or settings.database_url
    if not resolved_url:
        raise ValueError(
            "database_url is required to compile the graph (set "
            "NEWSRESEARCH_DATABASE_URL or pass database_url explicitly)"
        )

    checkpointer = build_checkpointer(resolved_url)
    return build_state_graph(lookback_days=lookback_days, pool=pool, settings=settings).compile(
        checkpointer=checkpointer
    )
