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


def build_state_graph() -> StateGraph:
    """Assemble the full no-op node topology, uncompiled."""
    builder = StateGraph(GraphState)
    for name in NODE_ORDER:
        builder.add_node(name, _make_passthrough_node(name))

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
