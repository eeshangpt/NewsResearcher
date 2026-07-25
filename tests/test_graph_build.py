import numpy as np
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from testcontainers.postgres import PostgresContainer

from newsresearch.graph.build import (
    NODE_ORDER,
    _make_clustering_node,
    _make_fan_out_target_node,
    _make_relay_router,
    build_checkpointer,
    build_graph,
    fan_out_router,
)
from newsresearch.graph.nodes.gate2 import gate2_node
from newsresearch.graph.state import GraphState, SubtopicState

# Hermetic `testcontainers[postgres]` per Story 0.4's own precedent, so this
# test doesn't depend on the dev `docker compose up -d` stack being up (that
# dependency is exercised separately, manually, per Story 0.5's runtime note).


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


def test_graph_state_schemas_construct_with_all_named_fields():
    state = GraphState(
        topic="AI regulation",
        canonical_topic="ai regulation",
        run_id="run-1",
        subtopics=["eu ai act", "us executive order"],
        approved=False,
        candidates=[{"label": "eu ai act", "article_count": 12}],
        excess=[{"label": "us executive order", "article_count": 3}],
    )
    assert state["topic"] == "AI regulation"
    assert state["subtopics"] == ["eu ai act", "us executive order"]
    assert state["candidates"] == [{"label": "eu ai act", "article_count": 12}]
    assert state["excess"] == [{"label": "us executive order", "article_count": 3}]

    sub_state = SubtopicState(
        run_id="run-1",
        subtopic_id="sub-1",
        label="eu ai act",
        cluster_report={
            "cluster_sizes": [5, 3],
            "sample_headlines": ["EU passes AI Act"],
            "source_spread": {"reuters.com": 4, "apnews.com": 4},
        },
    )
    assert sub_state["subtopic_id"] == "sub-1"
    assert sub_state["cluster_report"]["cluster_sizes"] == [5, 3]


def test_graph_invoke_runs_every_node_and_writes_a_durable_checkpoint(postgres_url):
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "test topic",
        "canonical_topic": "test topic",
        "run_id": "test-run",
        "subtopics": [],
        "approved": False,
    }
    config = {"configurable": {"thread_id": "test"}}

    result = graph.invoke(initial_state, config=config)

    # No-op nodes return {} so the state should be unchanged coming out.
    assert result["topic"] == "test topic"

    # Confirm every node in the topology actually ran: each checkpoint
    # snapshot's `.next` names the node about to execute at that step, so
    # the full NODE_ORDER should appear across the history in sequence.
    history = list(graph.get_state_history(config))
    pending_nodes = {node for snapshot in history for node in snapshot.next}
    assert set(NODE_ORDER) <= pending_nodes

    # Verify a checkpoint row exists in Postgres for real, not just that
    # invoke() didn't raise.
    checkpoint_tuple = graph.checkpointer.get_tuple(config)
    assert checkpoint_tuple is not None
    assert checkpoint_tuple.config["configurable"]["thread_id"] == "test"

    with graph.checkpointer.conn.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id FROM checkpoints WHERE thread_id = %s", ("test",)
        ).fetchall()
    assert len(rows) > 0


def test_fan_out_sends_one_concurrent_branch_per_approved_candidate(postgres_url, monkeypatch):
    """Task 2.4.1: N approved candidates -> N `Send`-fanned branches, each
    carrying its own `subtopic_id` through `sourcing`/`clustering`/`gate2`.

    `fan_trace` (a `GraphState` accumulator every fanned branch writes
    `(node_name, subtopic_id)` into) is the observable proof that each of
    those three downstream nodes actually ran once per candidate -- with a
    distinct `subtopic_id` -- rather than once overall.

    `clustering` is real (PR #32 rework), so its underlying sourcing call is
    mocked here the same way `test_gate2.py`'s real-cluster-report test does
    -- this test only needs to prove fan-out mechanics, not re-exercise live
    GDELT/RSS.
    """
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )
    graph = build_graph(database_url=postgres_url)

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
        {"label": "china ai regulation", "article_count": 5},
    ]
    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "fanout-run",
        "subtopics": [],
        "approved": True,
        "candidates": candidates,
        "excess": [],
    }
    config = {"configurable": {"thread_id": "fanout-test"}}

    result = graph.invoke(initial_state, config=config)

    fan_trace = result["fan_trace"]
    subtopic_ids_seen = {subtopic_id for _, subtopic_id, _ in fan_trace}
    assert len(subtopic_ids_seen) == len(candidates)

    for node_name in ("sourcing", "clustering", "gate2"):
        node_subtopic_ids = {
            subtopic_id for name, subtopic_id, _ in fan_trace if name == node_name
        }
        assert node_subtopic_ids == subtopic_ids_seen


def test_clustering_runs_exactly_once_per_branch_across_gate2_interrupt_and_resume(
    postgres_url, monkeypatch
):
    """PR #32 rework, required re-check.

    PR #32 was tech-lead-rejected for calling `topical_clustering_agent`
    *inside* `gate2_node`, before `interrupt()` -- LangGraph replays a node
    function from the top on every resume, so that call ran twice per Gate 2
    pass. The fix moves the real work into `clustering` (one of
    `FAN_OUT_TARGET_NODES`, `Send`-relayed exactly once per branch, never
    replayed by a downstream interrupt/resume).

    Production `build_graph()` still wires `gate2` as a passthrough stub
    (real interrupt wiring into the compiled pipeline is a separate,
    not-yet-scheduled task per `EXECUTION_PLAN.md` Task 2.6.2's own
    scoping), so this test assembles `graph/build.py`'s real
    `fan_out`/`sourcing`/`clustering` wiring together with the real
    interrupting `gate2_node` directly, to prove the property PR #32 broke.
    """
    call_counts: dict[str, int] = {}

    def counting_sourcing_agent(keywords, lookback_days, **kwargs):
        (label,) = keywords
        call_counts[label] = call_counts.get(label, 0) + 1
        return []

    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent", counting_sourcing_agent
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    builder = StateGraph(GraphState)
    builder.add_node("fan_out", _make_fan_out_target_node("fan_out"))
    builder.add_node("sourcing", _make_fan_out_target_node("sourcing"))
    builder.add_node("clustering", _make_clustering_node(7))
    builder.add_node("gate2", gate2_node)
    builder.add_edge(START, "fan_out")
    builder.add_conditional_edges("fan_out", fan_out_router, ["sourcing"])
    builder.add_conditional_edges(
        "sourcing", _make_relay_router("sourcing", "clustering"), ["clustering"]
    )
    builder.add_conditional_edges(
        "clustering",
        _make_relay_router("clustering", "gate2", carry_field="cluster_reports"),
        ["gate2"],
    )
    builder.add_edge("gate2", END)

    graph = builder.compile(checkpointer=build_checkpointer(postgres_url))

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
    ]
    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "gate2-real-run",
        "subtopics": [],
        "approved": True,
        "candidates": candidates,
        "excess": [],
    }
    config = {"configurable": {"thread_id": "gate2-real-test"}}

    result = graph.invoke(initial_state, config=config)
    interrupts = result["__interrupt__"]
    assert len(interrupts) == len(candidates)

    # Each branch's own sourcing call ran exactly once building the
    # interrupt payload -- not yet twice, before any resume has happened.
    assert call_counts == {"eu ai act": 1, "us executive order": 1}

    resume_map = {i.id: {"action": "continue"} for i in interrupts}
    graph.invoke(Command(resume=resume_map), config=config)

    # The bug PR #32 shipped: resuming a Gate 2 interrupt replays
    # `gate2_node` from the top, which re-ran `topical_clustering_agent`
    # (and its underlying sourcing call) a second time. Proving the count
    # is still 1 per branch after resume is the actual regression check.
    assert call_counts == {"eu ai act": 1, "us executive order": 1}
    assert graph.get_state(config).next == ()
