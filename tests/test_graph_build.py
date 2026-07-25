import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.graph.build import NODE_ORDER, build_graph
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


def test_fan_out_sends_one_concurrent_branch_per_approved_candidate(postgres_url):
    """Task 2.4.1: N approved candidates -> N `Send`-fanned branches, each
    carrying its own `subtopic_id` through `sourcing`/`clustering`/`gate2`.

    `fan_trace` (a `GraphState` accumulator every fanned branch writes
    `(node_name, subtopic_id)` into) is the observable proof that each of
    those three downstream nodes actually ran once per candidate -- with a
    distinct `subtopic_id` -- rather than once overall.
    """
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
