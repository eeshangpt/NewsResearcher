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
