"""Gate 2 per-subtopic-branch-independent interrupt/resume (Task 2.6.2).

Real fan-out (Task 2.4.1, `Send`-based) doesn't exist yet, so distinct
fanned-out branches are simulated the way LangGraph itself would keep them
independent: distinct `thread_id`s under a standalone `StateGraph
(SubtopicState)` graph, compiled with the real `PostgresSaver` via
`build.py`'s `build_checkpointer()` (reused, not reimplemented).
"""

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from testcontainers.postgres import PostgresContainer

from newsresearch.graph.build import build_checkpointer
from newsresearch.graph.nodes.gate2 import gate2_node
from newsresearch.graph.state import SubtopicState


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture(scope="module")
def gate2_graph(postgres_url):
    builder = StateGraph(SubtopicState)
    builder.add_node("gate2", gate2_node)
    builder.add_edge(START, "gate2")
    builder.add_edge("gate2", END)
    checkpointer = build_checkpointer(postgres_url)
    return builder.compile(checkpointer=checkpointer)


def _subtopic_state(subtopic_id: str, label: str, sample_headline: str):
    return {
        "run_id": "run-1",
        "subtopic_id": subtopic_id,
        "label": label,
        "cluster_report": {
            "cluster_sizes": [5, 3],
            "sample_headlines": [sample_headline],
            "source_spread": {"reuters.com": 4, "apnews.com": 4},
        },
    }


def test_gate2_interrupts_with_stubbed_cluster_report(gate2_graph):
    config = {"configurable": {"thread_id": "run-1:sub-1"}}
    state = _subtopic_state("sub-1", "eu ai act", "EU passes AI Act")

    result = gate2_graph.invoke(state, config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["cluster_report"] == state["cluster_report"]


def test_gate2_blocks_each_subtopic_branch_independently(gate2_graph):
    config_a = {"configurable": {"thread_id": "run-1:sub-1"}}
    config_b = {"configurable": {"thread_id": "run-1:sub-2"}}

    state_a = _subtopic_state("sub-1", "eu ai act", "EU passes AI Act")
    state_b = _subtopic_state("sub-2", "us executive order", "White House signs order")

    gate2_graph.invoke(state_a, config=config_a)
    gate2_graph.invoke(state_b, config=config_b)

    # Both branches paused independently -- branch A's pending interrupt
    # did not prevent branch B from also being invoked and paused.
    assert gate2_graph.get_state(config_a).next == ("gate2",)
    assert gate2_graph.get_state(config_b).next == ("gate2",)

    # Resuming A must not affect B.
    result_a = gate2_graph.invoke(Command(resume={"action": "continue"}), config=config_a)
    assert result_a["subtopic_id"] == "sub-1"
    assert gate2_graph.get_state(config_a).next == ()
    assert gate2_graph.get_state(config_b).next == ("gate2",)

    # B resumes independently afterward.
    result_b = gate2_graph.invoke(Command(resume={"action": "continue"}), config=config_b)
    assert result_b["subtopic_id"] == "sub-2"
    assert gate2_graph.get_state(config_b).next == ()


def test_gate2_unrecognized_resume_action_raises(gate2_graph):
    config = {"configurable": {"thread_id": "run-1:sub-3"}}
    state = _subtopic_state("sub-3", "china ai regs", "China issues AI rules")

    gate2_graph.invoke(state, config=config)
    with pytest.raises(ValueError, match="unrecognized resume action"):
        gate2_graph.invoke(Command(resume={"action": "reject"}), config=config)
