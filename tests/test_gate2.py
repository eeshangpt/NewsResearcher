"""Gate 2 per-subtopic-branch-independent interrupt/resume (Task 2.6.2).

Real fan-out (Task 2.4.1, `Send`-based) doesn't exist yet, so distinct
fanned-out branches are simulated the way LangGraph itself would keep them
independent: distinct `thread_id`s under a standalone `StateGraph
(SubtopicState)` graph, compiled with the real `PostgresSaver` via
`build.py`'s `build_checkpointer()` (reused, not reimplemented).
"""

import numpy as np
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.graph.build import build_checkpointer
from newsresearch.graph.nodes.gate2 import gate2_node, make_gate2_node, make_real_cluster_report
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


def _scored_article(title: str, domain: str) -> ScoredArticle:
    return ScoredArticle(
        article={"title": title, "url": f"https://{domain}/a", "domain": domain},
        reputation_score=0.9,
        reputation_tier="major",
    )


def test_gate2_real_cluster_report_runs_real_clustering_and_aggregation(postgres_url, monkeypatch):
    """Follow-up to Task 2.6.1: a Gate 2 node built with
    `make_real_cluster_report` presents genuinely computed `cluster_sizes`/
    `sample_headlines`/`source_spread` -- not the stub's pass-through of a
    pre-set placeholder -- through the real `interrupt()`/`Command(resume=
    ...)`/`PostgresSaver` graph.

    Mocks only the external sourcing call (`topical_clustering_agent.
    sourcing_agent`) and embeddings (`topical_clustering_agent.embed`), the
    same way Task 2.3.1's `test_gate1_edit_resume_runs_real_reconciliation`
    mocked only `subtopic_agent.embed` -- `cluster()` (real KMeans, below
    `Settings.clustering.kmeans_fallback_threshold` for this small article
    set) and `build_gate2_report`'s aggregation run unmocked end to end.
    """
    scored_articles = [
        _scored_article("EU passes AI Act", "reuters.com"),
        _scored_article("EU AI Act enters force", "apnews.com"),
        _scored_article("Brussels finalizes AI rules", "reuters.com"),
        _scored_article("Unrelated tech story A", "example-a.com"),
        _scored_article("Unrelated tech story B", "example-b.com"),
        _scored_article("Unrelated tech story C", "example-a.com"),
    ]
    # Two well-separated 2D groups -- deterministic KMeans(k=2) split.
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.1], [0.0, 0.2], [10.0, 10.0], [10.1, 10.1], [10.0, 10.2]]
    )

    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: scored_articles,
    )
    monkeypatch.setattr("newsresearch.agents.topical_clustering_agent.embed", lambda texts: vectors)

    node = make_gate2_node(cluster_report=make_real_cluster_report(7))
    builder = StateGraph(SubtopicState)
    builder.add_node("gate2", node)
    builder.add_edge(START, "gate2")
    builder.add_edge("gate2", END)
    graph = builder.compile(checkpointer=build_checkpointer(postgres_url))
    config = {"configurable": {"thread_id": "run-1:sub-real"}}

    initial_state = {
        "run_id": "run-1",
        "subtopic_id": "sub-real",
        "label": "eu ai act",
        "cluster_report": {},
    }
    result = graph.invoke(initial_state, config=config)

    payload = result["__interrupt__"][0].value["cluster_report"]
    # Not the stub's placeholder: real KMeans split into two clusters of 3,
    # and every article's domain counted into source_spread (6 total).
    assert sorted(payload["cluster_sizes"]) == [3, 3]
    assert len(payload["sample_headlines"]) == 4  # 2 per cluster
    assert set(payload["sample_headlines"]) <= {a.article["title"] for a in scored_articles}
    assert payload["source_spread"] == {"reuters.com": 2, "apnews.com": 1, "example-a.com": 2, "example-b.com": 1}

    result = graph.invoke(Command(resume={"action": "continue"}), config=config)
    assert result["cluster_report"] == payload
