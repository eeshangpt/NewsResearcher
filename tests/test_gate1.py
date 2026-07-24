"""Gate 1 interrupt/resume mechanics (Task 2.3.1) + durability (Task 2.3.2).

Builds a small standalone `StateGraph(GraphState)` with just the `gate1`
node wired `START -> gate1 -> END`, compiled with the real `PostgresSaver`
via `build.py`'s `build_checkpointer()` -- reused, not reimplemented -- so
Gate 1's interrupt/resume mechanics and Postgres durability are exercised
without touching `graph/build.py`'s NODE_ORDER topology (out of scope for
this task; real wiring lands with Task 2.4.1's fan-out).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from testcontainers.postgres import PostgresContainer

from newsresearch.graph.build import build_checkpointer
from newsresearch.graph.nodes.gate1 import gate1_node, make_gate1_node, make_real_reconcile
from newsresearch.graph.state import GraphState

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


def _build_gate1_graph(postgres_url, node=gate1_node):
    builder = StateGraph(GraphState)
    builder.add_node("gate1", node)
    builder.add_edge(START, "gate1")
    builder.add_edge("gate1", END)
    checkpointer = build_checkpointer(postgres_url)
    return builder.compile(checkpointer=checkpointer)


def _initial_state():
    return {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "run-1",
        "subtopics": [],
        "approved": False,
        "candidates": [
            {"label": "eu ai act", "article_count": 12},
            {"label": "us executive order", "article_count": 8},
        ],
        "excess": [{"label": "china ai regs", "article_count": 2}],
    }


def test_gate1_interrupts_with_candidates_and_excess(postgres_url):
    graph = _build_gate1_graph(postgres_url)
    config = {"configurable": {"thread_id": "gate1-interrupt"}}

    result = graph.invoke(_initial_state(), config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["candidates"] == _initial_state()["candidates"]
    assert payload["excess"] == _initial_state()["excess"]

    state = graph.get_state(config)
    assert state.next == ("gate1",)


def test_gate1_approve_resume_proceeds_with_candidates_unchanged(postgres_url):
    graph = _build_gate1_graph(postgres_url)
    config = {"configurable": {"thread_id": "gate1-approve"}}

    graph.invoke(_initial_state(), config=config)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert result["approved"] is True
    assert result["candidates"] == _initial_state()["candidates"]


def test_gate1_edit_resume_calls_the_pluggable_reconcile_hook(postgres_url):
    calls = []

    def custom_reconcile(candidates: list[dict]) -> list[dict]:
        calls.append(candidates)
        # Non-identity, to prove the hook is actually plugged in and called,
        # not just decorative -- real Task 2.2.3b reconciliation logic will
        # replace this function wholesale later.
        return [c for c in candidates if c["article_count"] > 5]

    node = make_gate1_node(reconcile=custom_reconcile)
    graph = _build_gate1_graph(postgres_url, node=node)
    config = {"configurable": {"thread_id": "gate1-edit"}}

    graph.invoke(_initial_state(), config=config)
    edited_candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "china ai regs", "article_count": 2},
    ]
    result = graph.invoke(
        Command(resume={"action": "edit", "candidates": edited_candidates}),
        config=config,
    )

    assert calls == [edited_candidates]
    assert result["approved"] is True
    assert result["candidates"] == [{"label": "eu ai act", "article_count": 12}]


def test_gate1_edit_resume_runs_real_reconciliation(postgres_url, monkeypatch):
    """Task 2.3.1's real acceptance: an edit-resume re-triggers
    `reconcile_subtopics` + `rank_and_cap_subtopics` on the edited candidate
    set (via `make_real_reconcile`), not `stub_reconcile`'s identity
    pass-through.

    Uses the data-scientist's real committed fixtures --
    `clustering_synthetic_topics.json`'s real embeddings as the broad-fetch
    article set, `reconciliation_merge.json`'s candidate labels/embeddings --
    with `subtopic_agent.embed` monkeypatched to return them directly (no
    live embedding-model call), while `cluster()`'s real HDBSCAN and
    `reconcile_subtopics`/`rank_and_cap_subtopics`'s real logic run
    unmocked.
    """
    clustering = json.loads((FIXTURES_DIR / "clustering_synthetic_topics.json").read_text())
    merge_fixture = json.loads((FIXTURES_DIR / "reconciliation_merge.json").read_text())

    article_vectors = np.array(clustering["embeddings"])
    # Edit-resume drops "US executive order on AI safety" -- 3 of the
    # fixture's 4 candidates survive the edit.
    edited_labels = merge_fixture["candidate_labels"][:-1]
    candidate_vectors = np.array(merge_fixture["candidate_embeddings"][:-1])

    def fake_embed(texts):
        if len(texts) == len(article_vectors):
            return article_vectors
        if len(texts) == len(candidate_vectors):
            return candidate_vectors
        raise AssertionError(f"unexpected embed() call with {len(texts)} texts")

    monkeypatch.setattr("newsresearch.agents.subtopic_agent.embed", fake_embed)

    articles = [{"title": s} for s in clustering["sentences"]]
    node = make_gate1_node(reconcile=make_real_reconcile(articles))
    graph = _build_gate1_graph(postgres_url, node=node)
    config = {"configurable": {"thread_id": "gate1-real-reconcile"}}

    initial_state = {
        **_initial_state(),
        "candidates": [{"label": label, "article_count": 0} for label in merge_fixture["candidate_labels"]],
        "excess": [],
    }
    graph.invoke(initial_state, config=config)

    edited_candidates = [{"label": label} for label in edited_labels]
    result = graph.invoke(
        Command(resume={"action": "edit", "candidates": edited_candidates}),
        config=config,
    )

    assert result["approved"] is True
    # Not a passthrough: the two near-duplicate EU AI Act candidates merge
    # into one subtopic with real, recomputed article counts/ordering --
    # never the stub's unchanged 3-item edited list.
    assert len(result["candidates"]) == 2
    merged = next(c for c in result["candidates"] if c["action"] == "merge")
    assert set(merged["merged_from"]) == {
        "EU AI Act enforcement actions",
        "European Union AI Act compliance crackdown",
    }
    for c in result["candidates"]:
        assert c["article_count"] > 0
        assert "distinctiveness_score" in c
        assert "centroid" not in c


def test_gate1_unrecognized_resume_action_raises(postgres_url):
    graph = _build_gate1_graph(postgres_url)
    config = {"configurable": {"thread_id": "gate1-bad-action"}}

    graph.invoke(_initial_state(), config=config)
    with pytest.raises(ValueError, match="unrecognized resume action"):
        graph.invoke(Command(resume={"action": "reject"}), config=config)


def test_gate1_interrupt_state_survives_simulated_process_restart(postgres_url):
    """Task 2.3.2: kill-and-restart durability check for Gate 1.

    A pytest process can't literally be killed and restarted mid-test, so
    this simulates a restart the way `test_graph_build.py`'s own durability
    checks do: discard the original in-process graph/checkpointer object
    entirely and construct a brand-new `ConnectionPool` / `PostgresSaver` /
    compiled graph from scratch against the *same* Postgres URL, then
    confirm the pending-approval state is read back correctly from
    Postgres via the new object -- not merely paused in the old one -- and
    that resuming via the same `thread_id` on the new object works.
    """
    config = {"configurable": {"thread_id": "gate1-durability"}}

    original_graph = _build_gate1_graph(postgres_url)
    original_graph.invoke(_initial_state(), config=config)
    del original_graph  # simulate the process dying; nothing in-memory survives

    restarted_graph = _build_gate1_graph(postgres_url)
    restarted_state = restarted_graph.get_state(config)
    assert restarted_state.next == ("gate1",)

    interrupt_payload = restarted_state.tasks[0].interrupts[0].value
    assert interrupt_payload["candidates"] == _initial_state()["candidates"]
    assert interrupt_payload["excess"] == _initial_state()["excess"]

    result = restarted_graph.invoke(Command(resume={"action": "approve"}), config=config)
    assert result["approved"] is True
    assert result["candidates"] == _initial_state()["candidates"]
