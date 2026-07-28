import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from langgraph.types import Command
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.graph.build import NODE_ORDER, build_graph
from newsresearch.graph.state import GraphState, SubtopicState
from newsresearch.llm.schemas import SubtopicCandidate, SubtopicCandidateList

# Hermetic `testcontainers[postgres]` per Story 0.4's own precedent, so this
# test doesn't depend on the dev `docker compose up -d` stack being up (that
# dependency is exercised separately, manually, per Story 0.5's runtime note).

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


def _stub_subtopic_pipeline(monkeypatch, *, candidates, excess=None, articles=None):
    """Stub the real `subtopic` node's four `agents/subtopic_agent.py` calls
    (as imported into `graph/build.py`) so tests can drive `build_graph()`'s
    real subtopic -> gate1 sequence with deterministic `candidates`/`excess`/
    `articles`, without `propose_candidates`/`broad_topic_fetch` hitting real
    OpenAI/GDELT/RSS. `rank_and_cap_subtopics` is stubbed to directly return
    the desired final `candidates`/`excess` -- `propose_candidates`/
    `reconcile_subtopics`'s stubbed return values are only plumbing to keep
    `_make_subtopic_node`'s call chain from erroring, their content is unused.
    """
    monkeypatch.setattr(
        "newsresearch.graph.build.propose_candidates",
        lambda *a, **kw: SimpleNamespace(candidates=[]),
    )
    monkeypatch.setattr(
        "newsresearch.graph.build.broad_topic_fetch", lambda *a, **kw: articles or []
    )
    monkeypatch.setattr(
        "newsresearch.graph.build.reconcile_subtopics",
        lambda *a, **kw: {"reconciled": [], "total_articles": 0},
    )
    monkeypatch.setattr(
        "newsresearch.graph.build.rank_and_cap_subtopics",
        lambda *a, **kw: {"candidates": candidates, "excess": excess or []},
    )


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


def test_graph_invoke_runs_every_node_and_writes_a_durable_checkpoint(postgres_url, monkeypatch):
    _stub_subtopic_pipeline(monkeypatch, candidates=[], excess=[], articles=[])
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "test topic",
        "canonical_topic": "test topic",
        "run_id": "test-run",
        "subtopics": [],
        "approved": False,
    }
    config = {"configurable": {"thread_id": "test"}}

    # Gate 1 is now real (Task 2.6.2 follow-up): it interrupts
    # unconditionally, even on this no-candidates topology-smoke-test path,
    # so an approve-resume is required before the rest of NODE_ORDER runs.
    interrupted = graph.invoke(initial_state, config=config)
    assert "__interrupt__" in interrupted
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)

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

    `clustering` is real (PR #32 rework) and `gate2` is real (Task 2.6.2
    follow-up), so this drives a full resume cycle: each branch parks at its
    own real `interrupt()` before `gate2`'s `fan_trace` entry is written (see
    `_make_gate2_node`), so the trace is only complete after resuming every
    pending interrupt.
    """
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
        {"label": "china ai regulation", "article_count": 5},
    ]
    _stub_subtopic_pipeline(monkeypatch, candidates=candidates)
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "fanout-run",
        "subtopics": [],
        "approved": True,
    }
    config = {"configurable": {"thread_id": "fanout-test"}}

    # Gate 1 is now real (Task 2.6.2 follow-up) and sits upstream of
    # `fan_out`: approve-resume it first before the per-branch Gate 2
    # interrupts this test actually exercises.
    graph.invoke(initial_state, config=config)
    interrupted = graph.invoke(Command(resume={"action": "approve"}), config=config)
    resume_map = {i.id: {"action": "continue"} for i in interrupted["__interrupt__"]}
    result = graph.invoke(Command(resume=resume_map), config=config)

    fan_trace = result["fan_trace"]
    subtopic_ids_seen = {subtopic_id for _, subtopic_id, _ in fan_trace}
    assert len(subtopic_ids_seen) == len(candidates)

    for node_name in ("sourcing", "clustering", "gate2"):
        node_subtopic_ids = {
            subtopic_id for name, subtopic_id, _ in fan_trace if name == node_name
        }
        assert node_subtopic_ids == subtopic_ids_seen


def test_clustering_runs_exactly_once_per_branch_through_real_build_graph_gate2(
    postgres_url, monkeypatch
):
    """Follow-up to Task 2.6.2's review: re-run PR #32's call-counter proof
    against the *actual* compiled `build_graph()` topology, now that `gate2`
    is wired to the real, interrupting `gate2_node` (not a hand-assembled
    `StateGraph` standing in for it).

    PR #32 was tech-lead-rejected for calling `topical_clustering_agent`
    *inside* `gate2_node`, before `interrupt()` -- LangGraph replays a node
    function from the top on every resume, so that call ran twice per Gate 2
    pass. The fix moved the real work into `clustering` (one of
    `FAN_OUT_TARGET_NODES`, `Send`-relayed exactly once per branch, never
    replayed by a downstream interrupt/resume) -- this test proves that
    property still holds now that `gate2` itself is real, wired production
    topology, not a synthetic stand-in.
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

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
    ]
    _stub_subtopic_pipeline(monkeypatch, candidates=candidates)
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "gate2-real-run",
        "subtopics": [],
        "approved": True,
    }
    config = {"configurable": {"thread_id": "gate2-real-test"}}

    # Gate 1 is now real (Task 2.6.2 follow-up): approve-resume it first.
    graph.invoke(initial_state, config=config)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
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
    # is still 1 per branch after resume is the actual regression check --
    # now against the real, compiled `build_graph()` output.
    assert call_counts == {"eu ai act": 1, "us executive order": 1}
    assert graph.get_state(config).next == ()


def test_gate2_blocks_each_fanned_branch_independently_via_real_send(postgres_url, monkeypatch):
    """Wave 1's `test_gate2_blocks_each_subtopic_branch_independently` proved
    independent blocking with hand-built distinct `thread_id`s standing in
    for distinct fanned-out branches. This re-verifies the same property
    through the real `Send`-based fan-out (one shared `thread_id`, N
    concurrently-pending Gate 2 interrupts): resuming one branch's interrupt
    must not force its sibling's to resolve too.
    """
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
    ]
    _stub_subtopic_pipeline(monkeypatch, candidates=candidates)
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "gate2-independent-run",
        "subtopics": [],
        "approved": True,
    }
    config = {"configurable": {"thread_id": "gate2-independent-test"}}

    # Gate 1 is now real (Task 2.6.2 follow-up): approve-resume it first.
    graph.invoke(initial_state, config=config)
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)
    interrupts = result["__interrupt__"]
    assert len(interrupts) == len(candidates)
    assert graph.get_state(config).next == ("gate2", "gate2")

    # Resume only the first branch's interrupt; the second's own id gets no
    # resume value, so it must re-park at its own `interrupt()` rather than
    # being forced through alongside the first.
    first, second = interrupts
    partial_result = graph.invoke(
        Command(resume={first.id: {"action": "continue"}}), config=config
    )

    assert graph.get_state(config).next == ("gate2",)
    gate2_trace = [entry for entry in partial_result["fan_trace"] if entry[0] == "gate2"]
    assert len(gate2_trace) == 1

    # Second branch still pending with its original interrupt id/payload --
    # resuming it independently afterward completes the run.
    still_pending = graph.get_state(config).tasks
    assert any(t.interrupts and t.interrupts[0].id == second.id for t in still_pending)

    final_result = graph.invoke(
        Command(resume={second.id: {"action": "continue"}}), config=config
    )
    assert graph.get_state(config).next == ()
    gate2_trace_final = [entry for entry in final_result["fan_trace"] if entry[0] == "gate2"]
    subtopic_ids = {entry[1] for entry in gate2_trace_final}
    assert len(subtopic_ids) == len(candidates)


def test_gate1_approve_resume_proceeds_through_real_build_graph(postgres_url, monkeypatch):
    """Follow-up to Task 2.6.2's review: `gate1` is now a real, interrupting
    node in the actual compiled `build_graph()` topology (`_make_gate1_node`),
    not the generic passthrough it was wired as before -- proves a plain
    approve-resume blocks at Gate 1 first and then proceeds unchanged into
    the rest of the topology (here, straight through to Gate 2's own
    interrupts, one per approved candidate).
    """
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    candidates = [
        {"label": "eu ai act", "article_count": 12},
        {"label": "us executive order", "article_count": 8},
    ]
    _stub_subtopic_pipeline(monkeypatch, candidates=candidates, articles=[])
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "gate1-real-approve-run",
        "subtopics": [],
        "approved": False,
    }
    config = {"configurable": {"thread_id": "gate1-real-approve-test"}}

    interrupted = graph.invoke(initial_state, config=config)
    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["candidates"] == candidates
    assert graph.get_state(config).next == ("gate1",)

    result = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert result["approved"] is True
    assert result["candidates"] == candidates
    # Approving proceeds straight past `fan_out` into Gate 2's own
    # per-branch interrupts -- proof gate1 isn't dead-ending the run.
    assert len(result["__interrupt__"]) == len(candidates)


def test_gate1_edit_resume_runs_real_reconciliation_through_real_build_graph(
    postgres_url, monkeypatch
):
    """Same acceptance as `test_gate1.py`'s
    `test_gate1_edit_resume_runs_real_reconciliation`, now proven through the
    actual compiled `build_graph()` topology instead of a hand-assembled
    `StateGraph(GraphState)` standing in for it -- confirms `_make_gate1_node`
    really does bind `make_real_reconcile` to `state["articles"]` at
    invocation time rather than silently falling back to `stub_reconcile`'s
    identity pass-through.
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
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    articles = [{"title": s} for s in clustering["sentences"]]
    # `_stub_subtopic_pipeline`'s `fake_embed` mock above (monkeypatched onto
    # `subtopic_agent.embed` directly) makes the *real* `reconcile_subtopics`
    # safe to call, so only `propose_candidates`/`broad_topic_fetch` need
    # stubbing here -- the subtopic node's own initial candidates/articles
    # still land deterministically, matching this test's original
    # invoke-time seed.
    monkeypatch.setattr(
        "newsresearch.graph.build.propose_candidates",
        lambda *a, **kw: SimpleNamespace(candidates=[]),
    )
    monkeypatch.setattr("newsresearch.graph.build.broad_topic_fetch", lambda *a, **kw: articles)
    monkeypatch.setattr(
        "newsresearch.graph.build.reconcile_subtopics",
        lambda *a, **kw: {"reconciled": [], "total_articles": 0},
    )
    monkeypatch.setattr(
        "newsresearch.graph.build.rank_and_cap_subtopics",
        lambda *a, **kw: {
            "candidates": [
                {"label": label, "article_count": 0} for label in merge_fixture["candidate_labels"]
            ],
            "excess": [],
        },
    )
    graph = build_graph(database_url=postgres_url)

    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "gate1-real-edit-run",
        "subtopics": [],
        "approved": False,
    }
    config = {"configurable": {"thread_id": "gate1-real-edit-test"}}

    graph.invoke(initial_state, config=config)
    edited_candidates = [{"label": label} for label in edited_labels]
    result = graph.invoke(
        Command(resume={"action": "edit", "candidates": edited_candidates}),
        config=config,
    )

    assert result["approved"] is True
    # Not a passthrough: the two near-duplicate EU AI Act candidates merge
    # into one subtopic with real, recomputed article counts/ordering --
    # never the edit-resume's unchanged 3-item edited list `stub_reconcile`
    # would have produced.
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


def test_subtopic_node_populates_candidates_excess_articles_through_real_build_graph(
    postgres_url, monkeypatch
):
    """Story 2.2 production-wiring follow-up: the real `subtopic` node
    (`_make_subtopic_node`) composes `propose_candidates` ->
    `broad_topic_fetch` -> `reconcile_subtopics` -> `rank_and_cap_subtopics`
    for real, unconditionally, upstream of Gate 1 -- proven through the
    actual compiled `build_graph()`, mocking only the external boundaries
    (`get_chat_model`, `sourcing_agent`, `embed`) the same way
    `test_subtopic_agent.py`/`test_gate1_edit_resume_runs_real_reconciliation_
    through_real_build_graph` already do, not the whole node.
    """
    clustering = json.loads((FIXTURES_DIR / "clustering_synthetic_topics.json").read_text())
    merge_fixture = json.loads((FIXTURES_DIR / "reconciliation_merge.json").read_text())

    article_vectors = np.array(clustering["embeddings"])
    candidate_vectors = np.array(merge_fixture["candidate_embeddings"])
    articles = [{"title": s} for s in clustering["sentences"]]
    candidate_list = SubtopicCandidateList(
        candidates=[
            SubtopicCandidate(label=label, rationale="rationale")
            for label in merge_fixture["candidate_labels"]
        ]
    )

    def fake_embed(texts):
        if len(texts) == len(article_vectors):
            return article_vectors
        if len(texts) == len(candidate_vectors):
            return candidate_vectors
        raise AssertionError(f"unexpected embed() call with {len(texts)} texts")

    mock_chat_model = MagicMock()
    mock_chat_model.with_structured_output.return_value = lambda *a, **kw: candidate_list
    monkeypatch.setattr(
        "newsresearch.agents.subtopic_agent.get_chat_model", lambda stage: mock_chat_model
    )
    monkeypatch.setattr(
        "newsresearch.agents.subtopic_agent.get_langfuse_callback_handler",
        lambda settings: MagicMock(),
    )
    monkeypatch.setattr(
        "newsresearch.agents.subtopic_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [
            ScoredArticle(article=a, reputation_score=1.0, reputation_tier="major")
            for a in articles
        ],
    )
    monkeypatch.setattr("newsresearch.agents.subtopic_agent.embed", fake_embed)
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.sourcing_agent",
        lambda keywords, lookback_days, **kwargs: [],
    )
    monkeypatch.setattr(
        "newsresearch.agents.topical_clustering_agent.embed", lambda texts: np.empty((0, 2))
    )

    graph = build_graph(database_url=postgres_url)
    initial_state: GraphState = {
        "topic": "AI regulation",
        "canonical_topic": "ai regulation",
        "run_id": "subtopic-real-run",
        "subtopics": [],
        "approved": False,
    }
    config = {"configurable": {"thread_id": "subtopic-real-test"}}

    interrupted = graph.invoke(initial_state, config=config)
    assert "__interrupt__" in interrupted
    assert graph.get_state(config).next == ("gate1",)

    state = graph.get_state(config).values
    assert state["articles"] == articles

    # Real reconciliation collapsed the fixture's 2 near-duplicate EU AI Act
    # candidates into 1 merged subtopic -- 3 subtopics total (down from 4
    # candidates), never the raw, unreconciled candidate list a passthrough
    # `subtopic` node would have left behind.
    all_subtopics = state["candidates"] + state["excess"]
    assert len(all_subtopics) == 3
    merged = next(c for c in all_subtopics if c["action"] == "merge")
    assert set(merged["merged_from"]) == {
        "EU AI Act enforcement actions",
        "European Union AI Act compliance crackdown",
    }
    for c in all_subtopics:
        assert c["article_count"] > 0
        assert "distinctiveness_score" in c
        assert "centroid" not in c
