"""Gate 1 interrupt node (Phase 2 Task 2.3.1).

Presents the current `candidates`/`excess` from `GraphState` and blocks via
`interrupt()` until a human resumes with an approve or edit decision.

Task 2.2.3b/2.2.4's real reconciliation (`reconcile_subtopics` +
`rank_and_cap_subtopics`, `agents/subtopic_agent.py`) has now landed, closing
Task 2.3.1's original acceptance: an edit-resume re-triggers real
reconciliation on the edited candidate set via `make_real_reconcile`, not an
identity pass-through. `stub_reconcile` is kept as the neutral default for
`make_gate1_node`'s bare mechanics (approve/durability/bad-action tests that
don't exercise reconciliation at all) -- a real edit-resume node is built by
passing `reconcile=make_real_reconcile(articles, settings=...)` explicitly,
the same pluggable-hook pattern already proven by
`test_gate1_edit_resume_calls_the_pluggable_reconcile_hook`.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.types import interrupt

from newsresearch.agents.subtopic_agent import rank_and_cap_subtopics, reconcile_subtopics
from newsresearch.config import Settings
from newsresearch.graph.state import GraphState
from newsresearch.llm.schemas import SubtopicCandidate

ReconcileFn = Callable[[list[dict]], list[dict]]


def make_real_reconcile(
    articles: list[dict[str, Any]], *, settings: Settings | None = None
) -> ReconcileFn:
    """Build a real `reconcile` hook bound to the broad-fetch `articles` the
    edited candidate set was originally proposed against (Task 2.2.3b/2.2.4).

    Re-runs `reconcile_subtopics` (embed + cluster `articles`, then
    merge/split/drop the edited candidates against those clusters) and then
    `rank_and_cap_subtopics` -- in that order, never skipped -- so the raw
    numpy `centroid` arrays `reconcile_subtopics` carries are stripped before
    the result could reach a JSON-serializable, Postgres-durable interrupt
    payload (per the tech-lead's Task 2.2.3b/2.2.4 review note).
    """

    def real_reconcile(candidates: list[dict]) -> list[dict]:
        subtopic_candidates = [
            SubtopicCandidate(label=c["label"], rationale=c.get("rationale", ""))
            for c in candidates
        ]
        reconciled = reconcile_subtopics(articles, subtopic_candidates, settings=settings)
        capped = rank_and_cap_subtopics(
            reconciled["reconciled"], reconciled["total_articles"], settings=settings
        )
        return capped["candidates"]

    return real_reconcile


def stub_reconcile(candidates: list[dict]) -> list[dict]:
    """Identity placeholder, kept as `make_gate1_node`'s neutral default.

    A real edit-resume should re-run reconciliation against the edited
    candidate set -- see `make_real_reconcile` for the real Task 2.2.3b/2.2.4
    logic. This identity pass-through remains the default so bare
    interrupt/resume mechanics tests (approve, unrecognized-action,
    durability) that never exercise reconciliation don't need to supply an
    `articles` context they have no use for.
    """
    return candidates


def make_gate1_node(reconcile: ReconcileFn = stub_reconcile) -> Callable[[GraphState], dict[str, Any]]:
    """Build the Gate 1 node, with a pluggable reconciliation hook.

    Interrupts with `{"candidates": ..., "excess": ...}` from `GraphState`.
    Resumes via `Command(resume=...)` with one of:
      - `{"action": "approve"}`: proceeds with `state["candidates"]`
        unchanged.
      - `{"action": "edit", "candidates": [...]}`: the edited candidate
        list is passed through `reconcile` (identity by default; pass
        `reconcile=make_real_reconcile(articles, settings=...)` for real
        Task 2.2.3b/2.2.4 reconciliation) before proceeding.
    """

    def gate1_node(state: GraphState) -> dict[str, Any]:
        decision = interrupt({"candidates": state["candidates"], "excess": state["excess"]})

        action = decision.get("action")
        if action == "approve":
            return {"candidates": state["candidates"], "approved": True}
        if action == "edit":
            edited = decision.get("candidates", [])
            return {"candidates": reconcile(edited), "approved": True}

        raise ValueError(f"gate1: unrecognized resume action {action!r}")

    return gate1_node


# Default instance for direct import/wiring convenience.
gate1_node = make_gate1_node()
