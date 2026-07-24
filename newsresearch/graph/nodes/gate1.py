"""Gate 1 interrupt node (Phase 2 Task 2.3.1) -- mechanics only.

Presents the current `candidates`/`excess` from `GraphState` (stubbed per
Task 2.2.4's not-yet-built rank/cap/excess-retention output, see
`graph/state.py`) and blocks via `interrupt()` until a human resumes with an
approve or edit decision.

This is Wave 1 of the tech-lead-approved Phase 2 re-plan: the real Subtopic
Agent reconciliation (Task 2.2.3 / 2.2.3b) doesn't exist yet, so an
edit-resume is wired to a pluggable `reconcile` hook that defaults to an
identity placeholder rather than blocking this task on unimplemented
clustering/reconciliation logic. The interrupt/resume *mechanics* --
interrupt, accept edited input, call *some* reconciliation step, resume --
are what this task delivers now. Full acceptance against Task 2.3.1's
original criterion ("an edit-resume ... re-triggers Task 2.2.3's
reconciliation on the edited set") is explicitly deferred to Wave 3, once
Task 2.2.3b's real reconciliation logic lands and can be swapped in via the
`reconcile` parameter -- a field-population change, not a mechanics rework.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.types import interrupt

from newsresearch.graph.state import GraphState

ReconcileFn = Callable[[list[dict]], list[dict]]


def stub_reconcile(candidates: list[dict]) -> list[dict]:
    """Identity placeholder for Task 2.2.3b's real reconciliation logic.

    Task 2.2.3 (embed+cluster reconciliation: merge candidates mapping to
    the same cluster, split clusters spanning multiple candidates, drop
    unsupported candidates) doesn't exist yet. A real edit-resume should
    re-run that reconciliation against the edited candidate set. Until
    Task 2.2.3b lands, edited candidates pass through unchanged so Gate 1's
    interrupt/resume mechanics can be built and tested now. Replace this
    with Task 2.2.3b's real reconciliation once it lands.
    """
    return candidates


def make_gate1_node(reconcile: ReconcileFn = stub_reconcile) -> Callable[[GraphState], dict[str, Any]]:
    """Build the Gate 1 node, with a pluggable reconciliation hook.

    Interrupts with `{"candidates": ..., "excess": ...}` from `GraphState`.
    Resumes via `Command(resume=...)` with one of:
      - `{"action": "approve"}`: proceeds with `state["candidates"]`
        unchanged.
      - `{"action": "edit", "candidates": [...]}`: the edited candidate
        list is passed through `reconcile` (stubbed by default; swap in
        Task 2.2.3b's real logic via this parameter) before proceeding.
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
