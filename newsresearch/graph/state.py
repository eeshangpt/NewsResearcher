"""Shared LangGraph state schemas (Phase 0 Task 0.5.1).

`GraphState` is the top-level state threaded through the whole pipeline
(TRD section 3.1: Subtopic -> Gate 1 -> fan-out -> ... -> Timeline).
`SubtopicState` is the per-subtopic sub-state shape that Phase 2's
`Send`-based fan-out will carry into each concurrent per-subtopic branch —
only the shape is defined here, not the fan-out logic itself.
"""

from typing import TypedDict


class GraphState(TypedDict):
    """Top-level pipeline state."""

    topic: str
    canonical_topic: str
    run_id: str
    subtopics: list[str]
    approved: bool


class SubtopicState(TypedDict):
    """Per-subtopic sub-state for Phase 2's `Send`-based fan-out.

    Carries just enough identity for a fanned-out branch to run
    independently (its own subtopic) while remaining traceable back to the
    parent run.
    """

    run_id: str
    subtopic_id: str
    label: str
