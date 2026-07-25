"""Shared LangGraph state schemas (Phase 0 Task 0.5.1).

`GraphState` is the top-level state threaded through the whole pipeline
(TRD section 3.1: Subtopic -> Gate 1 -> fan-out -> ... -> Timeline).
`SubtopicState` is the per-subtopic sub-state shape that Phase 2's
`Send`-based fan-out will carry into each concurrent per-subtopic branch —
only the shape is defined here, not the fan-out logic itself.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """Top-level pipeline state.

    `candidates`/`excess` are Phase 2 Gate 1 additions (Task 2.3.1), stubbed
    for now: Task 2.2.4 (rank/cap/excess-retention) doesn't exist yet, so
    these are populated with a stand-in shape until then. Each dict is
    shaped `{"label": str, "article_count": int}`, matching the TRD's
    description of the Subtopic Agent's real output ("label + supporting
    article count + distinctiveness score", TRD sec. "Subtopic Agent")
    minus the distinctiveness score, which Task 2.2.4 will add later as a
    field-population change, not a payload-shape rework. `candidates` is
    the ranked/capped list; `excess` is the "also detected" set retained
    separately per the same task.
    """

    topic: str
    canonical_topic: str
    run_id: str
    subtopics: list[str]
    approved: bool
    candidates: list[dict]
    excess: list[dict]

    # Task 2.4.1: accumulator every `Send`-fanned branch writes `(node_name,
    # subtopic_id, label)` into as it passes through `graph/build.py`'s
    # `sourcing`/`clustering`/`gate2` fan-out targets. `operator.add` lets
    # concurrent branches merge their writes instead of colliding (plain,
    # non-reducer fields can't hold N distinct per-branch values written in
    # the same superstep -- LangGraph rejects that outright). This is also
    # how each hop's routing rediscovers a branch's own identity to
    # re-`Send` it forward -- see `graph/build.py::_make_relay_router` --
    # and, incidentally, how a fan-out test can prove each downstream node
    # actually ran once per approved candidate with a distinct
    # `subtopic_id` (not once, per the task's acceptance).
    fan_trace: Annotated[list[tuple[str, str, str]], operator.add]


class SubtopicState(TypedDict):
    """Per-subtopic sub-state for Phase 2's `Send`-based fan-out.

    Carries just enough identity for a fanned-out branch to run
    independently (its own subtopic) while remaining traceable back to the
    parent run.

    `cluster_report` is a Phase 2 Gate 2 addition (Task 2.6.2), stubbed for
    now: Task 2.6.1's `reports/gate2_report.py` doesn't exist yet, so this
    carries a stand-in shape until then. Task 2.6.1's stated acceptance
    criterion is "cluster-size/sample-headline/source-spread fields with
    zero calls to `get_chat_model`" (EXECUTION_PLAN.md Task 2.6.1), so the
    shape here is `{"cluster_sizes": list[int], "sample_headlines":
    list[str], "source_spread": dict[str, int]}` -- one field per named
    criterion, so a Wave 4 `backend-engineer` populating this for real has
    an exact field to fill per aggregation.
    """

    run_id: str
    subtopic_id: str
    label: str
    cluster_report: dict
