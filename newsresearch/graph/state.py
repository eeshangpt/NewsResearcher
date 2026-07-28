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

    `candidates`/`excess` are Phase 2 Gate 1 additions (Task 2.3.1), now
    populated for real by the `subtopic` node
    (`graph/build.py::_make_subtopic_node`, Story 2.2 production wiring):
    each is `agents/subtopic_agent.py::rank_and_cap_subtopics`'s real output
    shape (label + article count + distinctiveness score, per TRD's Subtopic
    Agent description), not a stand-in. `candidates` is the ranked/capped
    list; `excess` is the "also detected" set retained separately.
    """

    topic: str
    canonical_topic: str
    run_id: str
    subtopics: list[str]
    approved: bool
    candidates: list[dict]
    excess: list[dict]

    # Gate 1 production-wiring follow-up (Task 2.6.2 review): the
    # broad-fetch article set `candidates`/`excess` were originally proposed
    # against (`agents/subtopic_agent.py::broad_topic_fetch`), threaded
    # through so a real Gate 1 edit-resume can re-run `make_real_reconcile`
    # against it. The real `subtopic` node (Story 2.2 production wiring,
    # `graph/build.py::_make_subtopic_node`) is now this field's sole
    # writer -- its own `broad_topic_fetch` call's output -- replacing the
    # invoke-time seed a stub `subtopic` node used to require.
    articles: list[dict]

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

    # Gate 2 real-clustering rework (PR #32 tech-lead-rejected re-fix):
    # bridges the `clustering` node's real `cluster_report` output
    # (computed exactly once per `Send`-fanned branch) across to
    # `_make_relay_router`'s clustering->gate2 hop, which folds each
    # branch's own entry into that branch's outgoing `Send` payload as a
    # plain `cluster_report` field -- reducer-safe for the same reason as
    # `fan_trace`: concurrent branches write this in the same superstep and
    # a plain field can't hold N distinct values.
    cluster_reports: Annotated[list[tuple[str, dict]], operator.add]


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
