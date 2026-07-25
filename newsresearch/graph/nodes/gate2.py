"""Gate 2 interrupt node (Phase 2 Task 2.6.2, real report wiring Task 2.6.1-follow-up).

Presents `cluster_report` from `SubtopicState` and blocks per-subtopic-branch
until a human reviews it and resumes with a continue decision.

Task 2.6.1's real aggregation (`build_gate2_report`, `reports/gate2_report.py`)
plus Task 2.5.1's real per-subtopic clustering (`topical_clustering_agent`)
have now landed, closing this follow-up: a Gate 2 node built with
`make_real_cluster_report(lookback_days, ...)` presents a genuinely computed
`cluster_report`, not the Wave 1 stub's pass-through of a pre-populated
placeholder. `stub_cluster_report` is kept as the neutral default for
`make_gate2_node`'s bare mechanics (interrupt/resume, per-branch isolation,
bad-action) tests that supply `cluster_report` directly on `SubtopicState`
and don't exercise real clustering at all -- a real node is built by passing
`cluster_report=make_real_cluster_report(lookback_days, settings=...)`
explicitly, the same pluggable-hook pattern already proven by `gate1.py`'s
`make_real_reconcile`.

Real per-branch isolation depends on `Send`-based fan-out (Task 2.4.1), which
now exists, but `graph/build.py`'s "clustering"/"gate2" nodes remain
passthrough stubs (Story 2.5/2.6 real wiring is a separate, not-yet-done
task) -- so this node's independent-blocking property is still exercised
directly against distinct `thread_id`s standing in for distinct fanned-out
branches, per Wave 1's original design.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.types import interrupt
from psycopg_pool import ConnectionPool

from newsresearch.agents.topical_clustering_agent import topical_clustering_agent
from newsresearch.config import Settings
from newsresearch.graph.state import SubtopicState
from newsresearch.reports.gate2_report import build_gate2_report

ClusterReportFn = Callable[[SubtopicState], dict[str, Any]]


def make_real_cluster_report(
    lookback_days: int, *, pool: ConnectionPool | None = None, settings: Settings | None = None
) -> ClusterReportFn:
    """Build a real `cluster_report` hook bound to `lookback_days` (Task 2.5.1/2.6.1).

    Runs `topical_clustering_agent` (per-subtopic sourcing + coarse
    clustering) for the branch's own `subtopic_id`/`label`, then
    `build_gate2_report` (zero-LLM-cost aggregation) over its output -- in
    that order, never skipped -- so Gate 2 presents genuine cluster
    sizes/sample headlines/source spread, not the stub's pre-set placeholder.
    """

    def real_cluster_report(state: SubtopicState) -> dict[str, Any]:
        clustering_result = topical_clustering_agent(
            state["subtopic_id"], state["label"], lookback_days, pool=pool, settings=settings
        )
        return build_gate2_report(clustering_result)

    return real_cluster_report


def stub_cluster_report(state: SubtopicState) -> dict[str, Any]:
    """Identity placeholder, kept as `make_gate2_node`'s neutral default.

    A real Gate 2 node should compute `cluster_report` from the branch's own
    per-subtopic clustering output -- see `make_real_cluster_report` for the
    real Task 2.5.1/2.6.1 logic. This pass-through remains the default so
    bare interrupt/resume mechanics tests (per-branch isolation, bad-action)
    that supply `cluster_report` directly on `SubtopicState` don't need to
    supply the `lookback_days`/`pool`/`settings` context they have no use
    for.
    """
    return state["cluster_report"]


def make_gate2_node(
    cluster_report: ClusterReportFn = stub_cluster_report,
) -> Callable[[SubtopicState], dict[str, Any]]:
    """Build the Gate 2 node, with a pluggable cluster-report-producer hook.

    Interrupts with `{"cluster_report": ...}` computed by `cluster_report`
    (identity read of `state["cluster_report"]` by default; pass
    `cluster_report=make_real_cluster_report(lookback_days, settings=...)`
    for real Task 2.5.1/2.6.1 aggregation). Resumes via `Command(resume=...)`
    with `{"action": "continue"}`; Gate 2 confirmation gates progression to
    the next (expensive) per-subtopic stage, so a successful resume writes
    the computed report back onto state and otherwise doesn't edit it.
    """

    def gate2_node(state: SubtopicState) -> dict[str, Any]:
        report = cluster_report(state)
        decision = interrupt({"cluster_report": report})

        if decision.get("action") != "continue":
            raise ValueError(f"gate2: unrecognized resume action {decision.get('action')!r}")

        return {"cluster_report": report}

    return gate2_node


# Default instance for direct import/wiring convenience.
gate2_node = make_gate2_node()
