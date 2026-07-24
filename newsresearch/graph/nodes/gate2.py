"""Gate 2 interrupt node (Phase 2 Task 2.6.2) -- mechanics only.

Presents the stubbed `cluster_report` from `SubtopicState` (Task 2.6.1's
zero-LLM-cost aggregation output, not yet built -- see `graph/state.py` for
the stand-in shape and its reasoning) and blocks per-subtopic-branch until a
human reviews it and resumes with a continue decision.

Real per-branch isolation depends on `Send`-based fan-out (Task 2.4.1),
which doesn't exist yet either. Per LangGraph's own model, distinct
concurrently-running branches get distinct `thread_id`/checkpoint
namespaces, so this node's independent-blocking property is exercised
directly in tests against distinct `thread_id`s standing in for distinct
fanned-out branches, rather than waiting on real fan-out wiring to exist.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from newsresearch.graph.state import SubtopicState


def gate2_node(state: SubtopicState) -> dict[str, Any]:
    """Interrupt with the stubbed `cluster_report`, resume on confirmation.

    Resume payload: `{"action": "continue"}`. Gate 2 confirmation gates
    progression to the next (expensive) per-subtopic stage; it doesn't
    edit the report itself, so a successful resume is a no-op on state.
    """
    decision = interrupt({"cluster_report": state["cluster_report"]})

    if decision.get("action") != "continue":
        raise ValueError(f"gate2: unrecognized resume action {decision.get('action')!r}")

    return {}
