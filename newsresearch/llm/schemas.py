"""Pydantic output schemas for structured LLM calls.

Call sites use `model.with_structured_output(Schema)` instead of hand-rolled
JSON parsing. Real schemas (subtopic proposals, claims, framing labels,
briefing sections, etc.) get added when the Phase 2+ agents that need them
are built.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubtopicCandidate(BaseModel):
    """One LLM-proposed candidate subtopic, pre-reconciliation (Task 2.2.1b).

    Field shape per `notebooks/phase2-subtopic-prompt-design.md`'s
    data-scientist-authored proposal, paired with
    `llm/prompts/subtopic_propose.txt`.
    """

    label: str = Field(
        ...,
        description=(
            "Short (roughly 3-10 word) label naming a specific, concrete "
            "angle on the topic. Never a political-leaning, outlet, or "
            "audience label."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "One sentence explaining why this is a distinct, reportable "
            "angle on the topic, not a restatement of the topic or of "
            "another candidate."
        ),
    )


class SubtopicCandidateList(BaseModel):
    """Structured output of the `subtopic_propose.txt` prompt (Task 2.2.1b).

    `candidates` is intentionally allowed to be shorter than the requested
    `n_candidates` (the prompt's rule 6) -- no min/max length validation
    here; Task 2.2.3/2.2.4 already handles capping/excess downstream.
    """

    candidates: list[SubtopicCandidate] = Field(
        ...,
        description="Non-overlapping candidate subtopics, at most n_candidates long.",
    )
