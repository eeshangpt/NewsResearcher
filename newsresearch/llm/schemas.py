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


class Claim(BaseModel):
    """One atomic, attributable factual assertion extracted from an article.

    Field shape per `notebooks/phase3_claim_extraction_prompt_review.md`'s
    data-scientist-authored proposal, paired with
    `llm/prompts/claim_extraction.txt` (Task 3.2.1a).
    """

    claim_text: str = Field(
        ...,
        description=(
            "One atomic factual assertion, in a single sentence, close to "
            "the article's own wording. Never a paraphrase of an entire "
            "paragraph, never multiple assertions joined by 'and'/'while'/"
            "semicolons -- split those into separate claims instead."
        ),
    )
    subject: str = Field(
        ...,
        description=(
            "The person, organization, or entity the claim is about (e.g. "
            "'the Federal Reserve', 'the plaintiff's attorneys'). Not the "
            "outlet reporting the claim."
        ),
    )
    attributed_source: str = Field(
        ...,
        description=(
            "Who the article credits this claim to, in the article's own "
            "words where possible (e.g. 'a Reuters analysis', 'the "
            "company's press release', 'unnamed White House officials'). "
            "Use exactly 'the article's own reporting' if the article "
            "states this as its own newsgathering rather than attributing "
            "it to another named source -- never invent an attribution "
            "the article doesn't state."
        ),
    )
    attribution_type: str = Field(
        ...,
        description=(
            "One of: 'reported' (the outlet's own newsgathering/verified "
            "fact), 'attributed' (sourced to a named person/org/document "
            "the article cites), 'alleged' (a claim one party makes that "
            "the article itself does not verify or that is disputed), or "
            "'opinion' (a stated viewpoint/interpretation, not a factual "
            "assertion)."
        ),
    )
    certainty: str = Field(
        ...,
        description=(
            "One of: 'confirmed' (stated as settled fact), 'developing' "
            "(reported as preliminary, ongoing, or subject to change -- "
            "articles using words like 'early reports', 'officials say "
            "they expect'), or 'disputed' (the article itself notes "
            "conflicting accounts or denial)."
        ),
    )


class ClaimList(BaseModel):
    """Structured output of the `claim_extraction.txt` prompt (Task 3.2.1b).

    One article -> one `ClaimList`. Downstream (Task 3.3.1) clusters
    `claim_text` across every article's `ClaimList` in a subtopic.
    """

    claims: list[Claim] = Field(
        ...,
        description="Every distinct atomic claim found in the article, deduplicated.",
    )
