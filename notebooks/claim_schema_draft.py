"""Draft `Claim` schema for Task 3.2.1a -- ready to lift into `llm/schemas.py`.

Matches the existing file's docstring/`Field(...)` convention (see
`SubtopicCandidate`/`SubtopicCandidateList` in `llm/schemas.py`). Kept as a
standalone module here (rather than editing `llm/schemas.py` directly) since
this branch is analysis-only -- `backend-engineer` lifts this class body
into `llm/schemas.py` verbatim when wiring Task 3.2.1b.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """One atomic, attributable factual assertion extracted from an article.

    Field shape per `notebooks/phase3_claim_extraction_prompt_review.md`'s
    data-scientist-authored proposal, paired with
    `llm/prompts/claim_extraction.txt`. Extends the TRD/EXECUTION_PLAN's
    minimal `{claim_text, subject, attributed_source}` shape with two fields
    (`attribution_type`, `certainty`) found necessary against real article
    text -- see the write-up's Design Question 1 for the failure modes each
    one fixes.
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
    """Structured output of the `claim_extraction.txt` prompt (Task 3.2.1a).

    One article -> one `ClaimList`. Downstream (Task 3.3.1) clusters
    `claim_text` across every article's `ClaimList` in a subtopic.
    """

    claims: list[Claim] = Field(
        ...,
        description="Every distinct atomic claim found in the article, deduplicated.",
    )
