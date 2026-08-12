"""Pydantic output schemas for structured LLM calls.

Call sites use `model.with_structured_output(Schema)` instead of hand-rolled
JSON parsing. Real schemas (subtopic proposals, claims, framing labels,
briefing sections, etc.) get added when the Phase 2+ agents that need them
are built.
"""

from __future__ import annotations

from typing import Literal

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


class ArticleFraming(BaseModel):
    """How ONE article presents ONE claim cluster.

    Deliberately keyed on `article_id`, not on the source domain: one domain
    can contribute several articles to a subtopic (in the real Leipzig-drone
    data, theguardian.com supplies both a news report and an opinion column),
    and blending them produces a false outlet-level bias signal. `domain` is
    carried alongside so downstream code can still roll up by domain when
    that is genuinely what it wants.
    """

    article_id: str = Field(
        ...,
        description=(
            "The article id exactly as given in this cluster's claim list. "
            "Never an id that does not appear in this cluster."
        ),
    )
    domain: str = Field(
        ...,
        description="The source domain shown next to that article id in the input.",
    )
    epistemic_treatment: Literal[
        "states_as_established",
        "attributes_to_named_party",
        "hedges_as_uncertain",
        "reports_as_contested",
    ] = Field(
        ...,
        description=(
            "How this article commits to the claim: asserted in its own "
            "voice as settled ('states_as_established'), relayed as what an "
            "identified party said without vouching for it "
            "('attributes_to_named_party'), qualified as possible/apparent/"
            "suspected/under investigation ('hedges_as_uncertain'), or "
            "presented alongside a denial or conflicting account "
            "('reports_as_contested')."
        ),
    )
    contextual_frame: str = Field(
        ...,
        description=(
            "Short descriptive phrase (roughly 4-12 words) naming the wider "
            "context this article places the claim in, or the aspect it "
            "foregrounds -- e.g. 'links incident to wider sabotage pattern', "
            "'keeps to procedural police response'. Describes what this "
            "article's own words do. Never a political leaning, ideological "
            "position, partisan alignment, quality rating, or comparison "
            "against another article."
        ),
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "A verbatim span copied from THIS article's claim text in THIS "
            "cluster -- a phrase, not the whole claim -- that the "
            "contextual_frame and epistemic_treatment are based on. Never a "
            "paraphrase, never another article's wording."
        ),
    )


class ClusterFraming(BaseModel):
    """Framing description for one claim cluster across its articles."""

    cluster_id: str = Field(
        ...,
        description="The cluster_id exactly as provided in this batch's input.",
    )
    cluster_coherence: Literal["single_shared_fact", "multiple_distinct_facts"] = Field(
        ...,
        description=(
            "Whether the grouped claims restate one underlying fact "
            "(allowing paraphrase/wire-rewrite differences) or contain "
            "several genuinely distinct facts that were grouped together. "
            "Upstream claim clustering is imperfect, so the latter is common."
        ),
    )
    shared_focus: str = Field(
        ...,
        description=(
            "One sentence naming what this cluster is actually about. When "
            "cluster_coherence is 'multiple_distinct_facts', name each "
            "distinct fact rather than only the most prominent one."
        ),
    )
    divergence: Literal[
        "single_article_only",
        "no_notable_divergence",
        "wording_or_emphasis_differs",
        "context_differs",
        "accounts_conflict",
    ] = Field(
        ...,
        description=(
            "Relationship between the articles' presentations. "
            "'accounts_conflict' is reserved for assertions about the same "
            "subject that cannot both be true -- two articles covering "
            "different facts are NOT in conflict."
        ),
    )
    divergence_note: str = Field(
        ...,
        description=(
            "One or two sentences explaining the divergence value, naming "
            "the specific article ids or domains involved and the actual "
            "differing wording. Never states a count of articles or sources "
            "-- names them instead."
        ),
    )
    per_article: list[ArticleFraming] = Field(
        ...,
        description=(
            "One entry per article id appearing in this cluster's claim "
            "list, and no others. Articles that omit the claim are NOT "
            "listed here -- omission is read deterministically from "
            "claim_cluster_articles, never generated by the model."
        ),
    )


class BiasFramingBatch(BaseModel):
    """Structured output of one `bias_framing.txt` call.

    One call covers `Settings.models.bias_framing_batch_size` clusters
    (default 8) from a single subtopic; each batch's `clusters` list merges
    additively into that subtopic's full result set.
    """

    clusters: list[ClusterFraming] = Field(
        ...,
        description="One entry per cluster given in this batch, keyed by cluster_id.",
    )
