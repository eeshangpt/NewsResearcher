"""Task 4.1.1a -- proposed schema shape for the Bias & Framing Agent.

Paste-ready for `newsresearch/llm/schemas.py` (Task 4.1.1b wires it via
`get_chat_model("bias_framing").with_structured_output(BiasFramingBatch)`).
Kept here rather than edited into `schemas.py` directly because 4.1.1a is a
design deliverable -- `backend-engineer` owns the production edit.

Design notes live in `notebooks/phase4_bias_framing_prompt_review.md`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


def _assert_no_political_scalar_field() -> None:
    """The structural claim of Task 4.1.1a, as a runnable check.

    Fails if anyone later adds (a) a numeric field anywhere in the schema
    tree, or (b) an enum member or field name carrying a political-lean
    term. See `notebooks/phase4_bias_framing_prompt_review.md`.
    """
    import typing

    banned = {
        "left",
        "right",
        "center",
        "centre",
        "lean",
        "leaning",
        "liberal",
        "conservative",
        "progressive",
        "partisan",
        "ideology",
        "ideological",
        "bias_score",
        "political",
    }

    def walk(model: type[BaseModel], seen: set[type[BaseModel]]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            ann = field.annotation
            assert not any(b in name.lower() for b in banned), f"political field name: {name}"
            # No numeric field anywhere: a left/center/right scale's natural
            # encoding is a number, so the schema simply offers no numeric slot.
            assert ann not in (int, float), f"numeric field in framing schema: {name}"
            for member in typing.get_args(ann) if typing.get_origin(ann) is Literal else ():
                assert isinstance(member, str)
                assert not any(b in member.lower() for b in banned), f"political enum value: {member}"
            # Recurse into nested models (incl. list[Model]).
            for candidate in (ann, *typing.get_args(ann)):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    walk(candidate, seen)

    walk(BiasFramingBatch, set())


def _assert_enums_are_closed() -> None:
    """`Literal` really does reject a smuggled-in lean label, not just discourage it."""
    import pydantic

    ok = ArticleFraming(
        article_id="a1",
        domain="example.com",
        epistemic_treatment="hedges_as_uncertain",
        contextual_frame="plain factual statement, no wider framing",
        evidence_quote="under ongoing investigation",
    )
    assert ok.epistemic_treatment == "hedges_as_uncertain"

    for smuggled in ("left_leaning", "centre", "conservative", "3"):
        try:
            ArticleFraming(
                article_id="a1",
                domain="example.com",
                epistemic_treatment=smuggled,
                contextual_frame="x",
                evidence_quote="y",
            )
        except pydantic.ValidationError:
            continue
        raise AssertionError(f"schema accepted a political/scalar enum value: {smuggled}")


if __name__ == "__main__":
    _assert_no_political_scalar_field()
    _assert_enums_are_closed()
    print("ok: no numeric slot, no political field/enum name, enums reject smuggled values")
