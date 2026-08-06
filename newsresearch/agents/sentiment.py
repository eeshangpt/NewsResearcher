"""Sentiment collector (Phase 3, Story 3.4, Task 3.4.1b).

Wires VADER lexicon sentiment per the data-scientist's `feat/datascientist`
recommendation (`notebooks/phase3_sentiment_approach_review.md`, commit
`b72d53d`): deterministic, near-zero-marginal-cost lexicon scoring rather
than a small-model LLM call, since FR-14 marks sentiment as an auxiliary,
descriptive attribute -- never a clustering axis, never a bias proxy.

3-class label (positive/neutral/negative) via VADER's own default compound
thresholds (`>= 0.05` positive, `<= -0.05` negative, else neutral), with the
raw compound score kept alongside for future finer-grained use. Claim-level
sentiment is the primary attribute (`score_claim_sentiment`); article-level
is a rougher secondary signal (`score_article_sentiment`) -- the write-up
found whole-article lexicon scoring can be dominated by surface-word
polarity in ways claim-level scoring avoids (its HIV/USAID-funding-cuts
example: article compound +0.98 despite a mixed/negative overall framing).

Attaches sentiment via a thin wrapper dataclass around the existing `Claim`,
matching `agents/sourcing_agent.py::ScoredArticle`'s wrap-don't-mutate
pattern, rather than adding fields to `Claim` itself (an LLM structured-
output schema) or inventing a new shape. This module never mutates or reads
back from `clustering/claim_clustering.py`'s `article_claims` input -- it is
a sibling computation over the same `Claim` objects, called independently of
clustering (see `tests/test_sentiment.py` for the proof).
"""

from __future__ import annotations

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from newsresearch.llm.schemas import Claim

_analyzer = SentimentIntensityAnalyzer()


def _label_for(compound: float) -> str:
    """VADER's own documented default 3-class thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


@dataclass(frozen=True)
class ClaimSentiment:
    """One claim plus its VADER sentiment (label + raw compound score)."""

    claim: Claim
    sentiment_label: str
    sentiment_compound: float


@dataclass(frozen=True)
class ArticleSentiment:
    """Whole-article-text VADER sentiment -- secondary/rougher signal (see module docstring)."""

    sentiment_label: str
    sentiment_compound: float


def score_claim_sentiment(article_claims: dict[str, list[Claim]]) -> dict[str, list[ClaimSentiment]]:
    """Score every claim's `claim_text` via VADER.

    Primary sentiment signal (Story 3.4). Kept in the same `article_id ->
    [...]` shape `clustering/claim_clustering.py::cluster_claims` consumes,
    so callers can attach sentiment without reshaping their input -- but this
    function never calls into clustering and clustering never calls into
    this function.
    """
    result: dict[str, list[ClaimSentiment]] = {}
    for article_id, claims in article_claims.items():
        scored = []
        for claim in claims:
            compound = _analyzer.polarity_scores(claim.claim_text)["compound"]
            scored.append(
                ClaimSentiment(
                    claim=claim,
                    sentiment_label=_label_for(compound),
                    sentiment_compound=compound,
                )
            )
        result[article_id] = scored
    return result


def score_article_sentiment(article_text: str) -> ArticleSentiment:
    """Score one article's full body text via VADER.

    Secondary signal only -- data-scientist's write-up found whole-document
    lexicon scoring can badly misjudge mixed/negative articles when surface
    positive words dominate. Prefer `score_claim_sentiment` wherever
    claim-level text is available.
    """
    compound = _analyzer.polarity_scores(article_text)["compound"]
    return ArticleSentiment(sentiment_label=_label_for(compound), sentiment_compound=compound)