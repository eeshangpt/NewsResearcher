import numpy as np
from sklearn.metrics import adjusted_rand_score

from newsresearch.agents.sentiment import score_article_sentiment, score_claim_sentiment
from newsresearch.clustering import claim_clustering
from newsresearch.clustering.cluster import cluster as real_cluster
from newsresearch.config import Settings
from newsresearch.llm.schemas import Claim


def _make_claim(claim_text: str) -> Claim:
    return Claim(
        claim_text=claim_text,
        subject="subject",
        attributed_source="the article's own reporting",
        attribution_type="reported",
        certainty="confirmed",
    )


def test_vader_labels_known_sentiment_samples():
    positive = score_article_sentiment("The team celebrated an outstanding, joyful victory.")
    negative = score_article_sentiment("The tragic disaster killed dozens and devastated the community.")
    neutral = score_article_sentiment("The meeting is scheduled for Tuesday at noon.")

    assert positive.sentiment_label == "positive"
    assert positive.sentiment_compound >= 0.05
    assert negative.sentiment_label == "negative"
    assert negative.sentiment_compound <= -0.05
    assert neutral.sentiment_label == "neutral"
    assert -0.05 < neutral.sentiment_compound < 0.05


def test_score_claim_sentiment_attaches_label_and_compound_per_claim():
    article_claims = {
        "a1": [_make_claim("The company posted record profits and thrilled investors.")],
        "a2": [_make_claim("The plane crash killed all passengers aboard.")],
    }
    result = score_claim_sentiment(article_claims)

    assert list(result.keys()) == ["a1", "a2"]
    assert result["a1"][0].claim is article_claims["a1"][0]
    assert result["a1"][0].sentiment_label == "positive"
    assert result["a2"][0].sentiment_label == "negative"
    assert isinstance(result["a2"][0].sentiment_compound, float)


def test_sentiment_does_not_affect_clustering(monkeypatch):
    """Task 3.4.1b's clustering-unaffected proof, per data-scientist's spec
    in `notebooks/phase3_sentiment_approach_review.md`: run `cluster_claims`
    twice on identical claims (once with no sentiment computed, once with
    sentiment computed exactly as production wiring would, before clustering
    runs) and assert the embeddings fed to `embed()`/`cluster()` and the
    resulting cluster assignments are identical -- and that no
    sentiment-derived text ever reaches the embedding step.
    """
    embed_calls: list[tuple[list[str], np.ndarray]] = []
    cluster_calls: list[np.ndarray] = []

    def fake_embed(texts):
        vectors = np.array(
            [[float(abs(hash(t)) % 1000), float(abs(hash(t[::-1])) % 1000)] for t in texts]
        )
        embed_calls.append((list(texts), vectors))
        return vectors

    def spy_cluster(vectors, **kwargs):
        labels = real_cluster(vectors, **kwargs)
        cluster_calls.append(labels)
        return labels

    monkeypatch.setattr(claim_clustering, "embed", fake_embed)
    monkeypatch.setattr(claim_clustering, "cluster", spy_cluster)

    article_claims = {
        "a1": [
            _make_claim("The mayor announced a new infrastructure plan."),
            _make_claim("Critics say the plan is underfunded."),
        ],
        "a2": [_make_claim("The city council approved the budget unanimously.")],
        "a3": [_make_claim("Residents protested the proposed tax increase.")],
    }
    expected_texts = [claim.claim_text for claims in article_claims.values() for claim in claims]

    # Run A: sentiment never computed.
    claim_clustering.cluster_claims(article_claims, settings=Settings())

    # Run B: sentiment computed first, exactly as production wiring would,
    # then clustering runs against the *same* article_claims dict.
    score_claim_sentiment(article_claims)
    claim_clustering.cluster_claims(article_claims, settings=Settings())

    assert len(embed_calls) == 2
    texts_a, vectors_a = embed_calls[0]
    texts_b, vectors_b = embed_calls[1]

    # No sentiment-derived text ever leaks into what gets embedded.
    assert texts_a == expected_texts
    assert texts_b == expected_texts

    # Embeddings fed to cluster() are byte-identical between runs.
    assert texts_a == texts_b
    assert np.array_equal(vectors_a, vectors_b)

    # Cluster assignments are identical between runs.
    assert len(cluster_calls) == 2
    assert adjusted_rand_score(cluster_calls[0], cluster_calls[1]) == 1.0
    assert list(cluster_calls[0]) == list(cluster_calls[1])
