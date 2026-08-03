import json
from pathlib import Path

import numpy as np

from newsresearch.clustering.claim_clustering import cluster_claims
from newsresearch.config import Settings
from newsresearch.llm.schemas import Claim

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase3_claim_clustering_corpus.json"


def _make_claim(claim_text: str) -> Claim:
    return Claim(
        claim_text=claim_text,
        subject="subject",
        attributed_source="the article's own reporting",
        attribution_type="reported",
        certainty="confirmed",
    )


def test_cluster_claims_uses_claim_specific_thresholds(monkeypatch):
    """Task 3.3.1b: claim clustering must pass `Settings.clustering.claim_*`
    into `cluster()`, not the article-level `hdbscan_*`/`kmeans_fallback_threshold`
    values -- asserted directly against the call, not inferred from output."""
    captured = {}

    def fake_cluster(vectors, k_hint=None, **kwargs):
        captured.update(kwargs)
        return np.zeros(len(vectors), dtype=int)

    monkeypatch.setattr("newsresearch.clustering.claim_clustering.cluster", fake_cluster)
    monkeypatch.setattr(
        "newsresearch.clustering.claim_clustering.embed",
        lambda texts: np.zeros((len(texts), 4)),
    )

    settings = Settings()
    article_claims = {
        "a1": [_make_claim("claim one"), _make_claim("claim two")],
        "a2": [_make_claim("claim three")],
    }
    cluster_claims(article_claims, settings=settings)

    assert captured["min_cluster_size"] == settings.clustering.claim_min_cluster_size
    assert captured["min_samples"] == settings.clustering.claim_min_samples
    assert captured["kmeans_fallback_threshold"] == settings.clustering.claim_kmeans_fallback_threshold
    assert captured["min_cluster_size"] != settings.clustering.hdbscan_min_cluster_size


def test_cluster_claims_recovers_assert_omit_membership(monkeypatch):
    def fake_cluster(vectors, k_hint=None, **kwargs):
        # First two claims (from a1, a2) cluster together; third (a3) is noise.
        return np.array([0, 0, -1])

    monkeypatch.setattr("newsresearch.clustering.claim_clustering.cluster", fake_cluster)
    monkeypatch.setattr(
        "newsresearch.clustering.claim_clustering.embed",
        lambda texts: np.zeros((len(texts), 4)),
    )

    article_claims = {
        "a1": [_make_claim("the model costs less to run")],
        "a2": [_make_claim("the model is cheaper to operate")],
        "a3": [_make_claim("unrelated claim")],
    }
    result = cluster_claims(article_claims, settings=Settings())

    cluster0 = result["clusters"][0]
    assert cluster0["asserting_article_ids"] == ["a1", "a2"]
    assert cluster0["omitting_article_ids"] == ["a3"]
    assert result["noise"] == [("a3", "unrelated claim")]
    assert result["total_claims"] == 3


def test_cluster_claims_empty_input():
    result = cluster_claims({})
    assert result == {"clusters": {}, "noise": [], "total_claims": 0}


def test_cluster_claims_real_corpus_same_fact_pairs_merge(monkeypatch):
    """Real-shaped regression using the data-scientist's Task 3.3.1a corpus
    (`notebooks/phase3_claim_clustering_review.md`, commit `f16c6d1`): a
    known same-fact duplicate pair (two outlets restating the same underlying
    fact, this corpus's dominant duplication shape) should land in the same
    cluster once run through the real `cluster()`/HDBSCAN path at the
    claim-specific thresholds -- a fake embedding stands in for a real
    embedding backend (no network/model dependency in this test), matching
    this test file's existing fake-embedding convention, but the rest of the
    120-claim corpus rides along unmodified as realistic background noise.
    """
    payload = json.loads(FIXTURE_PATH.read_text())
    dup_text_a = "duplicate fact stated one way"
    dup_text_b = "duplicate fact stated another way"

    article_claims = {}
    for article in payload:
        article_claims[article["url"]] = [_make_claim(c["claim_text"]) for c in article["claims"]]
    # Inject the known same-fact pair into two distinct articles.
    article_claims["dup_article_1"] = [_make_claim(dup_text_a)]
    article_claims["dup_article_2"] = [_make_claim(dup_text_b)]

    def fake_embed(texts):
        # Deterministic: the two known-duplicate texts share a distinguishing
        # vector so they land in the same cluster; everything else spreads
        # out by a hash-derived coordinate so it doesn't collide with them.
        vectors = []
        for text in texts:
            if text in (dup_text_a, dup_text_b):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, float(abs(hash(text)) % 1000)])
        return np.array(vectors)

    monkeypatch.setattr("newsresearch.clustering.claim_clustering.embed", fake_embed)

    result = cluster_claims(article_claims, settings=Settings())

    dup_cluster = next(
        c for c in result["clusters"].values() if "dup_article_1" in c["asserting_article_ids"]
    )
    assert "dup_article_2" in dup_cluster["asserting_article_ids"]
    assert result["total_claims"] == sum(a["n_claims"] for a in payload) + 2
