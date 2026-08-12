import numpy as np
import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sentiment import score_claim_sentiment
from newsresearch.clustering.claim_clustering import cluster_claims
from newsresearch.config import Settings
from newsresearch.llm.schemas import Claim
from newsresearch.persistence import claim_clusters
from newsresearch.persistence.db import init_db

# Hermetic testcontainers, per tests/test_reputation_cache.py's precedent.


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
def pool(postgres_url):
    pool = init_db(postgres_url)
    yield pool
    pool.close()


def _make_claim(claim_text: str) -> Claim:
    return Claim(
        claim_text=claim_text,
        subject="subject",
        attributed_source="the article's own reporting",
        attribution_type="reported",
        certainty="confirmed",
    )


def _seed_subtopic_and_articles(pool, subtopic_id, article_ids):
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO subtopics (subtopic_id, label) VALUES (%s, %s)",
            (subtopic_id, "test subtopic"),
        )
        for article_id in article_ids:
            conn.execute(
                "INSERT INTO articles (article_id, subtopic_id, url) VALUES (%s, %s, %s)",
                (article_id, subtopic_id, f"https://example.com/{article_id}"),
            )


def test_write_claim_clusters_matches_in_memory_assignment(pool, monkeypatch):
    def fake_cluster(vectors, k_hint=None, **kwargs):
        # a1/a2 claims cluster together (label 0); a3's claim is noise.
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
    sentiments = score_claim_sentiment(article_claims)

    _seed_subtopic_and_articles(pool, "sub1", ["a1", "a2", "a3"])
    cluster_ids = claim_clusters.write_claim_clusters(pool, "sub1", result, claim_sentiments=sentiments)

    label, in_memory = next(iter(result["clusters"].items()))
    persisted_id = cluster_ids[label]
    rows = claim_clusters.read_cluster_article_relations(pool, persisted_id)

    asserting = sorted(r["article_id"] for r in rows if r["relation"] == "asserts")
    omitting = sorted(r["article_id"] for r in rows if r["relation"] == "omits")
    assert asserting == in_memory["asserting_article_ids"]
    assert omitting == in_memory["omitting_article_ids"]

    # claim_text + sentiment round-trip for asserting rows.
    by_article = {r["article_id"]: r for r in rows if r["relation"] == "asserts"}
    for article_id, expected_text in zip(in_memory["article_ids"], in_memory["claim_text"]):
        assert by_article[article_id]["claim_text"] == expected_text
        expected_cs = sentiments[article_id][0]
        assert by_article[article_id]["sentiment_label"] == expected_cs.sentiment_label
        assert by_article[article_id]["sentiment_compound"] == pytest.approx(expected_cs.sentiment_compound)

    # omitting rows carry no claim_text/sentiment.
    for r in rows:
        if r["relation"] == "omits":
            assert r["claim_text"] is None
            assert r["sentiment_label"] is None


def test_read_subtopic_cluster_ids_returns_only_that_subtopics_clusters(pool):
    """Task 4.1.1b's batching entrypoint reads its clusters through this."""
    _seed_subtopic_and_articles(pool, "sub-a", ["a10"])
    _seed_subtopic_and_articles(pool, "sub-b", ["a11"])
    with pool.connection() as conn:
        for subtopic_id, cluster_id in [
            ("sub-a", "sub-a:1"),
            ("sub-a", "sub-a:0"),
            ("sub-b", "sub-b:0"),
        ]:
            conn.execute(
                "INSERT INTO claim_clusters (cluster_id, subtopic_id) VALUES (%s, %s)",
                (cluster_id, subtopic_id),
            )

    assert claim_clusters.read_subtopic_cluster_ids(pool, "sub-a") == ["sub-a:0", "sub-a:1"]
    assert claim_clusters.read_subtopic_cluster_ids(pool, "sub-b") == ["sub-b:0"]
    assert claim_clusters.read_subtopic_cluster_ids(pool, "sub-missing") == []
