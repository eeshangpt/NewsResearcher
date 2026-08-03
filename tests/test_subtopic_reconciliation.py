"""Unit tests for Task 2.2.3b (`reconcile_candidates`/`reconcile_subtopics`)
and Task 2.2.4 (`rank_and_cap_subtopics`).

Merge/split/drop fixtures are the data-scientist's committed
`tests/fixtures/reconciliation_{merge,split,drop}.json`, paired with
`clustering_synthetic_topics.json`'s real embeddings -- no live embedding
model call needed, per `notebooks/phase2-reconciliation-design.md`'s own
"How to use these fixtures" section.

The mixed 3-claimant case (a tech-lead-flagged gap in the design doc's own
fixtures) uses hand-constructed vectors with exact, checkable cosine
similarities rather than fixture data, so the merge-vs-split boundary is
exercised deterministically without relying on a real embedding model's
incidental geometry.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from newsresearch.agents.subtopic_agent import (
    rank_and_cap_subtopics,
    reconcile_candidates,
    reconcile_subtopics,
)
from newsresearch.config import Settings
from newsresearch.llm.schemas import SubtopicCandidate

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES_DIR / f"reconciliation_{name}.json").read_text())


def _clustering_fixture_vectors() -> np.ndarray:
    data = json.loads((FIXTURES_DIR / "clustering_synthetic_topics.json").read_text())
    return np.array(data["embeddings"])


def test_reconcile_merges_near_duplicate_candidates():
    fixture = _load("merge")
    article_vectors = _clustering_fixture_vectors()
    article_labels = np.array(fixture["article_true_labels"])
    cluster_ids = fixture["cluster_ids"]
    centroids = {cid: article_vectors[article_labels == cid].mean(axis=0) for cid in cluster_ids}

    result = reconcile_candidates(
        fixture["candidate_labels"],
        np.array(fixture["candidate_embeddings"]),
        cluster_ids,
        centroids,
        article_vectors,
        article_labels,
    )

    merges = [r for r in result["reconciled"] if r["action"] == "merge"]
    assert len(merges) == 1
    assert set(merges[0]["merged_from"]) == set(fixture["expected_outcome"]["merge"])
    assert merges[0]["article_count"] == int((article_labels == merges[0]["cluster_id"]).sum())

    single_labels = {r["label"] for r in result["reconciled"] if r["action"] == "single_match"}
    assert single_labels == set(fixture["expected_outcome"]["single_match"])
    assert result["dropped"] == []


def test_reconcile_splits_under_split_upstream_cluster():
    fixture = _load("split")
    article_vectors = _clustering_fixture_vectors()
    article_labels = np.array(fixture["article_cluster_labels"])
    cluster_ids = fixture["cluster_ids"]
    centroids = {cid: article_vectors[article_labels == cid].mean(axis=0) for cid in cluster_ids}

    result = reconcile_candidates(
        fixture["candidate_labels"],
        np.array(fixture["candidate_embeddings"]),
        cluster_ids,
        centroids,
        article_vectors,
        article_labels,
    )

    splits = [r for r in result["reconciled"] if r["action"] == "split"]
    assert {r["label"] for r in splits} == set(fixture["expected_outcome"]["split"])
    # Design doc: the synthetic 16-article merged cluster splits 8/8.
    assert sorted(r["article_count"] for r in splits) == [8, 8]
    assert sum(r["article_count"] for r in splits) == splits[0]["split_from_cluster_size"]

    single_labels = {r["label"] for r in result["reconciled"] if r["action"] == "single_match"}
    assert single_labels == set(fixture["expected_outcome"]["single_match"])


def test_reconcile_drops_unsupported_candidates():
    fixture = _load("drop")
    article_vectors = _clustering_fixture_vectors()
    article_labels = np.array(fixture["article_true_labels"])
    cluster_ids = fixture["cluster_ids"]
    centroids = {cid: article_vectors[article_labels == cid].mean(axis=0) for cid in cluster_ids}

    result = reconcile_candidates(
        fixture["candidate_labels"],
        np.array(fixture["candidate_embeddings"]),
        cluster_ids,
        centroids,
        article_vectors,
        article_labels,
    )

    dropped_labels = {d["candidate"] for d in result["dropped"]}
    assert dropped_labels == set(fixture["expected_outcome"]["dropped"])
    single_labels = {r["label"] for r in result["reconciled"] if r["action"] == "single_match"}
    assert single_labels == set(fixture["expected_outcome"]["single_match"])


def _unit(*coeffs: float) -> np.ndarray:
    return np.array(coeffs, dtype=float)


def test_reconcile_three_claimants_mixed_similarity_splits_two_from_one():
    """Tech-lead-flagged gap: the design doc's own fixtures only exercise
    2-claimant merge/split. Hand-built 5-dim vectors here give exact,
    checkable similarities: candidates A/B are near-duplicates of each other
    (cosine ~0.99) and of the shared cluster centroid; candidate C also
    claims the same cluster but is only loosely similar to A/B (~0.46,
    below `reconciliation_dup_threshold=0.65`) -- so the correct reconciled
    outcome is a 2-group split: {A, B} merged as one group, {C} alone as the
    other, not a 3-way merge or 3-way split.
    """
    e1, e2, e3, e4 = _unit(1, 0, 0, 0), _unit(0, 1, 0, 0), _unit(0, 0, 1, 0), _unit(0, 0, 0, 1)
    alpha, beta = 0.7, np.sqrt(1 - 0.7**2)
    theta = np.radians(10)
    candidate_a = alpha * e1 + beta * e2
    candidate_b = alpha * e1 + beta * (np.cos(theta) * e2 + np.sin(theta) * e3)
    gamma, delta = 0.65, np.sqrt(1 - 0.65**2)
    candidate_c = gamma * e1 + delta * e4
    candidate_embeddings = np.array([candidate_a, candidate_b, candidate_c])

    # Sanity-check the constructed geometry actually exercises the intended
    # merge-threshold boundary (>=0.65 for A/B, <0.65 for A/C and B/C).
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(candidate_a, candidate_b) >= 0.65
    assert cosine(candidate_a, candidate_c) < 0.65
    assert cosine(candidate_b, candidate_c) < 0.65

    # 4 member articles near the A/B direction, 2 near the C direction.
    theta5 = np.radians(5)
    articles = np.array(
        [
            candidate_a,
            candidate_b,
            (candidate_a + candidate_b) / 2,
            alpha * e1 + beta * (np.cos(theta5) * e2 + np.sin(theta5) * e3),
            candidate_c,
            candidate_c * 0.95 + _unit(0, 0, 0, 0.05),
        ]
    )
    article_labels = np.array([0, 0, 0, 0, 0, 0])
    centroids = {0: articles.mean(axis=0)}

    result = reconcile_candidates(
        ["Candidate A", "Candidate B", "Candidate C"],
        candidate_embeddings,
        [0],
        centroids,
        articles,
        article_labels,
    )

    assert result["dropped"] == []
    splits = [r for r in result["reconciled"] if r["action"] == "split"]
    assert len(splits) == 2, f"expected a 2-group split, got {result['reconciled']}"
    groups = {frozenset(r["merged_from"]) for r in splits}
    assert groups == {frozenset({"Candidate A", "Candidate B"}), frozenset({"Candidate C"})}
    ab_group = next(r for r in splits if len(r["merged_from"]) == 2)
    c_group = next(r for r in splits if len(r["merged_from"]) == 1)
    assert ab_group["article_count"] == 4
    assert c_group["article_count"] == 2


def test_reconcile_uses_relaxed_threshold_below_kmeans_fallback_threshold():
    """A candidate at 0.58 similarity to its best cluster: dropped under the
    healthy-pool `reconciliation_match_threshold=0.60`, kept under the
    relaxed `reconciliation_match_threshold_small_pool=0.55` -- but only when
    the article pool fed to `cluster()` is below
    `Settings.clustering.kmeans_fallback_threshold` (default 20).
    """
    e1, e2 = _unit(1, 0), _unit(0, 1)
    theta = np.radians(np.degrees(np.arccos(0.58)))  # candidate at exactly cos=0.58 to e1
    candidate = np.cos(theta) * e1 + np.sin(theta) * e2
    candidate_embeddings = np.array([candidate])
    centroids = {0: e1}

    small_pool_articles = np.tile(e1, (10, 1))  # n=10 < kmeans_fallback_threshold(20)
    small_labels = np.zeros(10, dtype=int)

    result_small = reconcile_candidates(
        ["Candidate"], candidate_embeddings, [0], centroids, small_pool_articles, small_labels
    )
    assert result_small["dropped"] == []
    assert len(result_small["reconciled"]) == 1

    large_pool_articles = np.tile(e1, (25, 1))  # n=25 >= kmeans_fallback_threshold(20)
    large_labels = np.zeros(25, dtype=int)

    result_large = reconcile_candidates(
        ["Candidate"], candidate_embeddings, [0], centroids, large_pool_articles, large_labels
    )
    assert result_large["reconciled"] == []
    assert len(result_large["dropped"]) == 1


def test_rank_and_cap_truncates_and_retains_excess():
    settings = Settings()
    total_articles = 40
    # 7 reconciled subtopics with distinct centroids and equal article count
    # -- more than the default max_subtopics=5, to exercise capping.
    reconciled = []
    for i in range(7):
        vec = np.zeros(7)
        vec[i] = 1.0
        reconciled.append(
            {
                "action": "single_match",
                "cluster_id": i,
                "label": f"subtopic-{i}",
                "merged_from": [f"subtopic-{i}"],
                "article_count": 5,
                "centroid": vec,
            }
        )

    result = rank_and_cap_subtopics(reconciled, total_articles, settings=settings)

    assert len(result["candidates"]) == settings.pipeline.max_subtopics
    assert len(result["excess"]) == 7 - settings.pipeline.max_subtopics
    all_labels = {r["label"] for r in result["candidates"]} | {r["label"] for r in result["excess"]}
    assert all_labels == {f"subtopic-{i}" for i in range(7)}
    for r in result["candidates"] + result["excess"]:
        assert "centroid" not in r
        assert "distinctiveness_score" in r


@patch("newsresearch.agents.subtopic_agent.embed")
@patch("newsresearch.agents.subtopic_agent.cluster")
def test_reconcile_subtopics_falls_back_to_unlabeled_clusters_when_all_candidates_dropped(
    mock_cluster, mock_embed
):
    """Regression for the "Iraq and WMDs"/"Taliban Takeover" zero-Gate-1-
    candidate live failures: a thin article batch where real clusters exist
    but every LLM-proposed candidate misses them (all dropped) must still
    produce >=1 Gate 1 candidate via the structural fallback, not an empty
    list.
    """
    articles = [{"title": f"Article {i}"} for i in range(4)]
    candidates = [SubtopicCandidate(label="Unrelated candidate", rationale="Doesn't match anything")]

    # 2 articles per cluster, 2 clusters -- real cluster structure exists.
    mock_cluster.return_value = np.array([0, 0, 1, 1])

    def _fake_embed(texts):
        if len(texts) == len(articles):
            return np.array(
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            )
        # Candidate embedding: orthogonal to both cluster centroids -- both
        # cosine similarities land at exactly 0, well below any configured
        # reconciliation threshold, so `reconcile_candidates` drops it.
        return np.array([[0.0, 0.0, 1.0]])

    mock_embed.side_effect = _fake_embed

    result = reconcile_subtopics(articles, candidates)

    assert result["reconciled"], "structural fallback must produce >=1 candidate, not empty"
    assert len(result["reconciled"]) == 2
    labels = {r["label"] for r in result["reconciled"]}
    assert all("unlabeled" in label for label in labels)
    assert {r["action"] for r in result["reconciled"]} == {"unlabeled_fallback"}
    assert sorted(r["article_count"] for r in result["reconciled"]) == [2, 2]


@patch("newsresearch.agents.subtopic_agent.embed")
@patch("newsresearch.agents.subtopic_agent.cluster")
def test_reconcile_subtopics_no_fallback_when_clustering_itself_finds_nothing(mock_cluster, mock_embed):
    """When `cluster()` itself finds zero real clusters (all noise, label
    -1), the structural fallback must NOT invent candidates -- empty
    `reconciled` stays correct, same as before this change.
    """
    articles = [{"title": f"Article {i}"} for i in range(4)]
    candidates = [SubtopicCandidate(label="Some candidate", rationale="rationale")]

    mock_cluster.return_value = np.array([-1, -1, -1, -1])
    mock_embed.return_value = np.array([[1.0, 0.0]] * 4)

    result = reconcile_subtopics(articles, candidates)

    assert result["reconciled"] == []


def test_rank_and_cap_single_subtopic_has_zero_avg_pairwise_distance():
    reconciled = [
        {
            "action": "single_match",
            "cluster_id": 0,
            "label": "only-subtopic",
            "merged_from": ["only-subtopic"],
            "article_count": 10,
            "centroid": np.array([1.0, 0.0]),
        }
    ]

    result = rank_and_cap_subtopics(reconciled, total_articles=10)

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["avg_pairwise_distance"] == 0.0
