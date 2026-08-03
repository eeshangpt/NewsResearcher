"""Claim-text clustering across a subtopic's full claim set (Phase 3, Story 3.3, Task 3.3.1b).

Embeds `claim_text` (Task 3.2.1b's `Claim` schema) across *every* article in
a subtopic and clusters via `clustering/cluster.py::cluster()`, unchanged,
passing the dedicated `Settings.clustering.claim_min_cluster_size`/
`claim_min_samples`/`claim_kmeans_fallback_threshold` values per the
data-scientist's Task 3.3.1a recommendation
(`notebooks/phase3_claim_clustering_review.md`, commit `f16c6d1`) and
tech-lead's architecture decision that claim-level clustering not share
Phase 2's article-level hyperparameters.

Clustering never reads assert/omit membership directly -- it falls out of
which articles' claims land in a cluster (Story 3.3's acceptance line):
`asserting_article_ids` is every article with a claim in a given cluster,
`omitting_article_ids` is every other article in the subtopic.
"""

from __future__ import annotations

from typing import Any

from newsresearch.clustering.cluster import cluster
from newsresearch.clustering.embeddings import embed
from newsresearch.config import Settings
from newsresearch.llm.schemas import Claim


def cluster_claims(
    article_claims: dict[str, list[Claim]],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cluster claims across all articles in one subtopic.

    `article_claims` maps `article_id -> [Claim, ...]` for every article in
    the subtopic (including articles with zero extracted claims, so they're
    still counted as "omitting" every cluster).

    Returns `{"clusters": {cluster_id: {"claim_text": [...], "article_ids":
    [...], "asserting_article_ids": [...], "omitting_article_ids": [...]}},
    "noise": [(article_id, claim_text), ...], "total_claims"}`.
    """
    settings = settings or Settings()
    all_article_ids = list(article_claims.keys())

    flat: list[tuple[str, Claim]] = [
        (article_id, claim) for article_id, claims in article_claims.items() for claim in claims
    ]
    if not flat:
        return {"clusters": {}, "noise": [], "total_claims": 0}

    vectors = embed([claim.claim_text for _, claim in flat])
    labels = cluster(
        vectors,
        min_cluster_size=settings.clustering.claim_min_cluster_size,
        min_samples=settings.clustering.claim_min_samples,
        kmeans_fallback_threshold=settings.clustering.claim_kmeans_fallback_threshold,
    )

    clusters: dict[int, dict[str, Any]] = {}
    noise: list[tuple[str, str]] = []
    for (article_id, claim), cluster_id in zip(flat, labels.tolist()):
        if cluster_id == -1:
            noise.append((article_id, claim.claim_text))
            continue
        entry = clusters.setdefault(
            cluster_id, {"claim_text": [], "article_ids": [], "asserting_article_ids": set()}
        )
        entry["claim_text"].append(claim.claim_text)
        entry["article_ids"].append(article_id)
        entry["asserting_article_ids"].add(article_id)

    for entry in clusters.values():
        asserting = entry["asserting_article_ids"]
        entry["asserting_article_ids"] = sorted(asserting)
        entry["omitting_article_ids"] = sorted(set(all_article_ids) - asserting)

    return {"clusters": clusters, "noise": noise, "total_claims": len(flat)}
