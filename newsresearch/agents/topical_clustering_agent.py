"""Topical Clustering Agent (Phase 2, Story 2.5, Task 2.5.1).

Given one Gate-1-approved subtopic, reuses Phase 1's `sourcing_agent`
(scoped to that subtopic's own label as the query keyword, not the parent
topic) to fetch its own article set, then embeds + coarse-clusters that set
via Story 2.1's `clustering/embeddings.py::embed` and
`clustering/cluster.py::cluster` -- the same primitives `subtopic_agent.py`'s
`reconcile_subtopics` already uses for the broad-set pass, applied here at
per-subtopic scope instead.

No `k_hint` is passed to `cluster()`: unlike Task 2.2.3's broad-set pass
(which has an LLM-proposed candidate count to hint with), there's no
upstream cluster-count estimate for a single subtopic's own article set --
`cluster()`'s silhouette-sweep fallback (`_estimate_k`) already handles this
per its own docstring.
"""

from __future__ import annotations

from typing import Any

from psycopg_pool import ConnectionPool

from newsresearch.agents.sourcing_agent import sourcing_agent
from newsresearch.clustering.cluster import cluster
from newsresearch.clustering.embeddings import embed
from newsresearch.config import Settings


def topical_clustering_agent(
    subtopic_id: str,
    label: str,
    lookback_days: int,
    *,
    pool: ConnectionPool | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Per-subtopic sourcing + coarse clustering (Task 2.5.1).

    Returns `{"subtopic_id", "label", "clusters": {cluster_id: [article,
    ...]}, "noise": [article, ...], "total_articles"}`. `clusters` maps each
    non-noise HDBSCAN/KMeans label to its member article dicts -- the
    cluster-id-to-articles shape Task 2.6.1's `reports/gate2_report.py` can
    aggregate directly into cluster sizes/sample headlines/source spread
    without needing to re-derive groupings. Noise points (HDBSCAN label
    `-1`, not produced by the KMeans fallback path) are kept separately
    rather than dropped, since Gate 2's human reviewer may still want to see
    what didn't cluster.
    """
    scored_articles = sourcing_agent([label], lookback_days, pool=pool, settings=settings)
    articles = [scored.article for scored in scored_articles]

    vectors = embed([a.get("title", "") for a in articles])
    labels = cluster(vectors)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for article, cluster_id in zip(articles, labels.tolist()):
        clusters.setdefault(int(cluster_id), []).append(article)
    noise = clusters.pop(-1, [])

    return {
        "subtopic_id": subtopic_id,
        "label": label,
        "clusters": clusters,
        "noise": noise,
        "total_articles": len(articles),
    }
