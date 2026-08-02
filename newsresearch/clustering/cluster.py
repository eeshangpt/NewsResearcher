"""HDBSCAN-primary, KMeans-fallback clustering over embedding vectors.

Task 2.1.2b. Hyperparameters come from `Settings.clustering.*`
(`hdbscan_min_cluster_size`/`hdbscan_min_samples`/`kmeans_fallback_threshold`),
per the data-scientist's Task 2.1.2a evaluation
(`notebooks/phase2-clustering-recommendation.md`), re-validated against the
real standalone `hdbscan` package (TRD §3's named library) as required by
tech-lead's review note on that task -- see `config.py`'s `ClusteringSettings`
docstring for the resulting value changes.
"""

from __future__ import annotations

import logging

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from newsresearch.config import Settings

logger = logging.getLogger(__name__)


def _estimate_k(vectors: np.ndarray) -> int:
    """Pick a KMeans `k` via a silhouette-score sweep when no hint is given.

    Per the recommendation doc's flagged open question (no upstream candidate
    count exists for Task 2.5.1's per-subtopic clustering path, unlike Task
    2.2.3's Subtopic Agent candidate count) -- sweeps k=2..min(5, n-1) and
    picks the best-scoring k. Falls back to k=1 for n<=2, where KMeans/
    silhouette scoring aren't meaningful.
    """
    n = len(vectors)
    if n <= 2:
        return 1

    max_k = min(5, n - 1)
    if max_k < 2:
        return 1

    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vectors)
        try:
            score = silhouette_score(vectors, labels)
        except ValueError:
            # All points landed in a single label for this k -- undefined.
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster(vectors: np.ndarray, k_hint: int | None = None) -> np.ndarray:
    """Cluster `vectors` (shape `(n, dim)`), returning an `(n,)` label array.

    HDBSCAN is primary: density-based, discovers the cluster count itself,
    and marks ambiguous/outlier points as noise (label `-1`). Below
    `Settings.clustering.kmeans_fallback_threshold` vectors, HDBSCAN can't
    reliably find cluster structure (per the data-scientist's subsample
    sweep), so `cluster()` falls back to KMeans instead, forcing every point
    into one of `k_hint` clusters (or a silhouette-estimated `k` if no hint
    is supplied -- see `_estimate_k`).

    `k_hint`, when given, is a caller-supplied expected cluster count (e.g.
    the Subtopic Agent's LLM-proposed candidate count for Task 2.2.3) used
    only for the KMeans fallback path; HDBSCAN never takes a k hint since it
    discovers cluster count from density.
    """
    settings = Settings()
    n = len(vectors)
    if n == 0:
        return np.array([], dtype=int)

    if n < settings.clustering.kmeans_fallback_threshold:
        k = k_hint if k_hint is not None else _estimate_k(vectors)
        k = max(1, min(k, n))
        logger.info("cluster: n=%d below kmeans_fallback_threshold, using KMeans(k=%d)", n, k)
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        return model.fit_predict(vectors)

    model = HDBSCAN(
        min_cluster_size=settings.clustering.hdbscan_min_cluster_size,
        min_samples=settings.clustering.hdbscan_min_samples,
        metric="euclidean",
    )
    return model.fit_predict(vectors)
