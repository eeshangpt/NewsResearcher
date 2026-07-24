import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from newsresearch.clustering.cluster import cluster
from newsresearch.config import Settings

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "clustering_synthetic_topics.json"


@pytest.fixture(scope="module")
def fixture_data():
    payload = json.loads(FIXTURE_PATH.read_text())
    vectors = np.array(payload["embeddings"])
    true_labels = np.array(payload["true_labels"])
    return vectors, true_labels


def test_cluster_uses_hdbscan_above_threshold(fixture_data):
    vectors, true_labels = fixture_data
    settings = Settings()
    assert len(vectors) >= settings.clustering.kmeans_fallback_threshold

    labels = cluster(vectors)

    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    assert n_clusters == 4
    ari = adjusted_rand_score(true_labels, labels)
    assert ari >= 0.7, f"expected HDBSCAN to recover the 4 known subtopics (ARI={ari})"


def test_cluster_below_threshold_uses_kmeans(fixture_data):
    vectors, true_labels = fixture_data
    settings = Settings()

    # Subsample to below `kmeans_fallback_threshold` (20): 2 points per each
    # of the 4 known subtopics = 8 total, well under the threshold.
    idx = []
    for topic_idx in range(4):
        positions = np.where(true_labels == topic_idx)[0][:2]
        idx.extend(positions.tolist())
    idx = np.array(sorted(idx))
    sub_vectors, sub_labels = vectors[idx], true_labels[idx]
    assert len(sub_vectors) < settings.clustering.kmeans_fallback_threshold

    labels = cluster(sub_vectors, k_hint=4)

    # KMeans force-assigns every point -- no noise label (-1), unlike HDBSCAN.
    assert -1 not in labels
    assert len(set(labels.tolist())) == 4
    ari = adjusted_rand_score(sub_labels, labels)
    assert ari == 1.0, f"expected KMeans(k=4) to perfectly recover known subtopics (ARI={ari})"


def test_cluster_empty_input_returns_empty_array():
    labels = cluster(np.empty((0, 4)))
    assert labels.shape == (0,)


def test_cluster_below_threshold_estimates_k_without_hint(fixture_data):
    vectors, true_labels = fixture_data
    idx = []
    for topic_idx in range(4):
        positions = np.where(true_labels == topic_idx)[0][:2]
        idx.extend(positions.tolist())
    idx = np.array(sorted(idx))
    sub_vectors = vectors[idx]

    labels = cluster(sub_vectors)

    assert -1 not in labels
    assert 1 <= len(set(labels.tolist())) <= min(5, len(sub_vectors) - 1)
