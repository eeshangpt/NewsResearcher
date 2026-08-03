"""Task 3.3.1a — HDBSCAN hyperparameter sweep for claim-text clustering.

Same ARI-sweep methodology as Phase 2's Task 2.1.2a
(`notebooks/phase2_clustering_eval.py`/`phase2-clustering-recommendation.md`),
applied to real claim text instead of synthetic headlines.

Ground truth here is *not* subtopic identity (all claims in this corpus
already come from single-subtopic-scoped article pulls, matching Task
3.3.1's real input: one subtopic's full claim set) -- it's whether two claims
from different articles assert the *same underlying fact* (what Task 3.3.1's
clustering step actually needs to get right, so the Bias & Framing agent can
compare same-fact claims across sources for agreement/disagreement).

Ground-truth labels are derived semi-automatically: claim texts within one
subtopic are pairwise-cosine-compared, pairs above a high similarity bar are
manually read and confirmed (or rejected) as same-fact duplicates, and
confirmed pairs are merged via union-find into ground-truth cluster ids.
Every other claim is its own singleton "unique fact" cluster. This is a real,
if manual-in-the-loop, ground truth grounded in actual article overlap (three
of the six `openai_gpt5_release`-subtopic articles are wire-near-duplicate
coverage of the same DeepSeek cost story) -- not a golden dataset, and stated
as such: labels come from one data-scientist's manual same-fact judgment call
per pair, not a validated multi-rater eval (PRD Sec.7's named v1 gap).

Input: `notebooks/phase3_claim_clustering_corpus.json` (built by
`phase3_claim_clustering_fetch.py`). No article full text in this file or
this script -- claim_text only, consistent with the no-full-text-storage
rule throughout Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from newsresearch.clustering.embeddings import embed

CORPUS_PATH = Path(__file__).resolve().parent / "phase3_claim_clustering_corpus.json"
GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "phase3_claim_clustering_ground_truth.json"


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text())


def build_candidate_pairs(claim_texts: list[str], vectors: np.ndarray, threshold: float = 0.80) -> list[tuple[int, int, float]]:
    """Cosine-similarity candidate same-fact pairs, for manual confirmation."""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = norm @ norm.T
    n = len(claim_texts)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                pairs.append((i, j, float(sims[i, j])))
    return sorted(pairs, key=lambda p: -p[2])


def eval_hdbscan(vectors: np.ndarray, true_labels: np.ndarray, min_cluster_size: int, min_samples: int) -> dict:
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    pred = model.fit_predict(vectors)
    n_noise = int(np.sum(pred == -1))
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    ari = adjusted_rand_score(true_labels, pred)
    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters_found": n_clusters,
        "n_noise_points": n_noise,
        "ari": round(float(ari), 3),
    }


def eval_kmeans(vectors: np.ndarray, true_labels: np.ndarray, k: int) -> dict:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred = model.fit_predict(vectors)
    return {"k": k, "ari": round(float(adjusted_rand_score(true_labels, pred)), 3)}


def main() -> None:
    corpus = load_corpus()
    subtopics = sorted({a["subtopic"] for a in corpus})
    print(f"Corpus: {len(corpus)} articles across subtopics {subtopics}")

    for subtopic in subtopics:
        articles = [a for a in corpus if a["subtopic"] == subtopic]
        claim_texts: list[str] = []
        claim_article: list[str] = []
        for a in articles:
            for c in a["claims"]:
                claim_texts.append(c["claim_text"])
                claim_article.append(a["url"])
        n = len(claim_texts)
        print(f"\n=== Subtopic {subtopic!r}: {len(articles)} articles, {n} claims ===")

        vectors = embed(claim_texts)

        candidate_pairs = build_candidate_pairs(claim_texts, vectors, threshold=0.80)
        print(f"{len(candidate_pairs)} candidate same-fact pairs (cosine >= 0.80) for manual review:")
        for i, j, sim in candidate_pairs[:40]:
            print(f"  sim={sim:.3f}")
            print(f"    A [{claim_article[i]}]: {claim_texts[i]}")
            print(f"    B [{claim_article[j]}]: {claim_texts[j]}")

        out_dir = Path(__file__).resolve().parent
        (out_dir / f"phase3_claim_pairs_{subtopic}.json").write_text(
            json.dumps(
                [
                    {"i": i, "j": j, "sim": sim, "claim_i": claim_texts[i], "claim_j": claim_texts[j]}
                    for i, j, sim in candidate_pairs
                ],
                indent=2,
            )
        )
        (out_dir / f"phase3_claim_vectors_{subtopic}.json").write_text(
            json.dumps(
                {"claim_texts": claim_texts, "claim_article": claim_article, "vectors": vectors.tolist()},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()