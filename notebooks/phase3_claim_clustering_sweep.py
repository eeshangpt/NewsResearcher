"""Task 3.3.1a — ARI sweep for claim-text HDBSCAN/KMeans hyperparameters.

Loads the real 120-claim corpus (`phase3_claim_clustering_corpus.json`,
6 articles, single real subtopic -- see write-up for why only one subtopic
survived this session's GDELT sustained rate-limit block), builds a
manually-confirmed same-fact ground truth from the cosine-similarity
candidate pairs (`phase3_claim_pairs_openai_gpt5_release.json`, produced by
`phase3_claim_clustering_eval.py`), then runs the same ARI-sweep methodology
as Phase 2's Task 2.1.2a (`notebooks/phase2_clustering_eval.py`):
  1. Full-corpus HDBSCAN min_cluster_size/min_samples sweep vs. ARI.
  2. Subsample sweep (progressively fewer claims) to find where claim-text
     HDBSCAN degrades -- informs whether `kmeans_fallback_threshold` needs a
     dedicated claim variant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from newsresearch.clustering.embeddings import embed

NB_DIR = Path(__file__).resolve().parent
CORPUS_PATH = NB_DIR / "phase3_claim_clustering_corpus.json"
PAIRS_PATH = NB_DIR / "phase3_claim_pairs_openai_gpt5_release.json"

AUTO_ACCEPT_SIM = 0.90

# Manually reviewed (real, manual same-fact judgment -- see write-up)
# same-fact pairs in the 0.80-0.90 band that the auto-accept threshold
# would otherwise miss (just an added subordinate clause/extra detail, same
# underlying fact). Identified by (rounded claim_i text, rounded claim_j
# text) since pair indices differ between fetch runs; matched by text below.
MANUAL_ACCEPT_TEXT_PAIRS: list[tuple[str, str]] = [
    (
        "DeepSeek once commanded most of the headlines about Chinese AI development",
        "DeepSeek once commanded most of the headlines about Chinese AI development but was quickly besieged by many domestic rivals",
    ),
    (
        "DeepSeek's V4-Flash model scored the same as Google's Gemini 3.6 Flash",
        "DeepSeek's V4-Flash Intelligence Index score is the same as Google's Gemini 3.6 Flash's score.",
    ),
    (
        "Artificial Analysis estimated Anthropic's Claude Fable 5 costs US$3.15 per test",
        "Artificial Analysis estimated Claude Fable 5's average cost at $3.15 per test.",
    ),
    (
        "DeepSeek's V4-Flash model scored 50 out of 100 on Artificial Analysis's Intelligence Index",
        "Artificial Analysis said DeepSeek's V4-Flash model scored 50 out of 100 on its Intelligence Index, combining results from nine benchmarks spanning coding, reasoning and workplace-style assignments.",
    ),
    (
        "DeepSeek's V4-Flash model scored one point behind Meta's Muse Spark 1.1 and GLM-5.2 from Z.AI",
        "DeepSeek's V4-Flash is one point behind Meta's Muse Spark 1.1 and GLM-5.2 from Z.AI on the Intelligence Index.",
    ),
    (
        "DeepSeek's V4-Flash scored one point behind Meta's Muse Spark 1.1 and GLM-5.2 from Z.AI",
        "DeepSeek's V4-Flash is one point behind Meta's Muse Spark 1.1 and GLM-5.2 from Z.AI on the Intelligence Index.",
    ),
    (
        "DeepSeek's R1 model became a global sensation in early 2025",
        "DeepSeek's R1 model became a global sensation in early 2025, triggering a selloff in global technology stocks and raising questions about large amounts U.S. companies were spending on AI.",
    ),
    (
        "DeepSeek is preparing a more powerful version of its model called the V4-Pro",
        "DeepSeek is preparing a more powerful version of its model called the V4-Pro and has not given a date for its official release.",
    ),
    (
        "DeepSeek is preparing a more powerful version of its model, called the V4-Pro",
        "DeepSeek is preparing a more powerful version of its model called the V4-Pro and has not given a date for its official release.",
    ),
    (
        "DeepSeek's R1 model triggered a selloff in global technology stocks",
        "DeepSeek's R1 model became a global sensation in early 2025, triggering a selloff in global technology stocks and raising questions about large amounts U.S. companies were spending on AI.",
    ),
]
# Explicitly rejected (different underlying facts despite high cosine
# similarity -- shared vocabulary, not shared assertion): Qwen benchmark
# score pairs across *different* named benchmarks, and V4-Flash-release vs.
# V4-Pro-preparation pairs (two different product-status facts), and the
# two different GPT-5.6 Luna/Terra price-cut claims (different products,
# different %). Left out of MANUAL_ACCEPT_TEXT_PAIRS deliberately.


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


def build_ground_truth(claim_texts: list[str]) -> np.ndarray:
    pairs = json.loads(PAIRS_PATH.read_text())
    text_to_idx = {t: i for i, t in enumerate(claim_texts)}
    uf = UnionFind(len(claim_texts))

    n_auto = 0
    for p in pairs:
        if p["sim"] >= AUTO_ACCEPT_SIM:
            uf.union(p["i"], p["j"])
            n_auto += 1

    n_manual = 0
    for text_a, text_b in MANUAL_ACCEPT_TEXT_PAIRS:
        if text_a in text_to_idx and text_b in text_to_idx:
            uf.union(text_to_idx[text_a], text_to_idx[text_b])
            n_manual += 1

    roots = [uf.find(i) for i in range(len(claim_texts))]
    root_to_label = {}
    labels = []
    for r in roots:
        if r not in root_to_label:
            root_to_label[r] = len(root_to_label)
        labels.append(root_to_label[r])
    labels = np.array(labels)

    n_multi_clusters = sum(1 for lbl in set(labels) if (labels == lbl).sum() > 1)
    print(
        f"Ground truth: {n_auto} auto-accepted pairs (sim>={AUTO_ACCEPT_SIM}), "
        f"{n_manual} manually-confirmed pairs, -> {len(set(labels))} true clusters "
        f"({n_multi_clusters} multi-claim same-fact groups, rest singletons)"
    )
    return labels


def pairwise_recall_precision(true_labels: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Same-fact pair recall and false-merge precision.

    ARI penalizes HDBSCAN's single `-1` noise sentinel against a ground
    truth dominated by true singletons (78 true clusters, only 21 with >1
    member) -- two unrelated true-singleton claims both landing in noise
    counts as "same predicted cluster" under ARI's bookkeeping even though
    nothing was actually merged, which isn't the failure mode Task 3.3.1
    actually cares about. Pairwise metrics measure the real thing directly:
    - recall: of all true same-fact pairs (same non-singleton true label),
      what fraction share a predicted label (excluding pairs where either
      side is noise, since noise means "not merged with anything")?
    - precision: of all pairs sharing a non-noise predicted label, what
      fraction are true same-fact pairs (not a false merge of two distinct
      facts)?
    """
    n = len(true_labels)
    true_pair_true = 0
    true_pair_recovered = 0
    pred_pair_total = 0
    pred_pair_correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_true = true_labels[i] == true_labels[j]
            same_pred = pred[i] == pred[j] and pred[i] != -1
            if same_true:
                true_pair_true += 1
                if same_pred:
                    true_pair_recovered += 1
            if same_pred:
                pred_pair_total += 1
                if same_true:
                    pred_pair_correct += 1
    recall = true_pair_recovered / true_pair_true if true_pair_true else 1.0
    precision = pred_pair_correct / pred_pair_total if pred_pair_total else 1.0
    return round(recall, 3), round(precision, 3)


def eval_hdbscan(vectors, true_labels, min_cluster_size, min_samples) -> dict:
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean")
    pred = model.fit_predict(vectors)
    n_noise = int(np.sum(pred == -1))
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    ari = adjusted_rand_score(true_labels, pred)
    recall, precision = pairwise_recall_precision(true_labels, pred)
    return {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters_found": n_clusters,
        "n_noise_points": n_noise,
        "ari": round(float(ari), 3),
        "same_fact_recall": recall,
        "false_merge_precision": precision,
    }


def eval_kmeans(vectors, true_labels, k) -> dict:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred = model.fit_predict(vectors)
    recall, precision = pairwise_recall_precision(true_labels, pred)
    return {
        "k": k,
        "ari": round(float(adjusted_rand_score(true_labels, pred)), 3),
        "same_fact_recall": recall,
        "false_merge_precision": precision,
    }


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    claim_texts: list[str] = []
    for a in corpus:
        for c in a["claims"]:
            claim_texts.append(c["claim_text"])
    n = len(claim_texts)
    print(f"Corpus: {len(corpus)} articles, {n} claims")

    vectors = embed(claim_texts)
    true_labels = build_ground_truth(claim_texts)
    true_k = len(set(true_labels))

    print(f"\n=== Full corpus (n={n}) HDBSCAN sweep ===")
    results = []
    for mcs in (2, 3, 4, 5, 6):
        for ms in (1, 2, 3):
            if ms > mcs:
                continue
            r = eval_hdbscan(vectors, true_labels, mcs, ms)
            results.append(r)
            print(r)

    def f1(r):
        p, rec = r["false_merge_precision"], r["same_fact_recall"]
        return 2 * p * rec / (p + rec) if (p + rec) else 0.0

    for r in results:
        r["f1"] = round(f1(r), 3)
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest full-corpus setting (by same-fact pairwise F1): {best}")

    print(f"\n=== KMeans (k={true_k}, known true count) on full corpus ===")
    print(eval_kmeans(vectors, true_labels, k=true_k))

    print("\n=== Subsample sweep (best HDBSCAN setting), decreasing n ===")
    mcs, ms = best["min_cluster_size"], best["min_samples"]
    rng = np.random.default_rng(42)
    subsample_results = []
    for frac in (1.0, 0.75, 0.5, 0.33, 0.25, 0.15):
        n_sub = max(4, int(n * frac))
        idx = np.sort(rng.choice(n, size=n_sub, replace=False))
        sub_vectors = vectors[idx]
        sub_labels = true_labels[idx]
        sub_true_k = len(set(sub_labels))
        hdb = eval_hdbscan(sub_vectors, sub_labels, mcs, ms)
        km = eval_kmeans(sub_vectors, sub_labels, k=sub_true_k)
        row = {
            "n_claims": n_sub,
            "true_k": sub_true_k,
            "hdbscan_n_clusters_found": hdb["n_clusters_found"],
            "hdbscan_n_noise": hdb["n_noise_points"],
            "hdbscan_same_fact_recall": hdb["same_fact_recall"],
            "hdbscan_false_merge_precision": hdb["false_merge_precision"],
            "kmeans_same_fact_recall": km["same_fact_recall"],
            "kmeans_false_merge_precision": km["false_merge_precision"],
        }
        subsample_results.append(row)
        print(row)

    NB_DIR.joinpath("phase3_claim_clustering_sweep_results.json").write_text(
        json.dumps(
            {
                "n_claims": n,
                "true_k": true_k,
                "full_sweep": results,
                "best_full_setting": best,
                "kmeans_full": eval_kmeans(vectors, true_labels, k=true_k),
                "subsample_sweep": subsample_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()