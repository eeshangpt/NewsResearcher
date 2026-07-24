"""Phase 2 Task 2.1.2a — HDBSCAN vs. KMeans clustering prototyping/evaluation.

data-scientist offline analysis, per EXECUTION_PLAN.md's Phase 2 Role
Ownership & Parallelization Re-analysis. Not production code; lives on the
data-scientist's `feat/datascientist`-style scratch workspace (here, this
task's branch since the launching agent asked for a task-scoped branch).

What this does:
  1. Builds a synthetic multi-cluster embedding fixture: 4 distinct news
     topics, 8 headline-style sentences each, deliberately similar within a
     topic and different across topics. Embeds them via the *real*
     `newsresearch.llm.models.get_embeddings()` factory (local
     sentence-transformers backend -- no OPENAI_API_KEY needed, matches
     `Settings.embeddings.backend == "local"` default) so the evaluation
     reflects the actual embedding space the pipeline will use, not a
     hand-rolled stand-in.
  2. Sweeps HDBSCAN `min_cluster_size`/`min_samples` and scores against the
     known ground-truth topic labels (Adjusted Rand Index + noise-point
     count), using both `sklearn.cluster.HDBSCAN` (available in sklearn>=1.3,
     already a project dependency at 1.9.0) since the standalone `hdbscan`
     package is not yet installed in this environment (Task 2.0.1, not yet
     landed) -- noted explicitly, see recommendation doc.
  3. Subsamples the 32-point fixture down to progressively smaller article
     counts (28, 20, 14, 10, 8, 6, 5, 4) and re-runs HDBSCAN at the
     recommended settings to find the point where density-based clustering
     stops reliably recovering the known groups -- this directly informs
     `kmeans_fallback_threshold`.
  4. Runs KMeans (k = true number of clusters) at the same low-count samples
     as a comparison point.

Writes the fixture (embeddings + true labels + source sentences) to
`tests/fixtures/clustering_synthetic_topics.json` for a backend-engineer to
build a deterministic unit test against in Wave 2 (Task 2.1.2b).

No full article body text is used or persisted anywhere in this script --
only short synthetic headline-style strings, never real full-text pulled
from `sourcing/fulltext.py` (that module doesn't exist yet in Phase 2 and,
per the no-full-text-persistence rule, never will write to disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from newsresearch.llm.models import get_embeddings

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "clustering_synthetic_topics.json"

# 4 subtopics *within a single broad topic* ("AI regulation"), 8 headlines
# each. This is deliberately the harder, more representative case: real
# Subtopic Agent clustering operates on candidates that are all angles on one
# topic string, so within-topic subtopics are semantically closer to each
# other than four unrelated topics would be (a first fixture draft using four
# totally unrelated topics scored a trivial ARI=1.0 at every setting tested,
# including degenerate min_cluster_size=2 -- not a useful discriminator).
# `ambiguous_outliers` are headlines that plausibly touch more than one
# subtopic or use generic language -- included to see whether HDBSCAN
# correctly marks them as noise (-1) rather than being forced into a cluster,
# which KMeans always does since it has no noise concept.
TOPIC_SENTENCES: dict[str, list[str]] = {
    "eu_ai_act_enforcement": [
        "The EU AI Act's high-risk obligations take effect for large model providers.",
        "Brussels fines a tech firm for non-compliance with the EU AI Act.",
        "European regulators publish new guidance on EU AI Act enforcement.",
        "Companies scramble to meet EU AI Act compliance deadlines this quarter.",
        "The European Commission opens an investigation under the AI Act.",
        "EU AI Act penalties could reach millions of euros for repeat violations.",
        "National regulators coordinate on EU AI Act enforcement standards.",
        "A tech lobby group challenges an EU AI Act compliance ruling in court.",
    ],
    "us_executive_order_ai_safety": [
        "The White House issues an executive order on AI safety standards.",
        "US agencies begin drafting rules under the new AI safety executive order.",
        "The executive order requires AI labs to report safety test results.",
        "Industry groups react to the administration's AI safety directive.",
        "The executive order on AI safety faces pushback from some lawmakers.",
        "Federal agencies outline how they will implement the AI safety order.",
        "The White House says the AI safety executive order protects consumers.",
        "Critics argue the executive order on AI safety lacks enforcement teeth.",
    ],
    "ai_copyright_lawsuits": [
        "Authors sue an AI company over unauthorized use of copyrighted books.",
        "A federal judge allows an AI copyright lawsuit to proceed to trial.",
        "News publishers file a new copyright lawsuit against an AI startup.",
        "The AI copyright case could set precedent for training data use.",
        "Musicians join a growing wave of lawsuits over AI training data.",
        "A settlement is reached in a closely watched AI copyright dispute.",
        "Publishers argue AI models were trained on pirated copyrighted works.",
        "The copyright lawsuit against the AI firm heads to appeals court.",
    ],
    "state_level_ai_legislation": [
        "A state legislature passes a bill regulating AI use in hiring decisions.",
        "Several states advance AI legislation targeting deepfake political ads.",
        "The governor signs a state AI transparency law into effect.",
        "State lawmakers debate competing AI regulation bills this session.",
        "A new state law requires disclosure when AI is used in consumer service.",
        "Tech companies lobby against a proposed state AI liability bill.",
        "State-level AI legislation varies widely across the country this year.",
        "The state's AI oversight bill passes committee on a party-line vote.",
    ],
}

# Ambiguous/cross-cutting headlines -- not assigned a "true" subtopic label
# (recorded separately, excluded from ARI scoring) since they plausibly
# straddle more than one subtopic. Used only in the qualitative "how does
# each algorithm handle ambiguous input" note in the write-up.
AMBIGUOUS_OUTLIERS: list[str] = [
    "Lawmakers around the world are racing to regulate artificial intelligence.",
    "AI regulation is emerging as a major policy issue this election cycle.",
    "A think tank report compares AI regulatory approaches across jurisdictions.",
]


def build_fixture(include_outliers: bool = False) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    labels_text: list[str] = []
    sentences: list[str] = []
    for topic, sents in TOPIC_SENTENCES.items():
        for s in sents:
            labels_text.append(topic)
            sentences.append(s)
    if include_outliers:
        for s in AMBIGUOUS_OUTLIERS:
            labels_text.append("__ambiguous__")
            sentences.append(s)

    embeddings = get_embeddings()
    vectors = np.array(embeddings.embed_documents(sentences))

    topic_names = sorted(TOPIC_SENTENCES.keys())
    topic_to_int = {t: i for i, t in enumerate(topic_names)}
    # Ambiguous outliers get -1 as a sentinel "no true label" (never a real
    # cluster id in this fixture), consistent with HDBSCAN's own noise
    # convention -- kept distinguishable from a real 4th topic.
    true_labels = np.array([topic_to_int.get(t, -1) for t in labels_text])
    return vectors, true_labels, sentences, labels_text


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
    ari = adjusted_rand_score(true_labels, pred)
    return {"k": k, "ari": round(float(ari), 3)}


def main() -> None:
    print("Embedding synthetic fixture via get_embeddings() (local sentence-transformers)...")
    vectors, true_labels, sentences, labels_text = build_fixture()
    print(f"Fixture: {len(sentences)} sentences, {len(set(labels_text))} true topics, vector dim {vectors.shape[1]}")

    # --- Full 32-point fixture: HDBSCAN hyperparameter sweep ---
    print("\n=== HDBSCAN sweep on full 32-point fixture (4 topics x 8 sentences) ===")
    results = []
    for min_cluster_size in (2, 3, 4, 5, 6):
        for min_samples in (1, 2, 3):
            if min_samples > min_cluster_size:
                continue
            r = eval_hdbscan(vectors, true_labels, min_cluster_size, min_samples)
            results.append(r)
            print(r)

    # --- KMeans at true k=4 on full fixture, for comparison ---
    print("\n=== KMeans (k=4, known) on full 32-point fixture ===")
    kmeans_full = eval_kmeans(vectors, true_labels, k=4)
    print(kmeans_full)

    # --- Subsample sweep: find where HDBSCAN degrades as article count drops ---
    print("\n=== Subsample sweep: HDBSCAN at best full-fixture settings, decreasing N ===")
    # Use the best-scoring (min_cluster_size, min_samples) from the sweep above.
    best = max(results, key=lambda r: (r["ari"], -r["n_noise_points"]))
    print(f"Best full-fixture HDBSCAN settings: {best}")
    mcs, ms = best["min_cluster_size"], best["min_samples"]

    rng = np.random.default_rng(42)
    subsample_results = []
    for n_per_topic in (7, 5, 3, 2, 1):
        idx = []
        for topic_idx in range(4):
            topic_positions = np.where(true_labels == topic_idx)[0]
            chosen = rng.choice(topic_positions, size=min(n_per_topic, len(topic_positions)), replace=False)
            idx.extend(chosen.tolist())
        idx = np.array(sorted(idx))
        sub_vectors = vectors[idx]
        sub_labels = true_labels[idx]
        n_total = len(idx)

        hdb = eval_hdbscan(sub_vectors, sub_labels, mcs, ms)
        km = eval_kmeans(sub_vectors, sub_labels, k=4)
        subsample_results.append(
            {
                "n_articles_total": n_total,
                "n_per_topic": n_per_topic,
                "hdbscan_ari": hdb["ari"],
                "hdbscan_n_clusters_found": hdb["n_clusters_found"],
                "hdbscan_n_noise": hdb["n_noise_points"],
                "kmeans_ari": km["ari"],
            }
        )
        print(subsample_results[-1])

    # --- Qualitative: ambiguous outlier handling, HDBSCAN vs. KMeans ---
    print("\n=== Ambiguous-outlier handling (qualitative, full fixture + 3 outliers) ===")
    vectors_out, true_labels_out, sentences_out, labels_text_out = build_fixture(include_outliers=True)
    hdb_model = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean")
    hdb_pred_out = hdb_model.fit_predict(vectors_out)
    km_model = KMeans(n_clusters=4, n_init=10, random_state=42)
    km_pred_out = km_model.fit_predict(vectors_out)
    for i, (sent, true_lbl) in enumerate(zip(sentences_out, labels_text_out)):
        if true_lbl == "__ambiguous__":
            print(
                f"  outlier: {sent!r}\n"
                f"    HDBSCAN label: {hdb_pred_out[i]} ({'noise' if hdb_pred_out[i] == -1 else 'forced into a cluster'})"
                f" | KMeans label: {km_pred_out[i]} (always forced)"
            )

    # --- Persist fixture for backend-engineer's Wave 2 unit test ---
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture_payload = {
        "description": (
            "Synthetic 4-topic, 8-sentence-per-topic embedding fixture for "
            "clustering/cluster.py unit tests (Phase 2 Task 2.1.2b). Embeddings "
            "generated via newsresearch.llm.models.get_embeddings() with the "
            "local sentence-transformers backend "
            "(sentence-transformers/all-MiniLM-L6-v2, 384-dim). "
            "`true_labels` are the ground-truth topic indices (0-3, alphabetical "
            "by topic key) a correct clustering should recover."
        ),
        "topic_names": sorted(TOPIC_SENTENCES.keys()),
        "sentences": sentences,
        "true_labels": true_labels.tolist(),
        "embeddings": vectors.tolist(),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    FIXTURE_PATH.write_text(json.dumps(fixture_payload, indent=2))
    print(f"\nWrote fixture to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
