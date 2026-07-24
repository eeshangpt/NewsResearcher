"""Phase 2 Task 2.2.3a -- reconciliation (merge/split/drop) rule prototyping.

data-scientist offline design/validation deliverable, per EXECUTION_PLAN.md's
Phase 2 Role Ownership & Parallelization Re-analysis (Task 2.2.3a). Not
production code -- lives on `feat/datascientist`. Handoff target:
`backend-engineer`, Task 2.2.3b.

Depends only on:
  - Task 2.2.1a's candidate-label shape (label + rationale strings) --
    already designed in `notebooks/phase2-subtopic-prompt-design.md`.
  - Task 2.1.2a's clustering *approach* (HDBSCAN primary / KMeans fallback)
    being settled -- it is; this script does not need the productionized
    `clustering/cluster.py` to exist (Task 2.1.2b is still in flight in
    parallel), only cluster *output shape*: a per-article integer label,
    -1 reserved for HDBSCAN noise, and a centroid computable per non-noise
    label. That shape is stood in for here using the *already-committed*
    `tests/fixtures/clustering_synthetic_topics.json` fixture's ground-truth
    `true_labels` as the "cluster assignment" (i.e. pretending a correctly
    working `cluster()` recovered the real 4 subtopics exactly, which
    Task 2.1.2a's own eval already showed HDBSCAN(min_cluster_size=4,
    min_samples=2) does closely -- ARI 0.781 -- on this same fixture). This
    keeps the reconciliation-rule validation decoupled from clustering
    algorithm noise, which is the right isolation: 2.2.3a's job is to
    validate the *reconciliation* logic given a cluster assignment, not to
    re-validate clustering itself (that's 2.1.2a's job, already done).

Reuses `newsresearch.llm.models.get_embeddings()` (local sentence-transformers
backend, no OPENAI_API_KEY needed) via the already-merged
`clustering/embeddings.py::embed()` wrapper, so this reflects the actual
embedding space the pipeline will use.

No full article text anywhere -- only short synthetic headline-style
sentences and short candidate labels/rationales, consistent with the
no-full-text-persistence rule.

Produces three fixture scenarios (merge / split / drop), each validated
independently against the rules written up in
`notebooks/phase2-reconciliation-design.md`, and writes them to
`tests/fixtures/reconciliation_*.json` for `backend-engineer`'s Wave 3
(Task 2.2.3b) unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from newsresearch.clustering.embeddings import embed

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
CLUSTERING_FIXTURE = FIXTURES_DIR / "clustering_synthetic_topics.json"

# --- Reconciliation thresholds under evaluation ------------------------
# (final recommended values are written up in the design doc; these are the
# candidate values this script empirically checks against the fixtures)
MATCH_THRESHOLD = 0.60  # candidate<->cluster-centroid cosine sim to "claim" a cluster
#
# Chosen from a real observed gap, not a guess: this script's DROP scenario
# below initially used MATCH_THRESHOLD=0.42, which incorrectly let a
# should-be-dropped candidate ("AI chip export control negotiations", sharing
# only broad-domain vocabulary with "AI regulation" but no real supporting
# cluster in this fixture) claim the state-legislation cluster at
# cosine=0.525. A follow-up probe against this same fixture showed a clean,
# wide gap between genuine single-cluster matches (0.776-0.908 across every
# true-label candidate tested) and same-broad-domain-but-unsupported
# candidates (0.363-0.525, including a deliberately different real AI-policy
# angle, "AI voice assistant privacy concerns", not just an obviously
# unrelated control like basketball, which scored ~0.0-0.13). 0.60 sits in
# the middle of that gap (0.525-0.776), so it drops the unsupported/loosely-
# domain-adjacent candidates while keeping every genuine match comfortably
# above threshold on this fixture.
CANDIDATE_DUP_THRESHOLD = 0.65  # candidate<->candidate cosine sim: same claim vs. distinct claim
#
# NOTE ON DESIGN ITERATION: an earlier version of this script decided
# merge-vs-split purely from the *per-article* vote split within a claimed
# cluster (a "dominant fraction" heuristic). Running it against real local
# embeddings (see git history / design doc "iteration" section) showed this
# heuristic is fragile: two genuinely near-duplicate candidate labels
# ("EU AI Act enforcement actions" / "European Union AI Act compliance
# crackdown", candidate-candidate cosine 0.828) still split a real 8-article
# cluster's per-article best-match 5/3 rather than one candidate dominating
# >=75% of members -- which would have wrongly triggered a SPLIT for what is
# clearly one story described two ways. Directly comparing the *claimant
# candidates to each other* (not just each to the cluster) is the much more
# robust signal: near-duplicate candidate labels are highly similar to each
# other (0.828 in that case) regardless of how individual member articles
# happen to vote; genuinely distinct candidates that both loosely match one
# coarse cluster are not (0.483 between "White House executive order on AI
# safety" and "State legislatures passing AI regulation bills" in the split
# fixture below). CANDIDATE_DUP_THRESHOLD=0.65 sits cleanly between these two
# real observed values (0.828 vs. 0.483), so it separates both constructed
# scenarios correctly -- see the design doc for the exact numbers.


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_centroid(vectors: np.ndarray, labels: np.ndarray, cluster_id: int) -> np.ndarray:
    return vectors[labels == cluster_id].mean(axis=0)


def candidate_text(label: str, rationale: str) -> str:
    # Matches how backend-engineer should embed a candidate: label + rationale
    # concatenated, since the label alone is often only 3-10 words and the
    # rationale carries real topical signal the bare label lacks.
    return f"{label}. {rationale}"


def load_clustering_fixture() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    data = json.loads(CLUSTERING_FIXTURE.read_text())
    vectors = np.array(data["embeddings"])
    true_labels = np.array(data["true_labels"])
    sentences = data["sentences"]
    topic_names = data["topic_names"]
    return vectors, true_labels, sentences, topic_names


def reconcile(
    candidate_labels: list[str],
    candidate_embeddings: np.ndarray,
    cluster_ids: list[int],
    centroids: dict[int, np.ndarray],
    article_vectors: np.ndarray,
    article_cluster_labels: np.ndarray,
) -> dict:
    """Reference implementation of the merge/split/drop reconciliation rules.

    Returns a dict describing which candidates were merged/split/dropped and
    the resulting reconciled subtopic list (label, cluster support, article
    count) -- this is the logic `backend-engineer` should port into
    `agents/subtopic_agent.py`'s reconciliation step (Task 2.2.3b).
    """
    n_candidates = len(candidate_labels)

    # Step 1: candidate -> cluster centroid similarity matrix.
    sim_matrix = np.zeros((n_candidates, len(cluster_ids)))
    for ci, cid in enumerate(cluster_ids):
        centroid = centroids[cid]
        for i in range(n_candidates):
            sim_matrix[i, ci] = cosine(candidate_embeddings[i], centroid)

    best_cluster_idx = sim_matrix.argmax(axis=1)
    best_cluster_sim = sim_matrix.max(axis=1)

    dropped = []
    claims: dict[int, list[int]] = {}  # cluster_id -> [candidate indices]
    for i in range(n_candidates):
        if best_cluster_sim[i] < MATCH_THRESHOLD:
            dropped.append({"candidate": candidate_labels[i], "best_sim": round(float(best_cluster_sim[i]), 3)})
            continue
        cid = cluster_ids[best_cluster_idx[i]]
        claims.setdefault(cid, []).append(i)

    reconciled = []
    for cid, claimant_idxs in claims.items():
        member_mask = article_cluster_labels == cid
        member_vectors = article_vectors[member_mask]
        n_members = int(member_mask.sum())

        if len(claimant_idxs) == 1:
            reconciled.append(
                {
                    "action": "single_match",
                    "cluster_id": int(cid),
                    "label": candidate_labels[claimant_idxs[0]],
                    "merged_from": [candidate_labels[claimant_idxs[0]]],
                    "article_count": n_members,
                }
            )
            continue

        # Multiple claimants on one cluster: decide merge vs. split by
        # directly comparing the claimant candidates to EACH OTHER (not by
        # how individual member articles vote -- see the design-iteration
        # note above CANDIDATE_DUP_THRESHOLD for why the article-vote
        # heuristic was replaced).
        claimant_embeds = candidate_embeddings[claimant_idxs]
        n_claim = len(claimant_idxs)
        pairwise = np.zeros((n_claim, n_claim))
        for a in range(n_claim):
            for b in range(n_claim):
                pairwise[a, b] = cosine(claimant_embeds[a], claimant_embeds[b])

        # Greedy single-linkage grouping: claimants whose pairwise cosine
        # similarity is >= CANDIDATE_DUP_THRESHOLD are the "same claim"
        # (near-duplicate proposals) and belong in one merge group.
        group_of = list(range(n_claim))  # union-find, initialized to singletons

        def find(x: int) -> int:
            while group_of[x] != x:
                group_of[x] = group_of[group_of[x]]
                x = group_of[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                group_of[ra] = rb

        for a in range(n_claim):
            for b in range(a + 1, n_claim):
                if pairwise[a, b] >= CANDIDATE_DUP_THRESHOLD:
                    union(a, b)

        groups: dict[int, list[int]] = {}
        for k in range(n_claim):
            groups.setdefault(find(k), []).append(k)

        if len(groups) == 1:
            # every claimant is a near-duplicate of every other -- merge all
            # into one subtopic, keyed to this cluster in full.
            canonical = claimant_idxs[int(np.argmax(sim_matrix[claimant_idxs, cluster_ids.index(cid)]))]
            reconciled.append(
                {
                    "action": "merge",
                    "cluster_id": int(cid),
                    "label": candidate_labels[canonical],
                    "merged_from": [candidate_labels[i] for i in claimant_idxs],
                    "article_count": n_members,
                    "claimant_pairwise_similarity": pairwise.round(3).tolist(),
                }
            )
        else:
            # >=2 distinct claim-groups map to one coarse cluster -- split
            # the cluster's articles by per-article closest claim-group
            # (using each group's best-matching candidate as representative).
            group_reps = {gid: members for gid, members in groups.items()}
            per_article_group = []
            for v in member_vectors:
                best_gid, best_sim = None, -2.0
                for gid, members in group_reps.items():
                    sims = [cosine(v, claimant_embeds[m]) for m in members]
                    m = max(sims)
                    if m > best_sim:
                        best_sim, best_gid = m, gid
                per_article_group.append(best_gid)
            per_article_group = np.array(per_article_group)

            for gid, members in group_reps.items():
                canonical_local = members[
                    int(np.argmax([sim_matrix[claimant_idxs[m], cluster_ids.index(cid)] for m in members]))
                ]
                sub_count = int(np.sum(per_article_group == gid))
                reconciled.append(
                    {
                        "action": "split",
                        "cluster_id": int(cid),
                        "label": candidate_labels[claimant_idxs[canonical_local]],
                        "merged_from": [candidate_labels[claimant_idxs[m]] for m in members],
                        "article_count": sub_count,
                        "split_from_cluster_size": n_members,
                        "claimant_pairwise_similarity": pairwise.round(3).tolist(),
                    }
                )

    return {
        "dropped": dropped,
        "reconciled": reconciled,
        "sim_matrix": sim_matrix.round(3).tolist(),
        "candidate_labels": candidate_labels,
        "cluster_ids": cluster_ids,
    }


def distinctiveness_score(
    reconciled: list[dict],
    centroids_by_label: dict[str, np.ndarray],
    total_articles: int,
) -> list[dict]:
    """Volume + distinctiveness ranking score, see design doc for the formula."""
    labels = [r["label"] for r in reconciled]
    scored = []
    for r in reconciled:
        label = r["label"]
        centroid = centroids_by_label[label]
        others = [centroids_by_label[o] for o in labels if o != label]
        if others:
            avg_dist = float(np.mean([1 - cosine(centroid, o) for o in others]))
        else:
            avg_dist = 0.0  # only one subtopic survived reconciliation -- no peer to compare against
        volume_norm = r["article_count"] / total_articles if total_articles else 0.0
        score = 0.5 * volume_norm + 0.5 * avg_dist
        scored.append({**r, "volume_norm": round(volume_norm, 3), "avg_pairwise_distance": round(avg_dist, 3), "distinctiveness_score": round(score, 3)})
    return sorted(scored, key=lambda x: -x["distinctiveness_score"])


def scenario_merge(vectors, true_labels, topic_names) -> None:
    print("\n=== SCENARIO 1: MERGE (two near-duplicate candidates -> one cluster) ===")
    # eu_ai_act_enforcement is topic index for "eu_ai_act_enforcement" (alphabetical in fixture)
    candidates = [
        ("EU AI Act enforcement actions", "Regulators fine and investigate firms for EU AI Act non-compliance."),
        ("European Union AI Act compliance crackdown", "Brussels and national regulators are enforcing AI Act rules against companies."),
        ("AI copyright lawsuits over training data", "Authors and publishers are suing AI firms over unauthorized use of copyrighted works."),
        ("US executive order on AI safety", "The White House issued an executive order setting AI safety standards for labs."),
    ]
    labels = [c[0] for c in candidates]
    cand_texts = [candidate_text(*c) for c in candidates]
    cand_embeds = embed(cand_texts)

    cluster_ids = sorted(set(int(x) for x in true_labels))
    centroids = {cid: cluster_centroid(vectors, true_labels, cid) for cid in cluster_ids}

    result = reconcile(labels, cand_embeds, cluster_ids, centroids, vectors, true_labels)
    print(json.dumps(result, indent=2))

    merge_actions = [r for r in result["reconciled"] if r["action"] == "merge"]
    assert len(merge_actions) == 1, "expected exactly one merge action"
    assert set(merge_actions[0]["merged_from"]) == {labels[0], labels[1]}, "expected the two EU AI Act candidates to merge"
    print("PASS: the two near-duplicate EU AI Act candidates merged into one subtopic as expected.")

    unclaimed = set(cluster_ids) - {r["cluster_id"] for r in result["reconciled"]}
    print(f"Unclaimed cluster ids (no candidate claimed them, topic={[topic_names[c] for c in unclaimed]}): {unclaimed}")

    return {
        "scenario": "merge",
        "candidate_labels": [c[0] for c in candidates],
        "candidate_rationales": [c[1] for c in candidates],
        "candidate_embeddings": cand_embeds.tolist(),
        "cluster_ids": cluster_ids,
        "article_true_labels": true_labels.tolist(),
        "expected_outcome": {
            "merge": [labels[0], labels[1]],
            "single_match": [labels[2], labels[3]],
            "dropped": [],
        },
    }


def scenario_split(vectors, true_labels, topic_names) -> None:
    print("\n=== SCENARIO 2: SPLIT (one synthetic merged cluster spans two candidates) ===")
    exec_idx = topic_names.index("us_executive_order_ai_safety")
    state_idx = topic_names.index("state_level_ai_legislation")

    # Simulate an under-split upstream cluster() call: pretend the executive-
    # order and state-legislation subtopics landed in ONE cluster (id 99)
    # instead of two, which is exactly the failure mode Task 2.2.3's split
    # rule exists to catch -- two genuinely distinct candidate-driven
    # subtopics whose article embeddings are close enough that clustering
    # merged them, but the LLM's candidate labels still distinguish them.
    merged_labels = true_labels.copy()
    merged_labels[(true_labels == exec_idx) | (true_labels == state_idx)] = 99
    # keep the copyright cluster distinct and untouched, as a control.

    candidates = [
        ("White House executive order on AI safety", "The administration issued an executive order requiring AI labs to report safety test results."),
        ("State legislatures passing AI regulation bills", "State lawmakers are advancing bills that regulate AI use in hiring and consumer services."),
        ("AI copyright lawsuits over training data", "Authors and publishers are suing AI firms over unauthorized use of copyrighted works."),
    ]
    labels = [c[0] for c in candidates]
    cand_embeds = embed([candidate_text(*c) for c in candidates])

    cluster_ids = sorted(set(int(x) for x in merged_labels))
    centroids = {cid: cluster_centroid(vectors, merged_labels, cid) for cid in cluster_ids}

    result = reconcile(labels, cand_embeds, cluster_ids, centroids, vectors, merged_labels)
    print(json.dumps(result, indent=2))

    split_actions = [r for r in result["reconciled"] if r["action"] == "split"]
    assert len(split_actions) == 2, f"expected 2 split actions, got {len(split_actions)}"
    assert {r["label"] for r in split_actions} == {labels[0], labels[1]}
    print("PASS: the merged 16-article cluster was split back into the two candidate-aligned subtopics.")

    return {
        "scenario": "split",
        "candidate_labels": labels,
        "candidate_rationales": [c[1] for c in candidates],
        "candidate_embeddings": cand_embeds.tolist(),
        "cluster_ids": cluster_ids,
        "article_cluster_labels": merged_labels.tolist(),
        "expected_outcome": {
            "split": [labels[0], labels[1]],
            "single_match": [labels[2]],
            "dropped": [],
        },
    }


def scenario_drop(vectors, true_labels, topic_names) -> None:
    print("\n=== SCENARIO 3: DROP (candidates with no supporting cluster) ===")
    candidates = [
        # Same broad domain (AI + government action) as the fixture's real
        # candidates but no supporting cluster in this article set at all --
        # the harder, more realistic drop case (a plausible-sounding subtopic
        # the LLM proposed that the broad fetch simply didn't surface any
        # articles for), not just an obviously off-domain control.
        ("AI chip export control negotiations", "Governments are negotiating restrictions on advanced AI chip exports to rival powers."),
        ("AI voice assistant privacy concerns", "Consumers worry smart speakers are recording conversations without consent."),
        ("EU AI Act enforcement actions", "Regulators fine and investigate firms for EU AI Act non-compliance."),
    ]
    labels = [c[0] for c in candidates]
    cand_embeds = embed([candidate_text(*c) for c in candidates])

    cluster_ids = sorted(set(int(x) for x in true_labels))
    centroids = {cid: cluster_centroid(vectors, true_labels, cid) for cid in cluster_ids}

    result = reconcile(labels, cand_embeds, cluster_ids, centroids, vectors, true_labels)
    print(json.dumps(result, indent=2))

    dropped_labels = {d["candidate"] for d in result["dropped"]}
    assert dropped_labels == {labels[0], labels[1]}, f"expected both no-support candidates dropped, got {dropped_labels}"
    print(f"PASS: both no-support candidates dropped (best_sim < {MATCH_THRESHOLD}), neither force-merged into an unrelated cluster.")

    return {
        "scenario": "drop",
        "candidate_labels": labels,
        "candidate_rationales": [c[1] for c in candidates],
        "candidate_embeddings": cand_embeds.tolist(),
        "cluster_ids": cluster_ids,
        "article_true_labels": true_labels.tolist(),
        "expected_outcome": {
            "dropped": [labels[0], labels[1]],
            "single_match": [labels[2]],
        },
    }


def scenario_distinctiveness(vectors, true_labels, topic_names) -> None:
    print("\n=== Distinctiveness-score demo (post-reconciliation ranking, uses SCENARIO 1's output) ===")
    candidates = [
        ("EU AI Act enforcement actions", "Regulators fine and investigate firms for EU AI Act non-compliance."),
        ("AI copyright lawsuits over training data", "Authors and publishers are suing AI firms over unauthorized use of copyrighted works."),
        ("US executive order on AI safety", "The White House issued an executive order setting AI safety standards for labs."),
        ("State legislatures passing AI regulation bills", "State lawmakers are advancing bills that regulate AI use in hiring and consumer services."),
    ]
    labels = [c[0] for c in candidates]
    cand_embeds = embed([candidate_text(*c) for c in candidates])
    cluster_ids = sorted(set(int(x) for x in true_labels))
    centroids = {cid: cluster_centroid(vectors, true_labels, cid) for cid in cluster_ids}
    result = reconcile(labels, cand_embeds, cluster_ids, centroids, vectors, true_labels)

    # Build label->centroid map for surviving subtopics (using the cluster
    # each landed on).
    centroids_by_label = {}
    for r in result["reconciled"]:
        centroids_by_label[r["label"]] = centroids[r["cluster_id"]]

    total_articles = len(vectors)
    ranked = distinctiveness_score(result["reconciled"], centroids_by_label, total_articles)
    for r in ranked:
        print(r)


def main() -> None:
    vectors, true_labels, sentences, topic_names = load_clustering_fixture()
    print(f"Loaded clustering fixture: {len(sentences)} sentences, topics={topic_names}")

    fixtures_out = {}
    fixtures_out["merge"] = scenario_merge(vectors, true_labels, topic_names)
    fixtures_out["split"] = scenario_split(vectors, true_labels, topic_names)
    fixtures_out["drop"] = scenario_drop(vectors, true_labels, topic_names)
    scenario_distinctiveness(vectors, true_labels, topic_names)

    for name, payload in fixtures_out.items():
        out_path = FIXTURES_DIR / f"reconciliation_{name}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
