"""Subtopic Agent (Phase 2, Story 2.2, TRD's `agents/subtopic_agent.py`).

This module is populated incrementally across Phase 2's Wave 1-3 tasks:
propose candidate subtopics (this file's `propose_candidates`, Task 2.2.1b,
LLM-driven), broad topic-scoped fetch (`broad_topic_fetch`, Task 2.2.2),
embed+cluster+reconcile the broad set (`reconcile_candidates`/
`reconcile_subtopics`, Task 2.2.3b, implementing the data-scientist's
`notebooks/phase2-reconciliation-design.md`), and rank/cap/excess-retention
(`rank_and_cap_subtopics`, Task 2.2.4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from psycopg_pool import ConnectionPool

from newsresearch.agents.sourcing_agent import sourcing_agent
from newsresearch.clustering.cluster import cluster
from newsresearch.clustering.embeddings import embed
from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.llm.schemas import SubtopicCandidate, SubtopicCandidateList
from newsresearch.observability.langfuse_setup import get_langfuse_callback_handler, trace_metadata

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"
logger = logging.getLogger(__name__)


def propose_candidates(
    topic: str,
    n_candidates: int = 8,
    *,
    run_id: str = "dev",
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> SubtopicCandidateList:
    """Propose `n_candidates` candidate subtopics for `topic` (Task 2.2.1b).

    Wires the data-scientist-authored `llm/prompts/subtopic_propose.txt`
    (Task 2.2.1a, see `notebooks/phase2-subtopic-prompt-design.md`) through
    `get_chat_model("subtopic").with_structured_output(SubtopicCandidateList)`
    per NFR-6's single-factory convention. `n_candidates` defaults to 8 per
    the design doc's recommendation (`Settings.pipeline.max_subtopics + 3`
    given the default `max_subtopics=5`) -- not itself a `Settings` field
    today, left as a caller-supplied parameter per the prompt design's intent
    that this is a starting recommendation, not a tuned config value.

    The returned `candidates` list may be shorter than `n_candidates` (the
    prompt's rule 6 lets the model propose fewer rather than pad with
    near-duplicates/filler) -- not treated as an error here.

    Traced via Langfuse per the CLI's established `get_langfuse_callback_handler`
    + `trace_metadata` convention, tagged with `run_id` and `stage=subtopic`.
    Accepts an ambient `config` (e.g. forwarded from a LangGraph node, per
    `graph/build.py::_make_subtopic_stub_node`'s established pattern) and
    merges it with the Langfuse callback/metadata this call attaches, via
    `merge_configs`, rather than replacing it outright -- so any
    already-attached callbacks (e.g. `cost_callback.py`'s handler, per its
    own "no per-agent instrumentation code is needed" propagation claim)
    still fire on this nested LLM call.
    """
    settings = settings or Settings()

    template_text = (_PROMPTS_DIR / "subtopic_propose.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("subtopic").with_structured_output(SubtopicCandidateList)
    chain = prompt | model

    call_config: RunnableConfig = {
        "callbacks": [get_langfuse_callback_handler(settings)],
        "metadata": {**trace_metadata(run_id), "stage": "subtopic"},
    }
    merged_config = merge_configs(config, call_config)
    return chain.invoke({"topic": topic, "n_candidates": n_candidates}, config=merged_config)


def broad_topic_fetch(
    topic: str,
    lookback_days: int,
    *,
    pool: ConnectionPool | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Broad, topic-scoped fetch reusing Phase 1's `sourcing_agent` (Task
    2.2.2) -- not yet subtopic-filtered, since subtopics aren't known this
    early in the pipeline (that filtering happens once Task 2.2.1's
    candidate proposal and Task 2.2.3's reconciliation exist).

    The `topic` string itself is used as a single keyword -- deliberately no
    tokenization/keyword-expansion logic here, since a broad single-keyword
    query against `sourcing_agent`'s GDELT+RSS(+backfill) sources is already
    broad enough to surface multiple candidate-subtopic angles (e.g.
    "renewable energy" plausibly surfacing solar/wind/policy/storage
    articles), and inventing a smarter keyword-derivation strategy would be
    a query-design decision outside this task's scope.

    Returns raw article dicts (not `ScoredArticle` wrappers) -- downstream
    embedding/clustering (Task 2.2.3) operates on article text/title, not
    reputation scores.
    """
    scored_articles = sourcing_agent([topic], lookback_days, pool=pool, settings=settings)
    return [scored.article for scored in scored_articles]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _cluster_centroid(vectors: np.ndarray, labels: np.ndarray, cluster_id: int) -> np.ndarray:
    return vectors[labels == cluster_id].mean(axis=0)


def _candidate_text(label: str, rationale: str) -> str:
    # Rationale carries real topical signal the bare label (often 3-10 words)
    # lacks -- matches the design doc's embedding convention exactly.
    return f"{label}. {rationale}"


def reconcile_candidates(
    candidate_labels: list[str],
    candidate_embeddings: np.ndarray,
    cluster_ids: list[int],
    centroids: dict[int, np.ndarray],
    article_vectors: np.ndarray,
    article_cluster_labels: np.ndarray,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Port of `notebooks/phase2_reconciliation_eval.py::reconcile()` (Task
    2.2.3a's reference implementation) into production code, Task 2.2.3b.

    Merge/split/drop rules and thresholds exactly per
    `notebooks/phase2-reconciliation-design.md`: a candidate claims its
    highest-similarity cluster if that similarity is >=
    `reconciliation_match_threshold` (else dropped); when 2+ candidates claim
    the same cluster, they're grouped by mutual candidate-candidate
    similarity (union-find, >= `reconciliation_dup_threshold` links them) --
    one resulting group merges into a single subtopic backed by the cluster's
    full article count, 2+ groups split the cluster's articles by
    per-article best-matching group.

    Clusters no candidate claims are dropped entirely (logged by the caller,
    `reconcile_subtopics`), per the design doc's explicit "Unclaimed
    clusters" call.

    Each item in the returned `reconciled` list carries a `"centroid"` (the
    full cluster centroid for merge/single-match subtopics, the assigned
    article subset's centroid for split subtopics -- per the design doc's
    distinctiveness-formula centroid definition) for Task 2.2.4's ranking.
    """
    settings = settings or Settings()
    match_threshold = settings.clustering.reconciliation_match_threshold
    dup_threshold = settings.clustering.reconciliation_dup_threshold

    n_candidates = len(candidate_labels)
    sim_matrix = np.zeros((n_candidates, len(cluster_ids)))
    for ci, cid in enumerate(cluster_ids):
        centroid = centroids[cid]
        for i in range(n_candidates):
            sim_matrix[i, ci] = _cosine(candidate_embeddings[i], centroid)

    best_cluster_idx = sim_matrix.argmax(axis=1)
    best_cluster_sim = sim_matrix.max(axis=1)

    dropped = []
    claims: dict[int, list[int]] = {}
    for i in range(n_candidates):
        if best_cluster_sim[i] < match_threshold:
            dropped.append({"candidate": candidate_labels[i], "best_sim": round(float(best_cluster_sim[i]), 3)})
            continue
        cid = cluster_ids[best_cluster_idx[i]]
        claims.setdefault(cid, []).append(i)

    reconciled: list[dict[str, Any]] = []
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
                    "centroid": centroids[cid],
                }
            )
            continue

        # 2+ claimants of one cluster: group by mutual candidate-candidate
        # similarity (not per-article voting, per the design doc's own
        # documented iteration away from that fragile heuristic).
        claimant_embeds = candidate_embeddings[claimant_idxs]
        n_claim = len(claimant_idxs)
        pairwise = np.zeros((n_claim, n_claim))
        for a in range(n_claim):
            for b in range(n_claim):
                pairwise[a, b] = _cosine(claimant_embeds[a], claimant_embeds[b])

        group_of = list(range(n_claim))  # union-find, singletons initially

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
                if pairwise[a, b] >= dup_threshold:
                    union(a, b)

        groups: dict[int, list[int]] = {}
        for k in range(n_claim):
            groups.setdefault(find(k), []).append(k)

        if len(groups) == 1:
            # Every claimant collapses into one claim-group -- merge.
            canonical = claimant_idxs[int(np.argmax(sim_matrix[claimant_idxs, cluster_ids.index(cid)]))]
            reconciled.append(
                {
                    "action": "merge",
                    "cluster_id": int(cid),
                    "label": candidate_labels[canonical],
                    "merged_from": [candidate_labels[i] for i in claimant_idxs],
                    "article_count": n_members,
                    "centroid": centroids[cid],
                }
            )
        else:
            # 2+ distinct claim-groups on one cluster -- split the cluster's
            # articles by per-article closest claim-group.
            per_article_group = []
            for v in member_vectors:
                best_gid, best_sim = None, -2.0
                for gid, members in groups.items():
                    sim = max(_cosine(v, claimant_embeds[m]) for m in members)
                    if sim > best_sim:
                        best_sim, best_gid = sim, gid
                per_article_group.append(best_gid)
            per_article_group = np.array(per_article_group)

            for gid, members in groups.items():
                canonical_local = members[
                    int(np.argmax([sim_matrix[claimant_idxs[m], cluster_ids.index(cid)] for m in members]))
                ]
                group_mask = per_article_group == gid
                sub_count = int(group_mask.sum())
                reconciled.append(
                    {
                        "action": "split",
                        "cluster_id": int(cid),
                        "label": candidate_labels[claimant_idxs[canonical_local]],
                        "merged_from": [candidate_labels[claimant_idxs[m]] for m in members],
                        "article_count": sub_count,
                        "split_from_cluster_size": n_members,
                        "centroid": member_vectors[group_mask].mean(axis=0)
                        if sub_count
                        else centroids[cid],
                    }
                )

    return {"dropped": dropped, "reconciled": reconciled}


def reconcile_subtopics(
    articles: list[dict[str, Any]],
    candidates: list[SubtopicCandidate],
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Combine `broad_topic_fetch()`'s articles, `propose_candidates()`'s
    candidate list, and `cluster()`'s output into a reconciled subtopic list
    (Task 2.2.3b's orchestration entrypoint).

    Embeds article titles and candidate `label + rationale` text, clusters
    the article embeddings (`k_hint=len(candidates)`), then applies
    `reconcile_candidates`'s merge/split/drop rules. Unclaimed clusters (no
    candidate claims them) are dropped per the design doc's explicit call --
    logged here for visibility, not force-labeled.
    """
    settings = settings or Settings()
    vectors = embed([a.get("title", "") for a in articles])
    labels = cluster(vectors, k_hint=len(candidates))

    cluster_ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    centroids = {cid: _cluster_centroid(vectors, labels, cid) for cid in cluster_ids}

    candidate_labels = [c.label for c in candidates]
    candidate_embeddings = embed([_candidate_text(c.label, c.rationale) for c in candidates])

    result = reconcile_candidates(
        candidate_labels,
        candidate_embeddings,
        cluster_ids,
        centroids,
        vectors,
        labels,
        settings=settings,
    )

    claimed_cluster_ids = {r["cluster_id"] for r in result["reconciled"]}
    unclaimed = set(cluster_ids) - claimed_cluster_ids
    if unclaimed:
        logger.info(
            "reconcile_subtopics: %d cluster(s) unclaimed by any candidate, dropped: %s",
            len(unclaimed),
            sorted(unclaimed),
        )

    result["total_articles"] = len(articles)
    return result


def rank_and_cap_subtopics(
    reconciled: list[dict[str, Any]],
    total_articles: int,
    *,
    settings: Settings | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Rank `reconcile_candidates`'s reconciled subtopics by the
    distinctiveness formula and truncate to `Settings.pipeline.max_subtopics`
    (Task 2.2.4), per `notebooks/phase2-reconciliation-design.md`:

        volume_norm(s) = article_count(s) / total_articles
        avg_pairwise_distance(s) = mean(1 - cosine(centroid(s), centroid(s')))
                                    over all other surviving subtopics s'
                                    (0.0 if s has no peer)
        distinctiveness_score(s) = volume_weight * volume_norm(s)
                                    + distance_weight * avg_pairwise_distance(s)

    Returns `{"candidates": <ranked, capped list>, "excess": <the rest>}` --
    every excess subtopic is retained, never silently dropped. Each output
    dict is the input dict plus `volume_norm`/`avg_pairwise_distance`/
    `distinctiveness_score`, with the internal `"centroid"` array stripped
    (not JSON/Gate-1-display safe).
    """
    settings = settings or Settings()
    volume_weight = settings.clustering.distinctiveness_volume_weight
    distance_weight = settings.clustering.distinctiveness_distance_weight

    scored = []
    for r in reconciled:
        others = [o["centroid"] for o in reconciled if o is not r]
        if others:
            avg_dist = float(np.mean([1 - _cosine(r["centroid"], o) for o in others]))
        else:
            avg_dist = 0.0
        volume_norm = r["article_count"] / total_articles if total_articles else 0.0
        score = volume_weight * volume_norm + distance_weight * avg_dist
        scored.append(
            {
                **{k: v for k, v in r.items() if k != "centroid"},
                "volume_norm": round(volume_norm, 3),
                "avg_pairwise_distance": round(avg_dist, 3),
                "distinctiveness_score": round(score, 3),
            }
        )

    scored.sort(key=lambda s: -s["distinctiveness_score"])
    max_subtopics = settings.pipeline.max_subtopics
    return {"candidates": scored[:max_subtopics], "excess": scored[max_subtopics:]}
