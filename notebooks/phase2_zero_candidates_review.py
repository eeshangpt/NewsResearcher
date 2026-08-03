"""Task: zero-subtopic-candidates-at-Gate-1 threshold review.

Reproduces the thin-RSS-only-batch scenario the two failed live runs
("Iraq and WMDs", "Taliban Takeover") hit when GDELT was down/rate-limited,
and traces both cliffs a small batch can fall off:

  1. `Settings.reputation.min_score_threshold` (sourcing_agent.py:179)
  2. `Settings.clustering.reconciliation_match_threshold`
     (subtopic_agent.py:175-177)

Runs the *real* `reputation/signals.py` collectors (real WHOIS/Tranco/HTTPS
network calls) and the *real* `reputation/scorer.py` formula against a real
RSS+Google-News-backfill fetch -- this script does not go through
`sourcing_agent()` itself only because that function requires a live
Postgres pool for the reputation cache (`reputation/cache.py`), which isn't
available in this analysis environment; the scoring math it wraps is
reproduced call-for-call here instead (`_score_domain_live`, no caching).

No article full text is persisted anywhere by this script or its output
JSON -- only url/title/domain, matching the no-full-text-storage rule.

Two topics, matching the two failed live runs' rough shape:
  - "Iraq WMD" -- GDELT down, RSS empty, backfill-only, ~12 articles.
  - "Taliban" -- GDELT down, RSS thin (a few), backfill fills the rest.

Because live GDELT/WHOIS/Tranco-snapshot/HTTPS results vary run to run, the
actual counts/scores are captured to
`notebooks/phase2_zero_candidates_sample.json` at the time this was run
(2026-08-02) and the write-up quotes those numbers -- re-running this script
later will very likely produce different absolute scores (fresh WHOIS ages,
different Google News backfill results) but the same qualitative shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from newsresearch.agents.subtopic_agent import _candidate_text, _cluster_centroid, _cosine, reconcile_candidates
from newsresearch.clustering.cluster import cluster
from newsresearch.clustering.embeddings import embed
from newsresearch.config import Settings
from newsresearch.reputation import scorer, signals
from newsresearch.sourcing import dedup as dedup_module
from newsresearch.sourcing import google_news_backfill as gnb
from newsresearch.sourcing import rss

OUT_DIR = Path(__file__).parent
SETTINGS = Settings()


def _has_required_fields(article: dict[str, Any]) -> bool:
    return bool(article.get("url")) and bool(article.get("title")) and bool(article.get("domain"))


def _score_domain_live(domain: str, presence_frequency: float | None) -> scorer.DomainReputationScore:
    """Same call sequence as `sourcing_agent._get_or_score_domain_reputation`,
    minus the Postgres cache (no DB available here) -- always a fresh
    WHOIS/Tranco/HTTPS lookup."""
    domain_age_years = signals.get_domain_age_years(domain)
    backlink_proxy = signals.get_backlink_proxy_score(domain)
    legitimacy = signals.check_https_and_about_page(domain)
    return scorer.score_domain(
        domain,
        domain_age_years=domain_age_years,
        backlink_proxy=backlink_proxy,
        presence_frequency=presence_frequency,
        https_present=legitimacy["https_present"],
        about_page_present=legitimacy["about_page_present"],
        settings=SETTINGS,
    )


def fetch_thin_batch(keywords: list[str], lookback_days: int = 14) -> list[dict[str, Any]]:
    """RSS + (Google-News-backfill-if-thin) fetch, exactly `sourcing_agent`'s
    step 1-3 (GDELT omitted -- treated as down/blocked, matching the two
    failed live runs and this environment's own known GDELT-block issue)."""
    rss_articles = rss.fetch_trusted_rss(keywords, lookback_days)
    primary = rss_articles  # gdelt_articles = [] (down)

    if len(primary) < SETTINGS.sourcing.min_primary_article_count:
        try:
            backfill = gnb.fetch_google_news_backfill(keywords, lookback_days=lookback_days)
        except Exception:
            backfill = []
        combined = primary + backfill
    else:
        combined = primary

    valid = [a for a in combined if _has_required_fields(a)]
    return dedup_module.dedup(valid, settings=SETTINGS)


def score_batch(deduped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    presence = signals.get_presence_frequency_scores(deduped)
    domain_scores: dict[str, scorer.DomainReputationScore] = {}
    rows = []
    for article in deduped:
        domain = article["domain"]
        if domain not in domain_scores:
            domain_scores[domain] = _score_domain_live(domain, presence.get(domain))
        s = domain_scores[domain]
        rows.append(
            {
                "domain": domain,
                "title": article["title"],
                "tier": s["tier"],
                "base_score": s["base_score"],
                "heuristic_adjustment": round(s["heuristic_adjustment"], 4),
                "final_score": round(s["final_score"], 4),
                "passes_0.50": s["final_score"] >= 0.50,
            }
        )
    return rows


# Fixed LLM-proposed candidate labels standing in for `propose_candidates()`
# (no OPENAI_API_KEY in this analysis environment) -- hand-written to be
# plausible subtopic proposals for each topic's real surviving-article
# content, same role as the reconciliation-design doc's fixture candidates.
# Explicitly NOT a live LLM call; flagged here, not disguised as one.
CANDIDATES_BY_TOPIC: dict[str, list[tuple[str, str]]] = {
    "iraq_wmd": [
        ("Iraq War rationale and WMD intelligence failures", "Coverage of the pre-war WMD intelligence claims and their politicization."),
        ("Iran-backed militia strikes in Iraq", "Coverage of US/Saudi strikes on Iran-backed fighters operating in Iraq."),
        ("Nuclear weapons proliferation history", "General nuclear-proliferation and deterrence explainer content."),
        ("Iraq War historical retrospectives", "Encyclopedia-style historical summaries of the Iraq War and Gulf War."),
    ],
    "taliban": [
        ("Taliban treatment of women and girls", "Coverage of Taliban restrictions on women's education, dress, and movement."),
        ("Armed resistance and attacks against the Taliban", "Coverage of insurgent/resistance attacks on Taliban officials and outposts."),
        ("Taliban foreign policy and international relations", "Coverage of Taliban diplomacy, deportations, and regional relations."),
        ("Taliban internal governance and administration", "Coverage of Taliban internal culture-ministry and administrative actions."),
    ],
}


def reconcile_topic(topic_key: str, surviving_articles: list[dict[str, Any]]) -> dict[str, Any]:
    if len(surviving_articles) < 2:
        return {"skipped": "fewer than 2 surviving articles, clustering not meaningful"}

    vectors = embed([a.get("title", "") for a in surviving_articles])
    candidates = CANDIDATES_BY_TOPIC[topic_key]
    labels = cluster(vectors, k_hint=len(candidates))
    cluster_ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    centroids = {cid: _cluster_centroid(vectors, labels, cid) for cid in cluster_ids}

    candidate_labels = [c[0] for c in candidates]
    candidate_embeddings = embed([_candidate_text(c[0], c[1]) for c in candidates])

    result = reconcile_candidates(
        candidate_labels,
        candidate_embeddings,
        cluster_ids,
        centroids,
        vectors,
        labels,
        settings=SETTINGS,
    )

    # Also report best-sim at a few alternate thresholds so the write-up can
    # quote exactly how many candidates would survive at each candidate bar
    # without re-running clustering.
    n_candidates = len(candidate_labels)
    sim_matrix = np.zeros((n_candidates, len(cluster_ids)))
    for ci, cid in enumerate(cluster_ids):
        centroid = centroids[cid]
        for i in range(n_candidates):
            sim_matrix[i, ci] = _cosine(candidate_embeddings[i], centroid)
    best_sim = sim_matrix.max(axis=1) if len(cluster_ids) else np.zeros(n_candidates)

    survivors_at = {}
    for t in (0.60, 0.55, 0.50, 0.45, 0.40):
        survivors_at[t] = int((best_sim >= t).sum())

    return {
        "n_articles": len(surviving_articles),
        "n_clusters_found": len(cluster_ids),
        "candidate_best_sim": {candidate_labels[i]: round(float(best_sim[i]), 3) for i in range(n_candidates)},
        "dropped_at_0.60": result["dropped"],
        "n_reconciled_at_0.60": len(result["reconciled"]),
        "survivors_by_threshold": survivors_at,
    }


def main() -> None:
    report: dict[str, Any] = {}

    for topic_key, keywords in [("iraq_wmd", ["Iraq", "WMD"]), ("taliban", ["Taliban"])]:
        deduped = fetch_thin_batch(keywords)
        scored_rows = score_batch(deduped)
        surviving = [r for r in scored_rows if r["passes_0.50"]]

        n_pass = sum(r["passes_0.50"] for r in scored_rows)
        under_by = sorted(round(0.50 - r["final_score"], 4) for r in scored_rows if not r["passes_0.50"])

        recon = reconcile_topic(
            topic_key,
            [{"title": r["title"], "domain": r["domain"]} for r in surviving],
        )

        report[topic_key] = {
            "n_deduped_articles": len(deduped),
            "score_rows": scored_rows,
            "n_pass_0.50": n_pass,
            "n_fail_0.50": len(scored_rows) - n_pass,
            "shortfall_below_0.50_for_failures": under_by,
            "reconciliation": recon,
        }
        print(f"=== {topic_key} ===")
        print(f"  deduped articles: {len(deduped)}")
        print(f"  pass 0.50: {n_pass} / {len(scored_rows)}")
        print(f"  shortfalls for failures: {under_by}")
        print(f"  reconciliation: {recon}")

    out_path = OUT_DIR / "phase2_zero_candidates_sample.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
