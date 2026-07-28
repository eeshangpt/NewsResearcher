"""Ad hoc real-data sampling script for Task 2.8 clustering-quality review.

Not committed to production code -- data-scientist scratch analysis run
against the real `topical_clustering_agent()` (Task 2.5.1, PR #30) over
several varied real subtopics via live GDELT/RSS sourcing + local
sentence-transformers embeddings. Output (article counts, cluster sizes,
sample headlines per cluster) is dumped to stdout and captured into the
write-up on `feat/datascientist`; no full article text is persisted to disk
by this script (titles/urls/domains only, transient in-memory objects from
`sourcing_agent`, never written to a fixture file).
"""

from __future__ import annotations

import json

from newsresearch.agents.topical_clustering_agent import topical_clustering_agent

SUBTOPICS = [
    # (subtopic_id, label, lookback_days) -- varied size/breadth:
    # a narrow corporate-action angle, a broad ongoing-conflict angle,
    # a policy/regulatory angle, and a single-company-earnings angle.
    ("s1", "OpenAI GPT-5 release", 14),
    ("s2", "Russia Ukraine ceasefire negotiations", 14),
    ("s3", "Federal Reserve interest rate decision", 21),
    ("s4", "Tesla quarterly earnings", 21),
    ("s5", "solar panel manufacturing", 30),  # re-run of the 2.5.1 ad hoc spot-check subtopic
]

import sys

results = []
for subtopic_id, label, lookback in SUBTOPICS:
    print(f"\n=== {subtopic_id}: {label!r} (lookback={lookback}d) ===", flush=True)
    try:
        result = topical_clustering_agent(subtopic_id, label, lookback)
    except Exception as e:  # noqa: BLE001 -- ad hoc script, want to see all failures
        print(f"FAILED: {e!r}", flush=True)
        continue

    print(f"total_articles={result['total_articles']}")
    print(f"n_clusters={len(result['clusters'])} noise={len(result['noise'])}")
    for cid, articles in sorted(result["clusters"].items()):
        print(f"  cluster {cid} (n={len(articles)}):")
        for a in articles[:6]:
            print(f"    - {a.get('title', '')[:100]}  [{a.get('domain', '')}]")
    if result["noise"]:
        print(f"  noise (n={len(result['noise'])}):")
        for a in result["noise"][:6]:
            print(f"    - {a.get('title', '')[:100]}  [{a.get('domain', '')}]")

    results.append(
        {
            "subtopic_id": subtopic_id,
            "label": label,
            "lookback_days": lookback,
            "total_articles": result["total_articles"],
            "n_clusters": len(result["clusters"]),
            "cluster_sizes": {str(k): len(v) for k, v in result["clusters"].items()},
            "noise_count": len(result["noise"]),
            "cluster_titles": {
                str(k): [a.get("title", "") for a in v] for k, v in result["clusters"].items()
            },
            "noise_titles": [a.get("title", "") for a in result["noise"]],
        }
    )
    # Flush incrementally after each subtopic so a slow/killed later subtopic
    # doesn't lose earlier results.
    with open("/tmp/clustering_sample_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"-- wrote /tmp/clustering_sample_results.json ({len(results)} subtopic(s) so far)", flush=True)

print("\nDone.", flush=True)
