"""Gate 2 report -- pure aggregation over Story 2.5's clustering output (Task 2.6.1).

No LLM calls: this module never imports or calls `llm/models.py::get_chat_model`,
so Gate 2's human review renders at zero marginal cost per subtopic.
"""

from __future__ import annotations

from typing import Any

_SAMPLE_HEADLINES_PER_CLUSTER = 2


def build_gate2_report(clustering_result: dict[str, Any]) -> dict[str, Any]:
    """Aggregate `topical_clustering_agent`'s output into `SubtopicState.cluster_report`'s shape.

    `clustering_result` is `{"subtopic_id", "label", "clusters": {cluster_id:
    [article, ...]}, "noise": [article, ...], "total_articles"}` (Task 2.5.1).

    Returns `{"cluster_sizes": list[int], "sample_headlines": list[str],
    "source_spread": dict[str, int]}` (`graph/state.py::SubtopicState.cluster_report`).

    `source_spread` counts domains across clusters *and* noise -- a reviewer
    deciding whether a subtopic's source diversity is real needs the full
    picture, not just the clustered subset.
    """
    clusters: dict[Any, list[dict]] = clustering_result.get("clusters", {})
    noise: list[dict] = clustering_result.get("noise", [])

    cluster_sizes = [len(articles) for articles in clusters.values()]

    sample_headlines: list[str] = []
    for articles in clusters.values():
        sample_headlines.extend(
            a["title"] for a in articles[:_SAMPLE_HEADLINES_PER_CLUSTER] if a.get("title")
        )

    source_spread: dict[str, int] = {}
    for article in [a for articles in clusters.values() for a in articles] + noise:
        domain = article.get("domain")
        if domain:
            source_spread[domain] = source_spread.get(domain, 0) + 1

    return {
        "cluster_sizes": cluster_sizes,
        "sample_headlines": sample_headlines,
        "source_spread": source_spread,
    }
