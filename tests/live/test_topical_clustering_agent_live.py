"""Real end-to-end smoke test for
`agents/topical_clustering_agent.py::topical_clustering_agent` (Task 2.5.1's
acceptance criterion).

Opt-in only (`@pytest.mark.live`, never run by default -- mirrors
`tests/live/test_subtopic_agent_live.py`'s structure exactly, at
per-subtopic scope instead of the broad-topic-fetch scope). Hits real
trusted-outlet RSS feeds, real WHOIS lookups, real HTTPS/about-page HEAD
requests, and a real (retry-bounded) GDELT DOC 2.0 call, via a hermetic
`testcontainers` Postgres for the reputation cache.

Run explicitly:

    uv run pytest -m live tests/live/test_topical_clustering_agent_live.py -v
"""

from unittest.mock import patch

import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.topical_clustering_agent import topical_clustering_agent
from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.sourcing import gdelt

pytestmark = pytest.mark.live

SUBTOPIC_LABEL = "solar panel manufacturing"
LOOKBACK_DAYS = 7

_real_gdelt_fetch = gdelt.fetch


def _bounded_real_gdelt_fetch(keywords, lookback_days, **kwargs):
    """Real `gdelt.fetch` call, retry-bounded for practicality -- see
    `tests/live/test_subtopic_agent_live.py`'s identical helper.
    """
    try:
        return _real_gdelt_fetch(keywords, lookback_days, max_retries=2, backoff_seconds=5.0, **kwargs)
    except gdelt.GDELTError:
        return []


def test_topical_clustering_agent_returns_plausible_clusters_from_real_apis():
    with PostgresContainer("postgres:16-alpine") as postgres:
        database_url = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")
        pool = init_db(database_url)
        try:
            settings = Settings(sourcing={"min_primary_article_count": 1})
            with patch("newsresearch.sourcing.gdelt.fetch", side_effect=_bounded_real_gdelt_fetch):
                result = topical_clustering_agent(
                    "st-live-1", SUBTOPIC_LABEL, LOOKBACK_DAYS, pool=pool, settings=settings
                )
        finally:
            pool.close()

    assert result["subtopic_id"] == "st-live-1"
    assert result["label"] == SUBTOPIC_LABEL
    assert result["total_articles"] > 0, (
        "expected at least one real article for a live, sustained subtopic like "
        f"{SUBTOPIC_LABEL!r} -- an empty result only proves the wrapper didn't "
        "crash, not that it meets this task's acceptance criterion"
    )

    clustered_count = sum(len(articles) for articles in result["clusters"].values())
    assert clustered_count + len(result["noise"]) == result["total_articles"]

    print(f"\n--- Task 2.5.1 live spot-check: {SUBTOPIC_LABEL!r} ---")
    print(f"total_articles={result['total_articles']}, noise={len(result['noise'])}")
    for cluster_id, articles in sorted(result["clusters"].items()):
        titles = [a["title"] for a in articles]
        print(f"cluster {cluster_id} ({len(articles)} articles): {titles}")
