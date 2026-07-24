"""Real end-to-end smoke test for `agents/subtopic_agent.py::broad_topic_fetch`
(Task 2.2.2's acceptance criterion).

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach and `tests/live/test_sourcing_agent_live.py`
for the existing precedent this test mirrors). Hits real trusted-outlet RSS
feeds, real WHOIS lookups, real HTTPS/about-page HEAD requests, and a real
(retry-bounded) GDELT DOC 2.0 call, via a hermetic `testcontainers` Postgres
for the reputation cache.

Topic choice: "renewable energy" is the task's own example of a broad topic
plausibly spanning multiple candidate-subtopic angles (solar, wind, policy,
storage, etc.) -- this test can't yet assert on actual subtopic labels (no
Subtopic Agent proposal/clustering exists yet, that's Wave 2/3), so it uses
domain diversity across the returned article set as the closest mechanical
proxy available for "spans multiple candidate subtopics" without inventing
subtopic-detection logic that belongs to a later task.

Run explicitly:

    uv run pytest -m live tests/live/test_subtopic_agent_live.py -v
"""

from unittest.mock import patch

import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.subtopic_agent import broad_topic_fetch
from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.sourcing import gdelt

pytestmark = pytest.mark.live

TOPIC = "renewable energy"
LOOKBACK_DAYS = 7

_real_gdelt_fetch = gdelt.fetch


def _bounded_real_gdelt_fetch(keywords, lookback_days, **kwargs):
    """Real `gdelt.fetch` call, retry-bounded for practicality (see
    `tests/live/test_sourcing_agent_live.py`'s module docstring for the
    rationale this mirrors) -- a `GDELTError` here is swallowed to an empty
    list rather than failing this test, since real trusted-tier RSS results
    alone already exercise this test's actual target.
    """
    try:
        return _real_gdelt_fetch(keywords, lookback_days, max_retries=2, backoff_seconds=5.0, **kwargs)
    except gdelt.GDELTError:
        return []


def test_broad_topic_fetch_returns_multi_domain_article_set_from_real_apis():
    with PostgresContainer("postgres:16-alpine") as postgres:
        database_url = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")
        pool = init_db(database_url)
        try:
            settings = Settings(sourcing={"min_primary_article_count": 1})
            with patch("newsresearch.sourcing.gdelt.fetch", side_effect=_bounded_real_gdelt_fetch):
                result = broad_topic_fetch(TOPIC, lookback_days=LOOKBACK_DAYS, pool=pool, settings=settings)
        finally:
            pool.close()

    assert len(result) > 0, (
        "expected at least one real article for a broad, sustained topic like "
        "'renewable energy' -- an empty result only proves the pipeline didn't "
        "crash, not that it meets this task's acceptance criterion"
    )
    assert all(isinstance(item, dict) for item in result)
    for article in result:
        assert article["url"]
        assert article["title"]
        assert article["domain"]

    domains = {article["domain"] for article in result}
    assert len(domains) > 1, (
        f"expected a broad topic fetch to span multiple domains (a proxy for "
        f"multiple candidate subtopics), got only: {domains}"
    )
