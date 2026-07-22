"""Real end-to-end smoke test for `agents/sourcing_agent.py` (Story 1.9's
Phase 1 phase-level "Done when" acceptance criterion).

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach and `tests/live/test_gdelt_live.py`/
`tests/live/test_rss_live.py` for the existing precedent). Hits the real
GDELT DOC 2.0 API, real trusted-outlet RSS feeds, real WHOIS lookups, and
real HTTPS/about-page HEAD requests for every distinct domain surfaced --
this can take a few minutes given GDELT's aggressive rate limiting and
per-domain WHOIS/HTTPS soft-fail timeouts. Uses a hermetic `testcontainers`
Postgres for the reputation cache rather than depending on the dev
`docker compose up -d` stack being up.

Run explicitly:

    uv run pytest -m live tests/live/test_sourcing_agent_live.py -v
"""

import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sourcing_agent import ScoredArticle, sourcing_agent
from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.sourcing.dedup import normalize_url

pytestmark = pytest.mark.live

# Same narrow, low-volume keyword used in tests/live/test_gdelt_live.py's
# single-window case -- confirmed live (while building this test) to stay
# comfortably under GDELT's 250-record cap for a 3-day window, so this test
# exercises gdelt.py's normal single-window path rather than its sub-window
# pagination/retry-exhaustion path (a genuinely broad phrase like "climate
# change" was tried here first and reliably exceeded the cap at every
# bisection depth, needing far more retries/runtime than is reasonable for a
# routine verification run).
KEYWORDS = ["artificial intelligence regulation"]
LOOKBACK_DAYS = 3


def test_sourcing_agent_returns_deduped_threshold_filtered_articles_from_real_apis():
    with PostgresContainer("postgres:16-alpine") as postgres:
        database_url = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")
        pool = init_db(database_url)
        try:
            # min_primary_article_count=1 keeps this test scoped to real
            # GDELT+RSS (per the acceptance criterion's own wording), not
            # also depending on the unofficial Google News backfill endpoint.
            settings = Settings(sourcing={"min_primary_article_count": 1})
            result = sourcing_agent(
                KEYWORDS, lookback_days=LOOKBACK_DAYS, pool=pool, settings=settings
            )
        finally:
            pool.close()

    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, ScoredArticle)
        assert item.article["url"]
        assert item.article["domain"]
        assert item.reputation_score >= settings.reputation.min_score_threshold

    urls = [item.article["url"] for item in result]
    normalized_urls = [normalize_url(url) for url in urls]
    assert len(normalized_urls) == len(set(normalized_urls)), f"duplicate URLs survived: {urls}"

    domain_title_pairs = [(item.article["domain"], item.article["title"]) for item in result]
    assert len(domain_title_pairs) == len(set(domain_title_pairs)), "duplicate (domain, title) pairs survived"
