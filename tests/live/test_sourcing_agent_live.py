"""Real end-to-end smoke test for `agents/sourcing_agent.py` (Story 1.9's
Phase 1 phase-level "Done when" acceptance criterion).

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach and `tests/live/test_gdelt_live.py`/
`tests/live/test_rss_live.py` for the existing precedent). Hits real
trusted-outlet RSS feeds, real WHOIS lookups, and real HTTPS/about-page HEAD
requests for every distinct domain surfaced, plus a real (but
retry-bounded, see below) GDELT DOC 2.0 call. Uses a hermetic
`testcontainers` Postgres for the reputation cache rather than depending on
the dev `docker compose up -d` stack being up.

Keyword choice, matching `tests/live/test_rss_live.py`'s own precedent
("the", chosen so the test "doesn't flake on a quiet news day"): "Iran" is a
broad, sustained current-events topic (tech-lead independently confirmed via
direct feed inspection that it overlaps real BBC/Guardian/NPR/Al Jazeera
headlines) rather than a narrow phrase that may or may not appear in any
given trusted-outlet feed on a given day -- a narrow keyword's empty result
only proves the pipeline didn't crash, not that it meets this test's actual
acceptance criterion ("returns ScoredArticle objects... above threshold").

GDELT-specific bounded-retry note: a broad/sustained topic like "Iran" is
also high-volume enough in GDELT to reliably trigger its 250-record-cap
sub-window pagination, and this sandbox has independently exhausted GDELT's
real rate-limit retry budget repeatedly while this suite was being built --
a known, already-accepted characteristic of the merged `gdelt.py` client
under sustained request volume, not a regression introduced here. Rather
than mocking GDELT out entirely (which would stop exercising it at all) or
leaving its default retry budget in place (worst case: several minutes of
exponential backoff before this test can even reach its assertions), this
test calls the *real* `gdelt.fetch` with a smaller, test-local retry budget
and treats a resulting `GDELTError` as "no GDELT contribution this run" --
real trusted-tier RSS results alone are enough to exercise/verify this
test's actual target (the reputation-threshold-filter + dedup path), so a
GDELT rate-limit exhaustion here is non-fatal to the test, while a genuine
GDELT success still contributes real results when it isn't currently
throttled.

Run explicitly:

    uv run pytest -m live tests/live/test_sourcing_agent_live.py -v
"""

from unittest.mock import patch

import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sourcing_agent import ScoredArticle, sourcing_agent
from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.sourcing import gdelt
from newsresearch.sourcing.dedup import normalize_url

pytestmark = pytest.mark.live

KEYWORDS = ["Iran"]
LOOKBACK_DAYS = 2

_real_gdelt_fetch = gdelt.fetch


def _bounded_real_gdelt_fetch(keywords, lookback_days, **kwargs):
    """Real `gdelt.fetch` call, retry-bounded for this test's practicality
    (see module docstring) -- a `GDELTError` here (rate-limit retries
    exhausted) is swallowed to an empty list rather than failing this test,
    since real trusted-tier RSS results alone already exercise the
    acceptance criterion this test targets.
    """
    try:
        return _real_gdelt_fetch(keywords, lookback_days, max_retries=2, backoff_seconds=5.0, **kwargs)
    except gdelt.GDELTError:
        return []


def test_sourcing_agent_returns_deduped_threshold_filtered_articles_from_real_apis():
    with PostgresContainer("postgres:16-alpine") as postgres:
        database_url = postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")
        pool = init_db(database_url)
        try:
            # min_primary_article_count=1 keeps this test scoped to real
            # GDELT+RSS (per the acceptance criterion's own wording), not
            # also depending on the unofficial Google News backfill endpoint.
            settings = Settings(sourcing={"min_primary_article_count": 1})
            with patch("newsresearch.sourcing.gdelt.fetch", side_effect=_bounded_real_gdelt_fetch):
                result = sourcing_agent(
                    KEYWORDS, lookback_days=LOOKBACK_DAYS, pool=pool, settings=settings
                )
        finally:
            pool.close()

    assert len(result) > 0, (
        "expected at least one real article to clear the reputation threshold -- an "
        "empty result only proves the pipeline didn't crash, not that it meets the "
        "acceptance criterion"
    )
    assert all(isinstance(item, ScoredArticle) for item in result)
    for item in result:
        assert item.article["url"]
        assert item.article["domain"]
        assert item.reputation_score >= settings.reputation.min_score_threshold

    # At least one result should actually be a known trusted-tier outlet
    # (bbc.com/theguardian.com/npr.org/aljazeera.com are all "major" tier in
    # data/trusted_outlets.yaml), not just an unknown-tier domain that
    # happened to clear the threshold on favorable signals alone.
    assert any(item.reputation_tier in ("wire", "major") for item in result)

    urls = [item.article["url"] for item in result]
    normalized_urls = [normalize_url(url) for url in urls]
    assert len(normalized_urls) == len(set(normalized_urls)), f"duplicate URLs survived: {urls}"

    domain_title_pairs = [(item.article["domain"], item.article["title"]) for item in result]
    assert len(domain_title_pairs) == len(set(domain_title_pairs)), "duplicate (domain, title) pairs survived"
