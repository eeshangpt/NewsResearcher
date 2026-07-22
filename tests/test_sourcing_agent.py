"""Unit tests for `agents/sourcing_agent.py`'s orchestration logic (Task 1.9.1).

Hermetic: real `testcontainers` Postgres (so `reputation/cache.py` +
`reputation/scorer.py` run for real, exercising the actual cache-first
wiring) but every network-calling collector (GDELT, RSS, Google News
backfill, WHOIS, HTTPS/about-page) is mocked -- matching this repo's
established pattern of a fast/deterministic unit test for orchestration plus
a separate `@pytest.mark.live` test for the real end-to-end run (see
`tests/live/test_sourcing_agent_live.py`).

`reuters.com`/`apnews.com` are real `data/trusted_outlets.yaml` wire-tier
entries (base_score_wire=0.8); `spam-site.example` is absent from that
whitelist (base_score_unknown=0.3). With every signal collector mocked to a
neutral value, unknown-tier domains land at ~0.3 + 0.15 = 0.45 (below the
default 0.5 threshold) while wire-tier domains land at ~0.8 + 0.15 = 0.95
(comfortably above) -- this lets threshold-filtering be exercised without
per-domain-differentiated signal mocking.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.agents.sourcing_agent import ScoredArticle, sourcing_agent
from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.reputation import cache

# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
def pool(postgres_url):
    pool = init_db(postgres_url)
    yield pool
    pool.close()


NEUTRAL_SIGNALS = patch.multiple(
    "newsresearch.reputation.signals",
    get_domain_age_years=lambda domain, timeout=10: None,
    get_backlink_proxy_score=lambda domain: 0.5,
    check_https_and_about_page=lambda domain, timeout=5.0: {
        "https_present": None,
        "about_page_present": None,
    },
)


def _article(
    title: str,
    url: str,
    domain: str,
    source_type: str = "gdelt",
    published_at: datetime | None = None,
) -> dict:
    return {
        "title": title,
        "url": url,
        "domain": domain,
        "published_at": published_at or datetime(2026, 7, 20, tzinfo=timezone.utc),
        "source_type": source_type,
    }


# --- happy path: dedup + threshold-filter ------------------------------------


@NEUTRAL_SIGNALS
@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_returns_deduped_threshold_filtered_scored_articles(
    mock_gdelt_fetch, mock_rss_fetch, pool
):
    mock_gdelt_fetch.return_value = [
        _article("Ceasefire talks resume in Geneva", "https://reuters.com/a1", "reuters.com"),
        # Exact-URL duplicate of the article above (different query params) -- must collapse.
        _article(
            "Ceasefire talks resume in Geneva",
            "https://reuters.com/a1?utm_source=twitter",
            "reuters.com",
        ),
        _article("Low-quality unverified rumor spreads", "https://spam-site.example/r1", "spam-site.example"),
    ]
    mock_rss_fetch.return_value = [
        # Cross-source near-dupe of the Reuters wire story -- must collapse.
        _article(
            "Ceasefire talks resume in Geneva today",
            "https://apnews.com/wire1",
            "apnews.com",
            source_type="rss",
        ),
    ]
    settings = Settings(sourcing={"min_primary_article_count": 1})

    result = sourcing_agent(["ceasefire"], lookback_days=7, pool=pool, settings=settings)

    assert all(isinstance(item, ScoredArticle) for item in result)

    urls = [item.article["url"] for item in result]
    assert len(urls) == len(set(urls)), f"duplicate URLs in result: {urls}"

    domains = {item.article["domain"] for item in result}
    assert domains == {"reuters.com"}  # the near-dupe AP article and the spam domain both drop out

    for item in result:
        assert item.reputation_score >= settings.reputation.min_score_threshold
        assert item.reputation_tier == "wire"


@NEUTRAL_SIGNALS
@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_filters_out_unknown_tier_domain_below_threshold(
    mock_gdelt_fetch, mock_rss_fetch, pool
):
    mock_gdelt_fetch.return_value = [
        _article("Reliable wire report", "https://reuters.com/a2", "reuters.com"),
    ]
    mock_rss_fetch.return_value = [
        _article("Unverified rumor", "https://spam-site.example/r2", "spam-site.example", source_type="rss"),
    ]
    settings = Settings(sourcing={"min_primary_article_count": 1})

    result = sourcing_agent(["news"], lookback_days=7, pool=pool, settings=settings)

    kept_domains = {item.article["domain"] for item in result}
    assert kept_domains == {"reuters.com"}
    assert "spam-site.example" not in kept_domains


# --- cache-first path ---------------------------------------------------------


@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_uses_fresh_cache_without_recomputing_signals(mock_gdelt_fetch, mock_rss_fetch, pool):
    mock_gdelt_fetch.return_value = [
        _article("Cached wire report", "https://reuters.com/cached1", "reuters.com"),
    ]
    mock_rss_fetch.return_value = []
    settings = Settings(sourcing={"min_primary_article_count": 1})

    cache.write_domain_reputation(
        pool,
        {
            "domain": "reuters.com",
            "tier": "wire",
            "base_score": 0.8,
            "heuristic_adjustment": 0.1,
            "final_score": 0.9,
        },
    )

    with patch("newsresearch.reputation.signals.get_domain_age_years") as mock_age, patch(
        "newsresearch.reputation.signals.get_backlink_proxy_score"
    ) as mock_backlink, patch("newsresearch.reputation.signals.check_https_and_about_page") as mock_https:
        result = sourcing_agent(["ceasefire"], lookback_days=7, pool=pool, settings=settings)

    mock_age.assert_not_called()
    mock_backlink.assert_not_called()
    mock_https.assert_not_called()
    assert len(result) == 1
    assert result[0].reputation_score == 0.9
    assert result[0].reputation_tier == "wire"


# --- backfill wiring -----------------------------------------------------------


@NEUTRAL_SIGNALS
@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_invokes_backfill_when_primary_count_below_threshold(
    mock_gdelt_fetch, mock_rss_fetch, mock_backfill_fetch, pool
):
    mock_gdelt_fetch.return_value = [
        _article("Thin primary result", "https://reuters.com/thin1", "reuters.com"),
    ]
    mock_rss_fetch.return_value = []
    mock_backfill_fetch.return_value = [
        _article(
            "Backfilled wire report",
            "https://apnews.com/backfill1",
            "apnews.com",
            source_type="google_news_backfill",
        ),
    ]
    settings = Settings(sourcing={"min_primary_article_count": 5})

    result = sourcing_agent(["ceasefire"], lookback_days=7, pool=pool, settings=settings)

    mock_backfill_fetch.assert_called_once()
    domains = {item.article["domain"] for item in result}
    assert domains == {"reuters.com", "apnews.com"}


@NEUTRAL_SIGNALS
@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_survives_backfill_failure_nfr3(
    mock_gdelt_fetch, mock_rss_fetch, mock_backfill_fetch, pool
):
    mock_gdelt_fetch.return_value = [
        _article("Thin primary result", "https://reuters.com/resilience1", "reuters.com"),
    ]
    mock_rss_fetch.return_value = []
    mock_backfill_fetch.side_effect = RuntimeError("simulated Google News RSS outage")
    settings = Settings(sourcing={"min_primary_article_count": 5})

    result = sourcing_agent(["ceasefire"], lookback_days=7, pool=pool, settings=settings)

    mock_backfill_fetch.assert_called_once()
    assert len(result) == 1
    assert result[0].article["domain"] == "reuters.com"


# --- pool lifecycle ------------------------------------------------------------


@NEUTRAL_SIGNALS
@patch("newsresearch.sourcing.rss.fetch_trusted_rss")
@patch("newsresearch.sourcing.gdelt.fetch")
def test_sourcing_agent_raises_without_database_url_or_pool(mock_gdelt_fetch, mock_rss_fetch):
    mock_gdelt_fetch.return_value = []
    mock_rss_fetch.return_value = []
    settings = Settings(database_url=None)

    with pytest.raises(ValueError, match="Postgres connection"):
        sourcing_agent(["ceasefire"], lookback_days=7, settings=settings)
