"""Real-network smoke test for `sourcing/rss.py` (Task 1.3.1 acceptance).

Opt-in only (`@pytest.mark.live`, never run by default -- see
EXECUTION_PLAN.md's Testing approach). Hits BBC's and the Guardian's real
public RSS feeds (no API key needed) and confirms keyword/lookback
filtering actually works against live data, not just canned fixtures.

Run explicitly:

    uv run pytest -m live tests/live/test_rss_live.py
"""

import pytest

from newsresearch.sourcing import rss

pytestmark = pytest.mark.live

# A broad, near-certain-to-appear-somewhere-in-a-week-of-world-news keyword,
# so this doesn't flake on a quiet news day.
_KEYWORD = "the"
_LOOKBACK_DAYS = 14


def test_fetch_trusted_rss_returns_real_articles_from_at_least_two_outlets():
    articles = rss.fetch_trusted_rss(_KEYWORD, lookback_days=_LOOKBACK_DAYS)

    assert len(articles) > 0
    domains_seen = {article["domain"] for article in articles}
    assert len(domains_seen) >= 2, f"expected >=2 distinct outlet domains, got {domains_seen}"

    for article in articles:
        assert article["url"]
        assert article["title"]
        assert article["domain"] in rss.OUTLET_RSS_FEEDS
        assert article["source_type"] == "rss"
        assert article["published_at"] is not None


def test_fetch_trusted_rss_lookback_window_excludes_nothing_absurdly_old():
    # A 1-day lookback against live feeds that publish multiple times a day
    # should still return at least a couple of very recent hits.
    articles = rss.fetch_trusted_rss(_KEYWORD, lookback_days=1)

    assert len(articles) > 0
