"""Unit tests for `sourcing/web_search.py` (GDELT replacement, issue #101).

Search tools and full-text fetch are mocked -- no live network calls, no
live API keys needed (mirrors `tests/test_gdelt.py`'s mocking conventions).
"""

from unittest.mock import patch

import pytest

from newsresearch.config import Settings
from newsresearch.sourcing.web_search import WebSearchError, _tavily_time_range, fetch


def test_tavily_time_range_maps_day_counts_to_nearest_bucket():
    assert _tavily_time_range(1) == "day"
    assert _tavily_time_range(7) == "week"
    assert _tavily_time_range(20) == "month"
    assert _tavily_time_range(90) == "year"


@patch("newsresearch.sourcing.web_search.fetch_fulltext")
@patch("newsresearch.sourcing.web_search._tavily_search")
@patch("newsresearch.sourcing.web_search._ddg_search")
def test_fetch_merges_ddg_and_tavily_and_dedups_by_url(mock_ddg, mock_tavily, mock_fulltext):
    mock_ddg.return_value = [
        {"link": "https://example.com/a1", "title": "Story A"},
        {"link": "https://example.com/a2", "title": "Story B"},
    ]
    mock_tavily.return_value = [
        # Cross-source exact-URL duplicate -- must collapse to one entry.
        {"url": "https://example.com/a1", "title": "Story A"},
        {"url": "https://other.com/a3", "title": "Story C"},
    ]
    mock_fulltext.return_value = "some article body text"
    settings = Settings(tavily_api_key="fake-key")

    articles = fetch(["ceasefire"], lookback_days=7, settings=settings)

    urls = {a["url"] for a in articles}
    assert urls == {"https://example.com/a1", "https://example.com/a2", "https://other.com/a3"}
    for a in articles:
        assert a["source_type"] == "web_search"
        assert a["domain"]
        assert a["published_at"] is None


@patch("newsresearch.sourcing.web_search.fetch_fulltext")
@patch("newsresearch.sourcing.web_search._ddg_search")
def test_fetch_skips_tavily_leg_without_api_key(mock_ddg, mock_fulltext):
    mock_ddg.return_value = [{"link": "https://example.com/a1", "title": "Story A"}]
    mock_fulltext.return_value = "body text"
    settings = Settings(tavily_api_key=None)

    articles = fetch(["ceasefire"], lookback_days=7, settings=settings)

    assert len(articles) == 1
    assert articles[0]["url"] == "https://example.com/a1"


@patch("newsresearch.sourcing.web_search.fetch_fulltext")
@patch("newsresearch.sourcing.web_search._ddg_search")
def test_fetch_drops_candidates_that_fail_fulltext_fetch(mock_ddg, mock_fulltext):
    mock_ddg.return_value = [
        {"link": "https://example.com/ok", "title": "Fetchable"},
        {"link": "https://example.com/dead", "title": "Dead link"},
    ]
    mock_fulltext.side_effect = lambda url, settings=None: "body" if url.endswith("ok") else None
    settings = Settings(tavily_api_key=None)

    articles = fetch(["ceasefire"], lookback_days=7, settings=settings)

    assert len(articles) == 1
    assert articles[0]["url"] == "https://example.com/ok"


@patch("newsresearch.sourcing.web_search._ddg_search")
def test_fetch_raises_web_search_error_when_both_legs_fail(mock_ddg):
    mock_ddg.side_effect = RuntimeError("DuckDuckGo backend error")
    settings = Settings(tavily_api_key=None)

    with pytest.raises(WebSearchError):
        fetch(["ceasefire"], lookback_days=7, settings=settings)
