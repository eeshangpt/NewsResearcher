"""Unit test for `agents/subtopic_agent.py::broad_topic_fetch` (Task 2.2.2).

`sourcing_agent` is Phase 1's already-approved, already-tested module (see
`tests/test_sourcing_agent.py`) -- this test mocks it out entirely rather
than re-testing its internals, and only confirms `broad_topic_fetch`'s own
thin-wrapper behaviour: it derives keywords from the topic string, forwards
`lookback_days`/`pool`/`settings` through, and unwraps `ScoredArticle`s to
plain article dicts.
"""

from datetime import datetime, timezone
from unittest.mock import patch

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.agents.subtopic_agent import broad_topic_fetch

ARTICLE_A = {
    "title": "Solar capacity hits record high",
    "url": "https://example-a.com/solar-record",
    "domain": "example-a.com",
    "published_at": datetime(2026, 7, 20, tzinfo=timezone.utc),
    "source_type": "gdelt",
}
ARTICLE_B = {
    "title": "Wind policy shake-up announced",
    "url": "https://example-b.com/wind-policy",
    "domain": "example-b.com",
    "published_at": datetime(2026, 7, 21, tzinfo=timezone.utc),
    "source_type": "rss",
}

MOCK_SCORED_ARTICLES = [
    ScoredArticle(article=ARTICLE_A, reputation_score=0.9, reputation_tier="major"),
    ScoredArticle(article=ARTICLE_B, reputation_score=0.85, reputation_tier="wire"),
]


@patch("newsresearch.agents.subtopic_agent.sourcing_agent")
def test_broad_topic_fetch_derives_keywords_and_calls_sourcing_agent(mock_sourcing_agent):
    mock_sourcing_agent.return_value = MOCK_SCORED_ARTICLES

    result = broad_topic_fetch("renewable energy", lookback_days=7)

    mock_sourcing_agent.assert_called_once_with(
        ["renewable energy"], 7, pool=None, settings=None
    )
    assert result == [ARTICLE_A, ARTICLE_B]
    assert all(isinstance(item, dict) for item in result)
    assert not any(isinstance(item, ScoredArticle) for item in result)


@patch("newsresearch.agents.subtopic_agent.sourcing_agent")
def test_broad_topic_fetch_forwards_pool_and_settings(mock_sourcing_agent):
    mock_sourcing_agent.return_value = []
    sentinel_pool = object()
    sentinel_settings = object()

    result = broad_topic_fetch(
        "renewable energy", lookback_days=14, pool=sentinel_pool, settings=sentinel_settings
    )

    mock_sourcing_agent.assert_called_once_with(
        ["renewable energy"], 14, pool=sentinel_pool, settings=sentinel_settings
    )
    assert result == []
