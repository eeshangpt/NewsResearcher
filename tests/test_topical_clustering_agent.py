"""Unit tests for `agents/topical_clustering_agent.py` (Task 2.5.1).

`sourcing_agent`, `embed`, and `cluster` are mocked out entirely -- this
file only confirms the wrapper's own wiring: it scopes `sourcing_agent` to
the subtopic label, forwards `pool`/`settings`, embeds article titles, feeds
those vectors to `cluster()`, and groups the resulting articles by cluster
id (noise, label `-1`, kept separately).
"""

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np

from newsresearch.agents.sourcing_agent import ScoredArticle
from newsresearch.agents.topical_clustering_agent import topical_clustering_agent

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
ARTICLE_C = {
    "title": "Unrelated outlier story",
    "url": "https://example-c.com/outlier",
    "domain": "example-c.com",
    "published_at": datetime(2026, 7, 22, tzinfo=timezone.utc),
    "source_type": "rss",
}

MOCK_SCORED_ARTICLES = [
    ScoredArticle(article=ARTICLE_A, reputation_score=0.9, reputation_tier="major"),
    ScoredArticle(article=ARTICLE_B, reputation_score=0.85, reputation_tier="wire"),
    ScoredArticle(article=ARTICLE_C, reputation_score=0.7, reputation_tier="regional"),
]


@patch("newsresearch.agents.topical_clustering_agent.cluster")
@patch("newsresearch.agents.topical_clustering_agent.embed")
@patch("newsresearch.agents.topical_clustering_agent.sourcing_agent")
def test_scopes_sourcing_to_subtopic_label_and_forwards_args(
    mock_sourcing_agent, mock_embed, mock_cluster
):
    mock_sourcing_agent.return_value = MOCK_SCORED_ARTICLES
    mock_embed.return_value = np.zeros((3, 4))
    mock_cluster.return_value = np.array([0, 0, -1])
    sentinel_pool, sentinel_settings = object(), object()

    topical_clustering_agent(
        "st-1", "Solar and wind energy", 7, pool=sentinel_pool, settings=sentinel_settings
    )

    mock_sourcing_agent.assert_called_once_with(
        ["Solar and wind energy"], 7, pool=sentinel_pool, settings=sentinel_settings
    )
    mock_embed.assert_called_once_with(
        ["Solar capacity hits record high", "Wind policy shake-up announced", "Unrelated outlier story"]
    )
    mock_cluster.assert_called_once()


@patch("newsresearch.agents.topical_clustering_agent.cluster")
@patch("newsresearch.agents.topical_clustering_agent.embed")
@patch("newsresearch.agents.topical_clustering_agent.sourcing_agent")
def test_groups_articles_by_cluster_id_and_separates_noise(
    mock_sourcing_agent, mock_embed, mock_cluster
):
    mock_sourcing_agent.return_value = MOCK_SCORED_ARTICLES
    mock_embed.return_value = np.zeros((3, 4))
    mock_cluster.return_value = np.array([0, 0, -1])

    result = topical_clustering_agent("st-1", "Solar and wind energy", 7)

    assert result["subtopic_id"] == "st-1"
    assert result["label"] == "Solar and wind energy"
    assert result["total_articles"] == 3
    assert result["clusters"] == {0: [ARTICLE_A, ARTICLE_B]}
    assert result["noise"] == [ARTICLE_C]


@patch("newsresearch.agents.topical_clustering_agent.cluster")
@patch("newsresearch.agents.topical_clustering_agent.embed")
@patch("newsresearch.agents.topical_clustering_agent.sourcing_agent")
def test_no_noise_when_all_articles_cluster(mock_sourcing_agent, mock_embed, mock_cluster):
    mock_sourcing_agent.return_value = MOCK_SCORED_ARTICLES
    mock_embed.return_value = np.zeros((3, 4))
    mock_cluster.return_value = np.array([0, 1, 1])

    result = topical_clustering_agent("st-2", "Solar and wind energy", 7)

    assert result["clusters"] == {0: [ARTICLE_A], 1: [ARTICLE_B, ARTICLE_C]}
    assert result["noise"] == []


@patch("newsresearch.agents.topical_clustering_agent.cluster")
@patch("newsresearch.agents.topical_clustering_agent.embed")
@patch("newsresearch.agents.topical_clustering_agent.sourcing_agent")
def test_handles_empty_article_set(mock_sourcing_agent, mock_embed, mock_cluster):
    mock_sourcing_agent.return_value = []
    mock_embed.return_value = np.zeros((0, 4))
    mock_cluster.return_value = np.array([], dtype=int)

    result = topical_clustering_agent("st-3", "Solar and wind energy", 7)

    assert result["clusters"] == {}
    assert result["noise"] == []
    assert result["total_articles"] == 0
