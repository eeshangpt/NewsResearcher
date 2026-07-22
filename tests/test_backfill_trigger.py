from unittest.mock import patch

from newsresearch.config import Settings
from newsresearch.sourcing.backfill_trigger import maybe_backfill

SETTINGS = Settings(sourcing={"min_primary_article_count": 10})


def _articles(n: int, source_type: str = "gdelt") -> list[dict]:
    return [{"title": f"article {i}", "url": f"https://example.com/{i}", "source_type": source_type} for i in range(n)]


@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
def test_backfill_invoked_when_primary_count_below_threshold(mock_fetch):
    mock_fetch.return_value = _articles(3, source_type="google_news_backfill")
    primary = _articles(5)

    result = maybe_backfill(primary, ["ukraine"], lookback_days=7, settings=SETTINGS)

    mock_fetch.assert_called_once_with(["ukraine"], lookback_days=7)
    assert len(result) == 8
    assert result[:5] == primary
    assert all(article["source_type"] == "google_news_backfill" for article in result[5:])


@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
def test_backfill_not_invoked_when_primary_count_at_threshold(mock_fetch):
    primary = _articles(10)

    result = maybe_backfill(primary, ["ukraine"], lookback_days=7, settings=SETTINGS)

    mock_fetch.assert_not_called()
    assert result == primary


@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
def test_backfill_not_invoked_when_primary_count_above_threshold(mock_fetch):
    primary = _articles(15)

    result = maybe_backfill(primary, ["ukraine"], lookback_days=7, settings=SETTINGS)

    mock_fetch.assert_not_called()
    assert result == primary


@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
def test_backfill_failure_logs_and_continues_with_primary_only(mock_fetch, caplog):
    mock_fetch.side_effect = RuntimeError("simulated Google News RSS failure")
    primary = _articles(2)

    with caplog.at_level("WARNING"):
        result = maybe_backfill(primary, ["ukraine"], lookback_days=7, settings=SETTINGS)

    assert result == primary
    assert "google_news_backfill failed" in caplog.text


@patch("newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill")
def test_default_settings_used_when_none_provided(mock_fetch):
    mock_fetch.return_value = []
    primary = _articles(2)

    result = maybe_backfill(primary, ["ukraine"], lookback_days=7)

    mock_fetch.assert_called_once()
    assert result == primary
