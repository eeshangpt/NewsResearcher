"""Backfill-trigger logic for Google News RSS (NFR-3 / Story 1.8).

`sourcing/google_news_backfill.py` raises on error by design (see its own
module docstring) -- it pushes soft-fail responsibility to whoever calls it.
This module is that caller: it decides *whether* backfill is needed (primary
GDELT+RSS count below `Settings.sourcing.min_primary_article_count`) and
wraps the call so a backfill failure never propagates out of the sourcing
path.
"""

from __future__ import annotations

import logging
from typing import Any

from newsresearch.config import Settings
from newsresearch.sourcing.google_news_backfill import fetch_google_news_backfill

logger = logging.getLogger(__name__)


def maybe_backfill(
    primary_articles: list[dict[str, Any]],
    keywords: list[str],
    lookback_days: int,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Invoke Google News RSS backfill only if primary results are thin.

    Returns `primary_articles` unchanged when the primary count already
    meets `Settings.sourcing.min_primary_article_count`, or when backfill is
    triggered but the call itself raises (logged and swallowed here, never
    propagated). Returns `primary_articles + backfill_articles` when backfill
    is triggered and succeeds.
    """
    settings = settings if settings is not None else Settings()
    min_primary_article_count = settings.sourcing.min_primary_article_count

    if len(primary_articles) >= min_primary_article_count:
        return primary_articles

    try:
        backfill_articles = fetch_google_news_backfill(keywords, lookback_days=lookback_days)
    except Exception:
        logger.warning(
            "backfill_trigger: google_news_backfill failed; continuing with %d primary "
            "article(s) only",
            len(primary_articles),
            exc_info=True,
        )
        return primary_articles

    return primary_articles + backfill_articles
