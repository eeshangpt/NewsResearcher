"""In-memory full-text article fetch (Task 3.1.1, Story 3.1).

`trafilatura`-based body-text extraction per article URL. Deliberately has
zero imports from `persistence/` -- this is in-memory-only for now (no new
DB table, no write path); Story 3.5 is where claim clusters (which this
feeds, downstream via Task 3.2.1's claim extraction) actually get persisted.

Soft-fails per NFR-3 (matching `reputation/signals.py`'s pattern): a single
unreachable or unparseable article returns `None` rather than raising, so one
bad URL in a cluster never crashes full-text fetch for the rest of the batch.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import trafilatura
from trafilatura.settings import use_config

from newsresearch.config import Settings

logger = logging.getLogger(__name__)


class ArticleFulltext(TypedDict):
    url: str
    fulltext: str | None


def _trafilatura_config(timeout_seconds: float) -> Any:
    """`trafilatura.fetch_url()` takes a request timeout only via its own
    `ConfigParser`, not a direct kwarg (issue #108: previously unbounded)."""
    config = use_config()
    config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout_seconds))
    return config


def fetch_fulltext(url: str, settings: Settings | None = None) -> str | None:
    """Best-effort full article body text for a single URL.

    Returns `None` (never raises) if the page can't be downloaded (including
    on timeout -- `trafilatura.fetch_url()` itself never raises for a failed
    download, it returns `None`), or if `trafilatura` can't extract a body
    (paywall, non-article page, empty extraction result).
    """
    settings = settings or Settings()

    try:
        downloaded = trafilatura.fetch_url(
            url,
            config=_trafilatura_config(settings.sourcing.fulltext_fetch_timeout_seconds),
        )
    except Exception:
        logger.warning("fulltext: download failed for url=%s", url, exc_info=True)
        return None

    if not downloaded:
        logger.warning("fulltext: no content downloaded for url=%s", url)
        return None

    try:
        text = trafilatura.extract(downloaded, url=url)
    except Exception:
        logger.warning("fulltext: extraction failed for url=%s", url, exc_info=True)
        return None

    return text or None


def fetch_fulltext_for_cluster(
    articles: list[dict[str, Any]], settings: Settings | None = None
) -> list[ArticleFulltext]:
    """Fetch full text for every article in a Gate-2-cleared topical cluster.

    `articles` is the same `{"url": str, "title": str, "domain": str, ...}`
    shape `sourcing/dedup.py` and `reports/gate2_report.py` already operate
    on. One unreachable article never drops the rest of the batch -- each
    entry's `fulltext` is independently `None` on failure.
    """
    settings = settings or Settings()
    return [
        {"url": article["url"], "fulltext": fetch_fulltext(article["url"], settings)}
        for article in articles
    ]