"""Google News RSS backfill (non-load-bearing, NFR-3 / TRD 4.2 step 2).

Wraps Google News's documented-but-unofficial RSS search endpoint
(`https://news.google.com/rss/search?q=...`). This is kept as a structurally
separate module -- not a conditionally-called function inside `rss.py` or
`gdelt.py` -- so the module boundary itself makes it obvious this is a
backfill-only source: `sourcing/gdelt.py` and `sourcing/rss.py` must never
import from this module, only a future orchestrator
(`agents/sourcing_agent.py`) does, and only when primary (GDELT + trusted
RSS) results fall below `Settings.sourcing.min_primary_article_count`
(Story 1.8).

This module does not soft-fail internally -- it raises normally on HTTP/parse
errors. The soft-fail (log-and-continue, never crash the run) behavior
belongs to the orchestrator that calls this as an optional fallback, per
NFR-3 and Story 1.8's acceptance criteria.
"""

from __future__ import annotations

from calendar import timegm
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"


def _build_query(keywords: list[str] | str, lookback_days: int | None) -> str:
    query = keywords if isinstance(keywords, str) else " ".join(keywords)
    if lookback_days is not None:
        # Google News RSS search operator: restrict results to the last N days.
        query = f"{query} when:{lookback_days}d"
    return query


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    netloc = urlparse(url).netloc
    # Lowercase + strip "www." to match the normalized domain form gdelt.py
    # and rss.py already emit (Story 1.9 integration checkpoint) -- without
    # this, dedup.py's cross-source-dedup same-domain guard (plain string
    # equality on `domain`) can silently misfire against a mixed-case source
    # href for the same real-world outlet.
    return netloc.lower().removeprefix("www.") or None


def _entry_published_at(entry: dict[str, Any]) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)


def _entry_title(entry: dict[str, Any], publisher_name: str | None) -> str | None:
    title = entry.get("title")
    # Google News suffixes each title with " - <publisher>"; strip it when
    # the publisher name is known so callers get the article's own title.
    if title and publisher_name and title.endswith(f" - {publisher_name}"):
        return title[: -len(f" - {publisher_name}")]
    return title


def fetch_google_news_backfill(
    keywords: list[str] | str,
    lookback_days: int | None = None,
    *,
    language: str = "en-US",
    country: str = "US",
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch Google News RSS search results for `keywords`.

    Independently callable -- has no dependency on `sourcing/rss.py` or
    `sourcing/gdelt.py`, and neither of those modules imports this one (see
    module docstring). Returns article dicts: `title`, `url` (Google News's
    redirect URL, not the publisher's canonical URL), `domain` (resolved
    from the feed's `<source>` tag), `published_at` (timezone-aware UTC
    `datetime`), `source_type` (`"google_news_backfill"`), `publisher_name`.
    """
    params = {
        "q": _build_query(keywords, lookback_days),
        "hl": language,
        "gl": country,
        "ceid": f"{country}:{language.split('-')[0]}",
    }
    response = httpx.get(
        GOOGLE_NEWS_RSS_URL,
        params=params,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "newsresearch/0.1"},
    )
    response.raise_for_status()
    parsed_feed = feedparser.parse(response.content)

    articles: list[dict[str, Any]] = []
    for entry in parsed_feed.entries:
        source = entry.get("source") or {}
        publisher_name = source.get("title")
        articles.append(
            {
                "title": _entry_title(entry, publisher_name),
                "url": entry.get("link"),
                "domain": _domain_from_url(source.get("href")),
                "published_at": _entry_published_at(entry),
                "source_type": "google_news_backfill",
                "publisher_name": publisher_name,
            }
        )

    return articles
