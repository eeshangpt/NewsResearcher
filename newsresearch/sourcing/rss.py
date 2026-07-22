"""Direct outlet RSS sourcing (TRD 4.2 step 1: trusted-tier outlet feeds).

Fetches each trusted-tier outlet's official RSS feed and filters entries by
keyword + lookback window. `data/trusted_outlets.yaml` only maps
domain -> tier, not feed URLs -- outlets don't expose a discoverable
feed-URL convention derivable from the domain alone -- so `OUTLET_RSS_FEEDS`
below hardcodes the actual public RSS endpoint for the subset of outlets
that still run a stable one (several trusted-tier wire services, e.g.
Reuters/AP, discontinued their public RSS feeds; domains with no known feed
URL are simply skipped, not an error).

Deliberately does not import `sourcing/google_news_backfill.py` -- Google
News backfill is invoked only by a future orchestrator
(`agents/sourcing_agent.py`), never by this module, per NFR-3 and Story
1.3's module-boundary requirement.
"""

from __future__ import annotations

import logging
from calendar import timegm
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
import yaml

logger = logging.getLogger(__name__)

TRUSTED_OUTLETS_PATH = Path(__file__).resolve().parent.parent / "data" / "trusted_outlets.yaml"

# Official RSS feed URL per trusted-tier domain. Sourced by hand from each
# outlet's own published RSS index page -- not derivable from
# `trusted_outlets.yaml` (which only has domain -> tier), and not every
# domain in that file has a surviving public feed, so this map is
# intentionally a subset.
OUTLET_RSS_FEEDS: dict[str, str] = {
    "bbc.com": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "theguardian.com": "https://www.theguardian.com/world/rss",
    "npr.org": "https://feeds.npr.org/1001/rss.xml",
    "aljazeera.com": "https://www.aljazeera.com/xml/rss/all.xml",
}


def load_trusted_domains() -> dict[str, str]:
    """Domain -> tier mapping from `data/trusted_outlets.yaml`."""
    return yaml.safe_load(TRUSTED_OUTLETS_PATH.read_text())


def _entry_published_at(entry: dict[str, Any]) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc)


def _matches_keywords(entry: dict[str, Any], keywords: list[str]) -> bool:
    haystack = f"{entry.get('title', '')} {entry.get('summary', '')}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def fetch_feed(feed_url: str, timeout: float = 10.0) -> feedparser.FeedParserDict:
    """Fetch and parse a single RSS feed.

    Goes through `httpx` (rather than feedparser's own URL-fetching) so all
    outbound HTTP in this module is made through the same client/timeout/
    header conventions used elsewhere, and so tests can mock at the `httpx`
    layer.
    """
    response = httpx.get(
        feed_url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "newsresearch/0.1"},
    )
    response.raise_for_status()
    return feedparser.parse(response.content)


def fetch_trusted_rss(
    keywords: list[str] | str,
    lookback_days: int,
    *,
    feeds: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch trusted-tier outlet RSS feeds, filtered by keyword + lookback.

    `keywords` is matched case-insensitively as a substring against each
    entry's title+summary -- real outlet RSS feeds aren't keyword-searchable
    themselves (unlike Google News's search endpoint), so filtering happens
    client-side after fetch. A per-outlet fetch failure is logged and
    skipped rather than aborting the whole call, so one feed being
    temporarily down doesn't take out every other trusted outlet.

    Returns article dicts: `title`, `url`, `domain`, `published_at`
    (timezone-aware UTC `datetime`), `source_type` (`"rss"`).
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    feeds = feeds if feeds is not None else OUTLET_RSS_FEEDS
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    articles: list[dict[str, Any]] = []
    for domain, feed_url in feeds.items():
        try:
            parsed_feed = fetch_feed(feed_url, timeout=timeout)
        except Exception:
            logger.warning(
                "rss: failed to fetch feed for %s (%s); skipping", domain, feed_url, exc_info=True
            )
            continue

        for entry in parsed_feed.entries:
            published_at = _entry_published_at(entry)
            if published_at is None or published_at < cutoff:
                continue
            if not _matches_keywords(entry, keywords):
                continue
            articles.append(
                {
                    "title": entry.get("title"),
                    "url": entry.get("link"),
                    "domain": domain,
                    "published_at": published_at,
                    "source_type": "rss",
                }
            )

    return articles
