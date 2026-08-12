"""DuckDuckGo + Tavily web-search discovery (replaces GDELT as sourcing_agent's
primary discovery source -- issue #101, GDELT is persistently IP-blocked).

Ports the design validated in
`notebooks/experiment_gdelt_vs_websearch_sourcing.py` (data-scientist,
feat/datascientist) into production: `DuckDuckGoSearchResults` (free, no key)
+ `TavilySearch` (needs `TAVILY_API_KEY`, skipped if absent) for discovery,
then this project's real `sourcing/fulltext.py::fetch_fulltext` to confirm
each candidate URL is actually fetchable news content (no raw vendor
trafilatura wiring here -- see `fulltext.py`'s own docstring for the
soft-fail/timeout contract). Full text itself is never returned or stored
(matches the no-full-text-storage-until-Story-3.5 rule) -- it's fetched only
to filter out dead links/paywalled/non-article results, same quality bar
GDELT/RSS implicitly clear by only returning confirmed article records.

Soft-fail per NFR-3, matching `gdelt.py`/`google_news_backfill.py`'s
convention: `WebSearchError` is the single exception type search-tool/network
failures get normalized into; callers (`sourcing_agent.py`) catch it and
continue on RSS/backfill alone.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

from newsresearch.config import Settings
from newsresearch.sourcing.fulltext import fetch_fulltext

logger = logging.getLogger(__name__)

# Tavily's `time_range` param is a coarse bucket (day/week/month/year), not an
# arbitrary day count -- `lookback_days` is mapped to the nearest bucket that
# doesn't undershoot it. DuckDuckGo's langchain wrapper has no date-range
# param at all (same limitation the reference notebook hit), so DDG results
# are unfiltered by lookback_days.
# ponytail: coarse day->bucket mapping, not exact; fine at these thresholds,
# revisit if a task ever needs precise lookback windows on the Tavily leg.
def _tavily_time_range(lookback_days: int) -> str:
    if lookback_days <= 1:
        return "day"
    if lookback_days <= 7:
        return "week"
    if lookback_days <= 31:
        return "month"
    return "year"


class WebSearchError(Exception):
    """Raised when web-search discovery fails for a non-recoverable reason."""


def _domain_of(url: str) -> str | None:
    domain = urlsplit(url).netloc.lower().removeprefix("www.")
    return domain or None


def _ddg_search(query: str, max_results: int) -> list[dict]:
    from langchain_community.tools import DuckDuckGoSearchResults

    tool = DuckDuckGoSearchResults(output_format="list", max_results=max_results)
    raw = tool.invoke(query)
    return raw if isinstance(raw, list) else []


def _tavily_search(query: str, max_results: int, lookback_days: int, api_key: str) -> list[dict]:
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=max_results, topic="news", tavily_api_key=api_key)
    response = tool.invoke({"query": query, "time_range": _tavily_time_range(lookback_days)})
    results = response.get("results", []) if isinstance(response, dict) else []
    return results if isinstance(results, list) else []


def fetch(
    keywords: list[str],
    lookback_days: int,
    *,
    max_results: int = 15,
    settings: Settings | None = None,
) -> list[dict]:
    """Discover + full-text-verify news articles for `keywords`.

    Same call shape as `gdelt.py::fetch()`: keyword list + lookback window
    in, a list of `{title, url, domain, published_at, source_type}` article
    dicts out. DuckDuckGo and Tavily are each individually soft-failed to an
    empty result (one leg's outage doesn't drop the other, and a missing
    `TAVILY_API_KEY` just skips that leg) -- `WebSearchError` is raised only
    if *both* legs fail/are unavailable, matching `gdelt.py`'s "single
    exception type for any operational failure" convention for
    `sourcing_agent.py` to catch.
    """
    settings = settings if settings is not None else Settings()
    query = " ".join(keywords)

    candidates: dict[str, dict] = {}
    ddg_failed = tavily_failed = False

    try:
        for r in _ddg_search(query, max_results):
            url = r.get("link") or r.get("url")
            if not url:
                continue
            candidates.setdefault(
                url, {"title": r.get("title"), "url": url, "domain": _domain_of(url)}
            )
    except Exception:
        logger.warning("web_search: DuckDuckGo search failed", exc_info=True)
        ddg_failed = True

    tavily_key = (settings.tavily_api_key or "").strip()
    if tavily_key:
        try:
            for r in _tavily_search(query, max_results, lookback_days, tavily_key):
                url = r.get("url")
                if not url:
                    continue
                candidates.setdefault(
                    url, {"title": r.get("title"), "url": url, "domain": _domain_of(url)}
                )
        except Exception:
            logger.warning("web_search: Tavily search failed", exc_info=True)
            tavily_failed = True
    else:
        logger.info("web_search: TAVILY_API_KEY not set, Tavily leg skipped")
        tavily_failed = True

    if ddg_failed and tavily_failed:
        raise WebSearchError("both DuckDuckGo and Tavily search legs failed or were unavailable")

    articles: list[dict[str, Any]] = []
    for candidate in candidates.values():
        if not candidate["url"] or not candidate["title"] or not candidate["domain"]:
            continue
        if not fetch_fulltext(candidate["url"], settings):
            continue
        articles.append(
            {
                "title": candidate["title"],
                "url": candidate["url"],
                "domain": candidate["domain"],
                "published_at": None,
                "source_type": "web_search",
            }
        )

    logger.info(
        "web_search: %d article(s) discovered and full-text-verified for keywords=%r",
        len(articles),
        keywords,
    )
    return articles
