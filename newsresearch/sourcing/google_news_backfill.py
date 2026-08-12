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

This module does not soft-fail internally on the top-level RSS search
request -- it raises normally on HTTP/parse errors there. The soft-fail
(log-and-continue, never crash the run) behavior for *that* belongs to the
orchestrator that calls this as an optional fallback, per NFR-3 and Story
1.8's acceptance criteria.

Per-article real-URL resolution (`_resolve_real_article_url`, issue #115) is
the one exception: it soft-fails internally, per article, because a single
unresolvable article must not block the rest of the batch, and there is no
useful fallback value to return for one (the raw redirect-shell URL is
useless to `sourcing/fulltext.py`) -- see that function's docstring.
"""

from __future__ import annotations

import json
import logging
import re
from calendar import timegm
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

import feedparser
import httpx

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# issue #115: Google's RSS `<link>` is a `news.google.com/rss/articles/...`
# redirect-shell that resolves the real article client-side via JS -- there
# is no server-side HTTP redirect for httpx/trafilatura to follow. This
# replays the same signed request Google's own JS makes: fetch the shell
# page for its per-request signature/timestamp (`data-n-a-sg`/`data-n-a-ts`),
# then POST that to Google's internal batchexecute endpoint, which returns
# the real article URL. No new dependency -- stdlib `re`/`json` + the
# already-installed `httpx`.
_GOOGLE_NEWS_BATCHEXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
_SIGNATURE_RE = re.compile(r'data-n-a-sg="([^"]+)"')
_TIMESTAMP_RE = re.compile(r'data-n-a-ts="([^"]+)"')


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


def _base64_id_from_redirect_url(redirect_url: str) -> str | None:
    path_parts = urlparse(redirect_url).path.rstrip("/").split("/")
    if len(path_parts) >= 2 and path_parts[-2] in ("articles", "read"):
        return path_parts[-1]
    return None


def _resolve_real_article_url(redirect_url: str, *, timeout: float) -> str | None:
    """Resolve a Google News RSS redirect-shell URL to the publisher's real
    article URL (issue #115). Returns `None` (never raises) on any
    network/parse failure -- callers drop the article rather than pass on a
    redirect-shell URL that `trafilatura` cannot extract text from.
    """
    base64_id = _base64_id_from_redirect_url(redirect_url)
    if base64_id is None:
        return None

    try:
        shell_response = httpx.get(
            redirect_url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "newsresearch/0.1"},
        )
        shell_response.raise_for_status()
        signature_match = _SIGNATURE_RE.search(shell_response.text)
        timestamp_match = _TIMESTAMP_RE.search(shell_response.text)
        if not signature_match or not timestamp_match:
            return None
        signature, timestamp = signature_match.group(1), timestamp_match.group(1)

        # Payload shape/endpoint reverse-engineered from Google News' own
        # front-end JS -- matches what a real browser sends when resolving
        # this same redirect-shell page.
        garturlreq_body = (
            '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,'
            "null,null,null,null,null,0,1],\"X\",\"X\",1,[1,1,1],1,1,null,0,0,null,0],"
            f'"{base64_id}",{timestamp},"{signature}"]'
        )
        rpc_payload = ["Fbv4je", garturlreq_body]
        batch_response = httpx.post(
            _GOOGLE_NEWS_BATCHEXECUTE_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "User-Agent": "newsresearch/0.1",
            },
            content=f"f.req={quote(json.dumps([[rpc_payload]]))}",
            timeout=timeout,
        )
        batch_response.raise_for_status()
        # Response body is `)]}'\n\n<json array>\n<footer lines>` -- strip
        # the anti-hijacking prefix and trailing footer rows before parsing.
        rpc_body = json.loads(batch_response.text.split("\n\n")[1])[:-2]
        return json.loads(rpc_body[0][2])[1]
    except Exception:
        logger.warning(
            "google_news_backfill: could not resolve real url for redirect=%s",
            redirect_url,
            exc_info=True,
        )
        return None


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
    module docstring). Returns article dicts: `title`, `url` (the
    publisher's real article URL, resolved from Google's redirect-shell link
    per `_resolve_real_article_url`, issue #115), `domain` (resolved from the
    feed's `<source>` tag), `published_at` (timezone-aware UTC `datetime`),
    `source_type` (`"google_news_backfill"`), `publisher_name`. An entry
    whose real URL can't be resolved is dropped from the returned list
    (logged, not raised) rather than passed on with an unfetchable
    redirect-shell URL.
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
        redirect_url = entry.get("link")
        real_url = (
            _resolve_real_article_url(redirect_url, timeout=timeout) if redirect_url else None
        )
        if real_url is None:
            logger.warning(
                "google_news_backfill: dropping entry, could not resolve real url "
                "(redirect=%s)",
                redirect_url,
            )
            continue

        source = entry.get("source") or {}
        publisher_name = source.get("title")
        articles.append(
            {
                "title": _entry_title(entry, publisher_name),
                "url": real_url,
                "domain": _domain_from_url(source.get("href")),
                "published_at": _entry_published_at(entry),
                "source_type": "google_news_backfill",
                "publisher_name": publisher_name,
            }
        )

    return articles
