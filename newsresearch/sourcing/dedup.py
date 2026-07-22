"""Article deduplication (Story 1.4): URL exact-match + cross-source fuzzy
title matching.

Pure, dependency-free (no network calls, no DB) -- operates entirely on
in-memory article dicts already fetched by other Phase 1 sourcing modules
(`gdelt.py`, `rss.py`, `google_news_backfill.py`).

Expected article shape (a plain dict, other keys pass through untouched):
    {"url": str, "title": str, "domain": str, ...}
`agents/sourcing_agent.py`'s orchestrator (Story 1.9) is expected to produce
and pass around this shape.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rapidfuzz import fuzz

from newsresearch.config import Settings

# Query params that identify traffic sources rather than content -- stripped
# during URL normalization so the same article shared via different
# campaigns/social channels normalizes to one canonical URL.
_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAM_NAMES = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src"}

# Fallback if Settings.clustering.similarity_threshold is ever removed --
# this reuses that field (originally added for embedding-based clustering)
# since rapidfuzz title-similarity operates on the same 0-1 scale. If
# data-scientist later finds title-dedup needs its own tuned value, split
# it into a dedicated Settings field instead of overloading this one.
_DEFAULT_TITLE_SIMILARITY_THRESHOLD = 0.75


def _is_tracking_param(name: str) -> bool:
    lname = name.lower()
    return lname.startswith(_TRACKING_PARAM_PREFIXES) or lname in _TRACKING_PARAM_NAMES


def normalize_url(url: str) -> str:
    """Canonicalize a URL for exact-match comparison.

    Normalizes scheme to `https`, lowercases the host, strips a trailing
    slash from the path, drops tracking query params (`utm_*`, `fbclid`,
    etc.), sorts remaining query params for order-independence, and drops
    any fragment.
    """
    parsed = urlsplit(url)
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    kept_params = sorted(
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    )
    query = urlencode(kept_params)

    return urlunsplit(("https", netloc, path, query, ""))


def dedup_by_url(articles: list[dict]) -> list[dict]:
    """Drop exact duplicates after URL normalization, keeping first-seen order."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for article in articles:
        key = normalize_url(article["url"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def dedup_by_title_similarity(
    articles: list[dict],
    similarity_threshold: float | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Drop cross-source near-duplicate titles (wire-story dupes).

    Only compares articles from *different* domains -- same-domain articles
    with similar titles (e.g. a headline update) are left alone; this pass
    exists to catch the same wire story republished near-verbatim across
    outlets. `similarity_threshold` is on a 0-1 scale (rapidfuzz's 0-100
    ratio divided by 100); defaults to `Settings.clustering.similarity_threshold`.
    """
    if similarity_threshold is None:
        settings = settings or Settings()
        similarity_threshold = settings.clustering.similarity_threshold

    kept: list[dict] = []
    for article in articles:
        is_duplicate = False
        for kept_article in kept:
            if article.get("domain") == kept_article.get("domain"):
                continue
            score = fuzz.ratio(article["title"], kept_article["title"]) / 100.0
            if score >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(article)
    return kept


def dedup(
    articles: list[dict],
    similarity_threshold: float | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Full dedup pass: exact-URL drop, then cross-source fuzzy title drop."""
    deduped = dedup_by_url(articles)
    return dedup_by_title_similarity(deduped, similarity_threshold=similarity_threshold, settings=settings)
