"""Sourcing Agent: end-to-end Phase 1 orchestration (TRD 4.2, Story 1.9).

Composes every other Phase 1 module in the order TRD 4.2 lays out:

    query (GDELT + trusted RSS)
      -> backfill-if-below-min-count (Google News RSS, NFR-3 soft-fail)
      -> dedup (exact-URL + cross-source fuzzy title)
      -> score each surviving domain's reputation (cache-first)
      -> filter to `Settings.reputation.min_score_threshold`

This is the Phase 1 phase-level "Done when" target itself
(EXECUTION_PLAN.md Story 1.9): given a hardcoded sample subtopic (no
Subtopic Agent yet -- that's Phase 2), calling `sourcing_agent(...)` against
real GDELT/RSS returns a deduplicated, reputation-scored, threshold-filtered
article list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.reputation import cache, scorer, signals
from newsresearch.sourcing import dedup as dedup_module
from newsresearch.sourcing import gdelt, rss
from newsresearch.sourcing.backfill_trigger import maybe_backfill

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredArticle:
    """An article dict (from `gdelt.py`/`rss.py`/`google_news_backfill.py`)
    plus its domain's reputation score, via the cache-first path in
    `reputation/cache.py` + `reputation/scorer.py`.

    Kept as a thin wrapper around the article dict -- rather than a fixed
    fielded dataclass -- because each sourcing module's article dict carries
    a slightly different shape (e.g. `google_news_backfill.py`'s extra
    `publisher_name`); wrapping avoids lossy field-by-field copying.
    """

    article: dict[str, Any]
    reputation_score: float
    reputation_tier: str


def _has_required_fields(article: dict[str, Any]) -> bool:
    """An article dict must carry a url/title/domain to be dedup-/score-able.

    A handful of malformed feed entries (missing link/title, or a Google
    News `<source>` tag with no resolvable domain) are common enough in
    practice across all three sourcing modules to filter defensively here,
    rather than letting `dedup.py`'s `article["url"]` indexing raise on one
    bad entry.
    """
    return bool(article.get("url")) and bool(article.get("title")) and bool(article.get("domain"))


def _get_or_score_domain_reputation(
    pool: ConnectionPool,
    domain: str,
    presence_frequency: float | None,
    settings: Settings,
) -> cache.CachedDomainReputation | scorer.DomainReputationScore:
    """Cache-first per-domain reputation lookup (Story 1.9 step 4).

    A fresh cached score is used as-is; a cache miss/stale entry falls back
    to collecting the four `reputation/signals.py` signals, scoring via
    `reputation/scorer.py::score_domain`, and writing the result back to the
    cache so the next run (within `Settings.reputation.staleness_days`)
    doesn't repeat the WHOIS/HTTPS network calls.
    """
    cached = cache.get_fresh_domain_reputation(pool, domain, settings=settings)
    if cached is not None:
        return cached

    domain_age_years = signals.get_domain_age_years(domain)
    backlink_proxy = signals.get_backlink_proxy_score(domain)
    legitimacy = signals.check_https_and_about_page(domain)

    score = scorer.score_domain(
        domain,
        domain_age_years=domain_age_years,
        backlink_proxy=backlink_proxy,
        presence_frequency=presence_frequency,
        https_present=legitimacy["https_present"],
        about_page_present=legitimacy["about_page_present"],
        settings=settings,
    )
    cache.write_domain_reputation(pool, score)
    return score


def sourcing_agent(
    keywords: list[str],
    lookback_days: int,
    *,
    pool: ConnectionPool | None = None,
    settings: Settings | None = None,
) -> list[ScoredArticle]:
    """Query -> backfill-if-thin -> dedup -> score -> threshold-filter.

    `pool` defaults to a new connection pool opened (and closed before
    returning) against `Settings.database_url` via
    `persistence/db.py::init_db` -- pass an existing pool (e.g. from a
    long-lived caller, or a `testcontainers` fixture in tests) to reuse one
    instead.

    Returns a `ScoredArticle` for every deduped article whose domain's
    reputation score is `>= Settings.reputation.min_score_threshold`.
    """
    settings = settings if settings is not None else Settings()

    owns_pool = pool is None
    if owns_pool:
        if not settings.database_url:
            raise ValueError(
                "sourcing_agent needs a Postgres connection for the reputation cache -- "
                "set NEWSRESEARCH_DATABASE_URL or pass an existing `pool`."
            )
        pool = init_db(settings.database_url)

    try:
        gdelt_articles = gdelt.fetch(keywords, lookback_days)
        rss_articles = rss.fetch_trusted_rss(keywords, lookback_days)
        primary_articles = gdelt_articles + rss_articles
        logger.info(
            "sourcing_agent: %d primary article(s) (%d gdelt, %d rss) for keywords=%r",
            len(primary_articles),
            len(gdelt_articles),
            len(rss_articles),
            keywords,
        )

        combined_articles = maybe_backfill(primary_articles, keywords, lookback_days, settings=settings)

        valid_articles = [article for article in combined_articles if _has_required_fields(article)]
        deduped_articles = dedup_module.dedup(valid_articles, settings=settings)
        logger.info(
            "sourcing_agent: %d article(s) survive dedup (from %d combined)",
            len(deduped_articles),
            len(combined_articles),
        )

        presence_frequency_scores = signals.get_presence_frequency_scores(deduped_articles)

        domain_reputations: dict[str, cache.CachedDomainReputation | scorer.DomainReputationScore] = {}
        for domain in {article["domain"] for article in deduped_articles}:
            domain_reputations[domain] = _get_or_score_domain_reputation(
                pool, domain, presence_frequency_scores.get(domain), settings
            )

        scored_articles: list[ScoredArticle] = []
        for article in deduped_articles:
            reputation = domain_reputations[article["domain"]]
            if reputation["final_score"] >= settings.reputation.min_score_threshold:
                scored_articles.append(
                    ScoredArticle(
                        article=article,
                        reputation_score=reputation["final_score"],
                        reputation_tier=reputation["tier"],
                    )
                )

        logger.info(
            "sourcing_agent: %d article(s) meet min_score_threshold=%.2f",
            len(scored_articles),
            settings.reputation.min_score_threshold,
        )
        return scored_articles
    finally:
        if owns_pool:
            pool.close()
