"""Reputation signal collectors (TRD 4.2, Story 1.6).

Each collector below is independently mockable/callable and soft-fails per
NFR-3: a collector that can't produce a real signal (network error, timeout,
rate limit, absence from a snapshot) returns `None`/a neutral value rather
than raising, so a single flaky external dependency never crashes a
sourcing-agent run. `reputation/scorer.py` (Story 1.7, a later wave) is
responsible for turning these raw signals into the TRD 4.2 weighted score --
this module only collects them.
"""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import httpx
import whois

logger = logging.getLogger(__name__)


# --- Task 1.6.1: WHOIS domain-age signal -----------------------------------


def get_domain_age_years(domain: str, timeout: int = 10) -> float | None:
    """Best-effort domain age in years via WHOIS (`python-whois`).

    WHOIS is frequently rate-limited/blocked for bulk automated lookups
    (EXECUTION_PLAN Story 1.6 flagged risk) -- catches broadly and returns
    `None` on any failure (raise, timeout, unparsable response, missing
    creation date) rather than propagating, per NFR-3. Callers should cache
    aggressively (Story 1.7's `domain_reputation` cache) so this is a rare
    call, not a per-run one.
    """
    try:
        record = whois.whois(domain, timeout=timeout)
        creation_date = record.get("creation_date")
    except Exception:
        logger.warning("signals: WHOIS lookup failed for domain=%s", domain, exc_info=True)
        return None

    if isinstance(creation_date, list):
        creation_date = creation_date[0] if creation_date else None
    if creation_date is None:
        return None

    now = datetime.now(timezone.utc)
    if creation_date.tzinfo is None:
        creation_date = creation_date.replace(tzinfo=timezone.utc)
    age_days = (now - creation_date).days
    if age_days < 0:
        return None
    return age_days / 365.25


# --- Task 1.6.2: Tranco backlink/popularity-proxy signal --------------------

# Tranco is a combined-ranking *popularity/traffic* list, not a literal
# backlink-graph rank -- it's used here as a backlink/authority proxy per the
# tech-lead-resolved decision (EXECUTION_PLAN Story 1.6), not "fixed" later to
# expect literal backlink counts. The TRD 4.2 formula variable is kept as
# `backlink_proxy` as-is.
#
# This CSV is a frozen artifact refreshed manually/occasionally by
# re-downloading Tranco and re-running the top-100k truncation -- a separate
# freshness axis from `Settings.reputation.staleness_days`, which governs
# per-domain *cached score* recomputation, not this snapshot file's vintage.
# Don't conflate the two.
_TRANCO_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "tranco_top100k.csv"
_TRANCO_MAX_RANK = 100_000
_TRANCO_NEUTRAL_SCORE = 0.5

_tranco_ranks_cache: dict[str, int] | None = None


def _load_tranco_ranks() -> dict[str, int]:
    ranks: dict[str, int] = {}
    with _TRANCO_CSV_PATH.open(newline="", encoding="utf-8") as f:
        for rank_str, domain in csv.reader(f):
            ranks[domain.strip().lower()] = int(rank_str)
    return ranks


def _get_tranco_ranks() -> dict[str, int]:
    global _tranco_ranks_cache
    if _tranco_ranks_cache is None:
        _tranco_ranks_cache = _load_tranco_ranks()
    return _tranco_ranks_cache


def get_backlink_proxy_score(domain: str) -> float:
    """Tranco-derived backlink/authority proxy (TRD 4.2 `backlink_proxy`).

    Continuous log-scale decay for a domain present in the committed
    top-100k snapshot: `clip(1 - log10(rank) / log10(100_000), 0.0, 1.0)`
    (rank 1 -> 1.0, rank 10 -> 0.8, rank 1,000 -> 0.4, rank 100,000 -> 0.0).
    Returns the neutral value `0.5` (not `0.0`) for a domain absent from the
    snapshot -- absence from a global top-100k popularity list is expected
    for legitimate small/regional/niche outlets and shouldn't read as
    evidence of illegitimacy.
    """
    ranks = _get_tranco_ranks()
    rank = ranks.get(domain.strip().lower())
    if rank is None:
        return _TRANCO_NEUTRAL_SCORE

    normalized = 1 - math.log10(rank) / math.log10(_TRANCO_MAX_RANK)
    return max(0.0, min(1.0, normalized))


# --- Task 1.6.3: HTTPS + about-page legitimacy heuristic --------------------

_ABOUT_PAGE_PATHS = ("/about", "/about-us", "/en/about")


class LegitimacySignals(TypedDict):
    https_present: bool | None
    about_page_present: bool | None


def check_https_and_about_page(domain: str, timeout: float = 5.0) -> LegitimacySignals:
    """Best-effort HTTPS + about-page heuristic (TRD 4.2 `legitimacy_flags`).

    Soft-fails per NFR-3: if the homepage itself can't be reached at all
    (timeout, DNS failure, connection error), both flags come back `None`
    (neutral) rather than being treated as a legitimacy penalty. Individual
    about-page path misses (site reachable, no about page found) are a real
    negative signal and come back as `False`.
    """
    try:
        response = httpx.head(f"https://{domain}/", timeout=timeout, follow_redirects=True)
        https_present = response.status_code < 400
    except (httpx.HTTPError, OSError):
        logger.warning(
            "signals: HTTPS/about-page check failed to reach domain=%s", domain, exc_info=True
        )
        return {"https_present": None, "about_page_present": None}

    about_page_present = False
    for path in _ABOUT_PAGE_PATHS:
        try:
            about_response = httpx.head(
                f"https://{domain}{path}", timeout=timeout, follow_redirects=True
            )
            if about_response.status_code < 400:
                about_page_present = True
                break
        except (httpx.HTTPError, OSError):
            continue

    return {"https_present": https_present, "about_page_present": about_page_present}
