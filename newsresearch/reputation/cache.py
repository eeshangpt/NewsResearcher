"""`domain_reputation` cache read/write (Story 1.7, Task 1.7.2).

Follows `persistence/db.py`'s connection-handling convention: every function
here takes an already-open `ConnectionPool` (from `init_db`) and opens/closes
its own short-lived connection per call via `pool.connection()` -- this
module never owns a pool itself.

Staleness is a read-time check against `Settings.reputation.staleness_days`,
not a write-time TTL: a cached row is considered fresh if `computed_at` is
within the staleness window of "now", and callers (`agents/sourcing_agent.py`,
Story 1.9) are expected to recompute via `reputation/scorer.py` and call
`write_domain_reputation` again whenever `get_fresh_domain_reputation`
returns `None`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypedDict

from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.reputation.scorer import DomainReputationScore

# Deliberately far in the past (well before this project exists) rather than
# `datetime.min` -- avoids any timezone-arithmetic edge case at the extreme
# end of `datetime`'s range while still being unconditionally "stale" against
# any realistic `staleness_days` setting.
_INVALIDATED_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class CachedDomainReputation(TypedDict):
    domain: str
    tier: str
    base_score: float
    heuristic_adjustment: float
    final_score: float
    computed_at: datetime


def write_domain_reputation(
    pool: ConnectionPool,
    score: DomainReputationScore,
    computed_at: datetime | None = None,
) -> None:
    """Insert or overwrite a domain's cached reputation score.

    `computed_at` defaults to now (UTC); tests can pass an explicit value
    (or rely on `freeze_time`) to control staleness-window boundaries.
    """
    computed_at = computed_at if computed_at is not None else datetime.now(timezone.utc)
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO domain_reputation
                (domain, tier, base_score, heuristic_adjustment, final_score, computed_at)
            VALUES
                (%(domain)s, %(tier)s, %(base_score)s, %(heuristic_adjustment)s,
                 %(final_score)s, %(computed_at)s)
            ON CONFLICT (domain) DO UPDATE SET
                tier = EXCLUDED.tier,
                base_score = EXCLUDED.base_score,
                heuristic_adjustment = EXCLUDED.heuristic_adjustment,
                final_score = EXCLUDED.final_score,
                computed_at = EXCLUDED.computed_at
            """,
            {
                "domain": score["domain"],
                "tier": score["tier"],
                "base_score": score["base_score"],
                "heuristic_adjustment": score["heuristic_adjustment"],
                "final_score": score["final_score"],
                "computed_at": computed_at,
            },
        )


def read_domain_reputation(pool: ConnectionPool, domain: str) -> CachedDomainReputation | None:
    """Raw cache read, regardless of staleness -- `None` if never cached.

    Most callers want `get_fresh_domain_reputation` instead, which also
    applies the staleness-window check; this is exposed separately for
    inspection/debugging and because `get_fresh_domain_reputation` is built
    on top of it.
    """
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT domain, tier, base_score, heuristic_adjustment, final_score, computed_at
            FROM domain_reputation WHERE domain = %s
            """,
            (domain.strip().lower(),),
        ).fetchone()

    if row is None:
        return None
    return {
        "domain": row[0],
        "tier": row[1],
        "base_score": row[2],
        "heuristic_adjustment": row[3],
        "final_score": row[4],
        "computed_at": row[5],
    }


def is_stale(
    computed_at: datetime,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether `computed_at` falls outside `Settings.reputation.staleness_days`.

    `now` defaults to `datetime.now(timezone.utc)` -- tests use `freeze_time`
    to control it precisely at the staleness-window boundary rather than
    passing `now` explicitly, matching this project's existing convention
    (e.g. `sourcing/backfill_trigger.py`'s `settings: Settings | None = None`).
    """
    settings = settings if settings is not None else Settings()
    now = now if now is not None else datetime.now(timezone.utc)
    if computed_at.tzinfo is None:
        computed_at = computed_at.replace(tzinfo=timezone.utc)
    return (now - computed_at) > timedelta(days=settings.reputation.staleness_days)


def get_fresh_domain_reputation(
    pool: ConnectionPool,
    domain: str,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> CachedDomainReputation | None:
    """Cached score if present and within the staleness window, else `None`.

    A `None` return means "cache miss or stale" -- the caller should
    recompute via `reputation/scorer.py::score_domain` and write the result
    back with `write_domain_reputation`. This is the single read entry point
    `agents/sourcing_agent.py` (Story 1.9) is expected to use.
    """
    cached = read_domain_reputation(pool, domain)
    if cached is None:
        return None
    if is_stale(cached["computed_at"], settings=settings, now=now):
        return None
    return cached


def invalidate_domain_reputation(pool: ConnectionPool, domain: str) -> None:
    """Force the next lookup to recompute, regardless of the staleness window.

    Sets `computed_at` to a fixed far-past epoch rather than deleting the
    row: `articles.domain` carries a foreign key against
    `domain_reputation.domain` (schema.sql), so a `DELETE` would raise on any
    domain that already has recorded articles. Rewinding `computed_at` has
    the same practical effect -- `is_stale`/`get_fresh_domain_reputation`
    treat it as unconditionally stale -- without that FK risk, and leaves the
    domain's last-known tier/score queryable until the next successful
    recompute overwrites it. A no-op (0 rows affected) if the domain was
    never cached.
    """
    with pool.connection() as conn:
        conn.execute(
            "UPDATE domain_reputation SET computed_at = %s WHERE domain = %s",
            (_INVALIDATED_EPOCH, domain.strip().lower()),
        )
