from datetime import datetime, timedelta, timezone

import pytest
from freezegun import freeze_time
from testcontainers.postgres import PostgresContainer

from newsresearch.config import Settings
from newsresearch.persistence.db import init_db
from newsresearch.reputation import cache
from newsresearch.reputation.scorer import DomainReputationScore

# Hermetic `testcontainers[postgres]`, per Story 0.4/Story 1.7's own
# precedent -- no dependency on the dev `docker compose up -d` stack being up.

SETTINGS = Settings(reputation={"staleness_days": 30})


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


@pytest.fixture
def pool(postgres_url):
    pool = init_db(postgres_url)
    yield pool
    pool.close()


def _score(domain: str = "reuters.com") -> DomainReputationScore:
    return {
        "domain": domain,
        "tier": "wire",
        "base_score": 0.8,
        "heuristic_adjustment": 0.1,
        "final_score": 0.9,
    }


# --- write + read round-trip -------------------------------------------------


def test_write_then_read_round_trips_all_fields(pool):
    computed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cache.write_domain_reputation(pool, _score(), computed_at=computed_at)

    row = cache.read_domain_reputation(pool, "reuters.com")

    assert row == {
        "domain": "reuters.com",
        "tier": "wire",
        "base_score": 0.8,
        "heuristic_adjustment": 0.1,
        "final_score": 0.9,
        "computed_at": computed_at,
    }


def test_read_domain_reputation_returns_none_for_uncached_domain(pool):
    assert cache.read_domain_reputation(pool, "never-scored.example") is None


def test_write_domain_reputation_upserts_on_conflict(pool):
    cache.write_domain_reputation(
        pool, _score("upsert-test.example"), computed_at=datetime(2020, 1, 1, tzinfo=timezone.utc)
    )
    updated_score: DomainReputationScore = {
        "domain": "upsert-test.example",
        "tier": "major",
        "base_score": 0.7,
        "heuristic_adjustment": 0.2,
        "final_score": 0.9,
    }
    new_computed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    cache.write_domain_reputation(pool, updated_score, computed_at=new_computed_at)

    row = cache.read_domain_reputation(pool, "upsert-test.example")
    assert row["tier"] == "major"
    assert row["computed_at"] == new_computed_at


# --- staleness-window cache hit / miss ---------------------------------------


def test_get_fresh_domain_reputation_is_a_cache_hit_within_staleness_window(pool):
    with freeze_time("2026-01-01 00:00:00"):
        cache.write_domain_reputation(pool, _score("fresh.example"))

    with freeze_time("2026-01-15 00:00:00"):  # 14 days later, within 30-day window
        result = cache.get_fresh_domain_reputation(pool, "fresh.example", settings=SETTINGS)

    assert result is not None
    assert result["domain"] == "fresh.example"


def test_get_fresh_domain_reputation_is_none_when_stale(pool):
    with freeze_time("2026-01-01 00:00:00"):
        cache.write_domain_reputation(pool, _score("stale.example"))

    with freeze_time("2026-03-01 00:00:00"):  # ~59 days later, beyond 30-day window
        result = cache.get_fresh_domain_reputation(pool, "stale.example", settings=SETTINGS)

    assert result is None


def test_get_fresh_domain_reputation_is_none_for_never_cached_domain(pool):
    result = cache.get_fresh_domain_reputation(pool, "not-cached.example", settings=SETTINGS)
    assert result is None


# --- freezegun-controlled staleness-window boundary, precisely --------------


def test_is_stale_boundary_exactly_at_staleness_days_is_not_yet_stale():
    computed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with freeze_time(computed_at + timedelta(days=30)):
        assert cache.is_stale(computed_at, settings=SETTINGS) is False


def test_is_stale_boundary_one_second_past_staleness_days_is_stale():
    computed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with freeze_time(computed_at + timedelta(days=30, seconds=1)):
        assert cache.is_stale(computed_at, settings=SETTINGS) is True


def test_is_stale_boundary_one_second_before_staleness_days_is_not_stale():
    computed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with freeze_time(computed_at + timedelta(days=30, seconds=-1)):
        assert cache.is_stale(computed_at, settings=SETTINGS) is False


# --- manual invalidation forces recompute ------------------------------------


def test_invalidate_domain_reputation_forces_recompute_even_within_staleness_window(pool):
    with freeze_time("2026-01-01 00:00:00"):
        cache.write_domain_reputation(pool, _score("invalidate-me.example"))

    with freeze_time("2026-01-02 00:00:00"):  # 1 day later, well within window
        # Still fresh before invalidation.
        assert cache.get_fresh_domain_reputation(pool, "invalidate-me.example", settings=SETTINGS) is not None

        cache.invalidate_domain_reputation(pool, "invalidate-me.example")

        # Same instant, same staleness window -- now forced stale by manual invalidation.
        assert cache.get_fresh_domain_reputation(pool, "invalidate-me.example", settings=SETTINGS) is None

    # A fresh recompute-and-write after invalidation is a cache hit again.
    with freeze_time("2026-01-03 00:00:00"):
        cache.write_domain_reputation(pool, _score("invalidate-me.example"))
        assert cache.get_fresh_domain_reputation(pool, "invalidate-me.example", settings=SETTINGS) is not None


def test_invalidate_domain_reputation_is_a_noop_for_never_cached_domain(pool):
    # Must not raise even though nothing exists to invalidate.
    cache.invalidate_domain_reputation(pool, "never-existed.example")
    assert cache.read_domain_reputation(pool, "never-existed.example") is None
