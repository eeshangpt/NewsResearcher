import pytest
from testcontainers.postgres import PostgresContainer

from newsresearch.persistence.db import init_db

EXPECTED_TABLES = {
    "topics",
    "runs",
    "subtopics",
    "domain_reputation",
    "articles",
    "claim_clusters",
    "claim_cluster_articles",
    "briefings",
    "subtopic_matches",
    "run_costs",
    "schema_version",
}


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres.get_connection_url().replace("postgresql+psycopg2", "postgresql")


def _existing_tables(pool):
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {row[0] for row in rows}


def test_init_db_creates_every_table_and_schema_version_row(postgres_url):
    pool = init_db(postgres_url)
    try:
        assert EXPECTED_TABLES <= _existing_tables(pool)

        with pool.connection() as conn:
            rows = conn.execute("SELECT version FROM schema_version").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 2
    finally:
        pool.close()


def test_init_db_is_idempotent_on_second_call(postgres_url):
    pool = init_db(postgres_url)
    try:
        tables_before = _existing_tables(pool)
        with pool.connection() as conn:
            version_rows_before = conn.execute("SELECT version FROM schema_version").fetchall()
    finally:
        pool.close()

    # Second call against the same database must not raise, and must not
    # duplicate/change table set or the schema_version row.
    pool2 = init_db(postgres_url)
    try:
        assert _existing_tables(pool2) == tables_before

        with pool2.connection() as conn:
            version_rows_after = conn.execute("SELECT version FROM schema_version").fetchall()
        assert version_rows_after == version_rows_before
    finally:
        pool2.close()
