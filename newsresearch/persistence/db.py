"""Postgres connection pool + idempotent schema application.

`init_db` is the single entry point every later module should use to obtain
a `ConnectionPool` against the app's Postgres database and guarantee the
schema (TRD section 5 tables + `run_costs` + `schema_version`) exists.
Applying `schema.sql` twice is a no-op — every statement in it is either
`CREATE TABLE IF NOT EXISTS` or an idempotent guarded `INSERT`.
"""

from pathlib import Path

from psycopg_pool import ConnectionPool

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_db(database_url: str) -> ConnectionPool:
    """Open a connection pool against `database_url` and apply `schema.sql`.

    Safe to call repeatedly against the same database: schema.sql is
    idempotent, so a second call neither raises nor duplicates/changes state.
    """
    pool = ConnectionPool(conninfo=database_url, open=True)
    schema_sql = SCHEMA_PATH.read_text()
    with pool.connection() as conn:
        conn.execute(schema_sql)
    return pool
