"""`claim_clusters` + `claim_cluster_articles` write path (Story 3.5, Task 3.5.1).

Follows `reputation/cache.py`'s connection-handling convention: every
function here takes an already-open `ConnectionPool` (from `init_db`) and
opens/closes its own short-lived connection per call.

Persists `clustering/claim_clustering.py::cluster_claims()`'s in-memory
output verbatim: one `claim_clusters` row per cluster, one
`claim_cluster_articles` row per asserting (article, claim) pair plus one per
omitting article. Sentiment (`agents/sentiment.py::ClaimSentiment`) is
optional and, if given, is matched onto each 'asserts' row by
(article_id, claim_text) -- it is per-claim, never a per-cluster average.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from psycopg_pool import ConnectionPool

from newsresearch.agents.sentiment import ClaimSentiment


def write_claim_clusters(
    pool: ConnectionPool,
    subtopic_id: str,
    cluster_result: dict[str, Any],
    claim_sentiments: dict[str, list[ClaimSentiment]] | None = None,
) -> dict[int, str]:
    """Persist one subtopic's `cluster_claims()` output.

    Returns `{in_memory_cluster_label: persisted_cluster_id}` so callers
    (e.g. Task 3.6.1's summarization agent) can address a cluster by its
    persisted id right after writing it.
    """
    sentiment_lookup: dict[tuple[str, str], deque[ClaimSentiment]] = defaultdict(deque)
    for article_id, scored in (claim_sentiments or {}).items():
        for cs in scored:
            sentiment_lookup[(article_id, cs.claim.claim_text)].append(cs)

    cluster_ids: dict[int, str] = {}
    with pool.connection() as conn:
        for label, entry in cluster_result.get("clusters", {}).items():
            cluster_id = f"{subtopic_id}:{label}"
            cluster_ids[label] = cluster_id

            conn.execute(
                """
                INSERT INTO claim_clusters (cluster_id, subtopic_id)
                VALUES (%s, %s)
                ON CONFLICT (cluster_id) DO UPDATE SET subtopic_id = EXCLUDED.subtopic_id
                """,
                (cluster_id, subtopic_id),
            )

            for article_id, claim_text in zip(entry["article_ids"], entry["claim_text"]):
                cs = sentiment_lookup.get((article_id, claim_text))
                sentiment_label = sentiment_compound = None
                if cs:
                    matched = cs.popleft()
                    sentiment_label = matched.sentiment_label
                    sentiment_compound = matched.sentiment_compound
                conn.execute(
                    """
                    INSERT INTO claim_cluster_articles
                        (cluster_id, article_id, relation, claim_text,
                         sentiment_label, sentiment_compound)
                    VALUES (%s, %s, 'asserts', %s, %s, %s)
                    """,
                    (cluster_id, article_id, claim_text, sentiment_label, sentiment_compound),
                )

            for article_id in entry["omitting_article_ids"]:
                conn.execute(
                    """
                    INSERT INTO claim_cluster_articles
                        (cluster_id, article_id, relation, claim_text,
                         sentiment_label, sentiment_compound)
                    VALUES (%s, %s, 'omits', NULL, NULL, NULL)
                    """,
                    (cluster_id, article_id),
                )

    return cluster_ids


def write_cluster_summary(pool: ConnectionPool, cluster_id: str, summary: str) -> None:
    """Write `agents/summarization_agent.py`'s output to `claim_clusters.summary`."""
    with pool.connection() as conn:
        conn.execute(
            "UPDATE claim_clusters SET summary = %s WHERE cluster_id = %s",
            (summary, cluster_id),
        )


def read_subtopic_cluster_ids(pool: ConnectionPool, subtopic_id: str) -> list[str]:
    """Every persisted cluster_id for one subtopic, in stable id order.

    Needed by `agents/bias_framing_agent.py`, which batches a subtopic's
    clusters rather than addressing them one at a time.
    """
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT cluster_id FROM claim_clusters WHERE subtopic_id = %s ORDER BY cluster_id",
            (subtopic_id,),
        ).fetchall()
    return [r[0] for r in rows]


def read_cluster_article_relations(pool: ConnectionPool, cluster_id: str) -> list[dict[str, Any]]:
    """Every `claim_cluster_articles` row for one persisted cluster.

    Returns `[{"article_id", "relation", "claim_text", "sentiment_label",
    "sentiment_compound", "domain"}, ...]`, one dict per row (asserting
    articles may have more than one row). `domain` is joined in from
    `articles` -- needed by `agents/summarization_agent.py`'s prompt, which
    lists claims as "source domain, then the claim".
    """
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT cca.article_id, cca.relation, cca.claim_text,
                   cca.sentiment_label, cca.sentiment_compound, a.domain
            FROM claim_cluster_articles cca
            JOIN articles a ON a.article_id = cca.article_id
            WHERE cca.cluster_id = %s
            """,
            (cluster_id,),
        ).fetchall()
    return [
        {
            "article_id": r[0],
            "relation": r[1],
            "claim_text": r[2],
            "sentiment_label": r[3],
            "sentiment_compound": r[4],
            "domain": r[5],
        }
        for r in rows
    ]
