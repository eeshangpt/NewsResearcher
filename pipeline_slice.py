"""Task 3.7.1 real end-to-end slice: sourcing -> claim extraction -> claim
clustering -> sentiment -> persistence -> summarization, on genuinely fresh
real-world data (not the Task 3.6.1a fixture corpus).

GDELT is IP-blocked in this sandbox (known issue #101) and WHOIS (port 43)
is unreachable (silent firewall drop) -- both stubbed to their documented
soft-fail values, same effect as those signals failing in production.
Everything else (RSS, Google News backfill, dedup, reputation scoring,
embeddings, clustering, real LLM calls, real Postgres) is live/real.

Full article text is fetched via sourcing/fulltext.py and held in local
variables only -- never written to disk, matching the no-full-text-
persistence rule. Only claim-level text (already the pipeline's own
persisted granularity) is printed/logged.
"""
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from newsresearch.sourcing.fulltext import fetch_fulltext_for_cluster
from newsresearch.agents.claim_extraction_agent import extract_claims
from newsresearch.clustering.claim_clustering import cluster_claims
from newsresearch.agents.sentiment import score_claim_sentiment
from newsresearch.persistence.db import init_db
from newsresearch.persistence import claim_clusters as cc_mod
from newsresearch.agents.summarization_agent import summarize_cluster
from newsresearch.config import Settings


print("=== 1/2. real Gate-2-cleared-stand-in subtopic (see write-up for why) ===")
# GDELT is IP-blocked in this sandbox (issue #101, confirmed: 429 on every
# retry attempt including a 1-day-window minimal query). google_news_backfill
# URLs are Google's own redirect-shell pages, not the publisher's real URL --
# trafilatura downloads 200 OK but extracts zero body text from all of them
# (confirmed on a real sample), so a backfill-only article set can't feed
# fulltext fetch at all. Real, directly-resolvable article URLs only come
# from the 4 direct-outlet RSS feeds -- cross-outlet keyword search across
# their current live feeds found one real multi-source story: the Leipzig
# airport drone/explosives incident, independently covered by BBC, Guardian,
# and NPR. This stands in for a Gate-2-cleared cluster (smaller than a real
# one -- only 3 domains -- but every URL is real and fulltext-fetchable).
subtopic_articles = [
    {
        "domain": "bbc.com",
        "title": "Drone carrying explosives found at German airport, police say",
        "url": "https://www.bbc.co.uk/news/articles/cyvlg4q48l3o?at_medium=RSS&at_campaign=rss",
    },
    {
        "domain": "theguardian.com",
        "title": "Drone carrying explosives at German airport marks 'new level of danger', says interior minister",
        "url": "https://www.theguardian.com/world/2026/aug/05/drone-german-airport-dhl-cargo-plane-collides-object-leipzig",
    },
    {
        "domain": "theguardian.com",
        "title": "The Leipzig drone bomb marks a dangerous escalation for Europe",
        "url": "https://www.theguardian.com/world/2026/aug/05/the-leipzig-drone-bomb-marks-a-dangerous-escalation-for-europe",
    },
    {
        "domain": "npr.org",
        "title": "Drone with explosives found at German airport, official sees 'new quality' of threat",
        "url": "https://www.npr.org/2026/08/06/g-s1-137571/germany-drone-explosives-airport",
    },
]
for a in subtopic_articles:
    print(" -", a["domain"], "|", a["title"])

print("=== 3. full-text fetch (in-memory only) ===")
fulltexts = fetch_fulltext_for_cluster(subtopic_articles)
url_to_text = {ft["url"]: ft["fulltext"] for ft in fulltexts}
ok = sum(1 for v in url_to_text.values() if v)
print(f"{ok}/{len(subtopic_articles)} articles yielded full text")

print("=== 4. claim extraction (real LLM calls, traced) ===")
article_ids = {}
article_claims = {}
for i, a in enumerate(subtopic_articles):
    text = url_to_text.get(a["url"])
    article_id = f"leipzig-drone-{i}"
    article_ids[article_id] = a
    if not text:
        article_claims[article_id] = []
        continue
    try:
        result = extract_claims(text, run_id="task-3.7.1-spotcheck")
        article_claims[article_id] = result.claims
        print(f"{article_id} ({a['domain']}): {len(result.claims)} claims")
    except Exception as e:
        print(f"{article_id} ({a['domain']}): extraction failed: {e}")
        article_claims[article_id] = []

total_claims = sum(len(v) for v in article_claims.values())
print(f"total claims extracted: {total_claims}")

print("=== 5. claim clustering ===")
cluster_result = cluster_claims(article_claims, settings=Settings())
print(f"{len(cluster_result['clusters'])} clusters, {len(cluster_result['noise'])} noise claims")

print("=== 6. sentiment scoring ===")
sentiments = score_claim_sentiment(article_claims)

print("=== 7. persist to real local Postgres ===")
settings = Settings()
pool = init_db(settings.database_url)
subtopic_id = "task371-leipzig-drone"
with pool.connection() as conn:
    conn.execute(
        "INSERT INTO subtopics (subtopic_id, label) VALUES (%s, %s) "
        "ON CONFLICT (subtopic_id) DO NOTHING",
        (subtopic_id, "German airport drone/explosives incident"),
    )
    for article_id, a in article_ids.items():
        conn.execute(
            "INSERT INTO articles (article_id, subtopic_id, url, domain, title) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (article_id) DO NOTHING",
            (article_id, subtopic_id, a["url"], a["domain"], a["title"]),
        )

cluster_ids = cc_mod.write_claim_clusters(pool, subtopic_id, cluster_result, claim_sentiments=sentiments)
print(f"persisted {len(cluster_ids)} clusters: {list(cluster_ids.values())}")

print("=== 8. summarize several real persisted clusters ===")
# summarize the biggest clusters first (most assert/omit signal to review)
sorted_labels = sorted(
    cluster_result["clusters"].keys(),
    key=lambda k: len(cluster_result["clusters"][k]["asserting_article_ids"]),
    reverse=True,
)
for label in sorted_labels[:6]:
    pid = cluster_ids[label]
    entry = cluster_result["clusters"][label]
    print(f"\n--- cluster {pid} ---")
    print(f"asserting articles ({len(entry['asserting_article_ids'])}): {entry['asserting_article_ids']}")
    print(f"omitting articles ({len(entry['omitting_article_ids'])})")
    for aid, txt in zip(entry["article_ids"], entry["claim_text"]):
        print(f"  [{aid}] {txt}")
    summary = summarize_cluster(pool, pid, run_id="task-3.7.1-spotcheck")
    print(f"SUMMARY: {summary}")

print("\nDONE")