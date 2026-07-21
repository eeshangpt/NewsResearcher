-- Postgres schema for NewsResearch app persistence.
--
-- Transcribed from TRD.md section 5 (originally SQLite-flavored) per
-- EXECUTION_PLAN.md's explicit guidance: BYTEA instead of BLOB, TIMESTAMPTZ
-- instead of TIMESTAMP. Adds `run_costs` (Cross-Cutting Concerns, NFR-1) and
-- `schema_version` (single-row, for lightweight future migrations), neither
-- of which are in TRD's literal table list.
--
-- Every CREATE TABLE is IF NOT EXISTS so re-applying this file is a no-op.

-- One row per topic, canonicalized + hashed for stable scheduling identity
CREATE TABLE IF NOT EXISTS topics (
    topic_hash TEXT PRIMARY KEY,
    canonical_topic TEXT NOT NULL,
    raw_topic_original TEXT NOT NULL,
    max_subtopics INTEGER DEFAULT 5,
    created_at TIMESTAMPTZ
);

-- One row per run (a run = one execution of the pipeline for a topic at a point in time)
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    topic_hash TEXT REFERENCES topics(topic_hash),
    run_type TEXT CHECK(run_type IN ('manual','scheduled')),
    schedule_cadence TEXT, -- 'daily' | 'weekly' | 'monthly' | NULL
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

-- One row per subtopic within a run
CREATE TABLE IF NOT EXISTS subtopics (
    subtopic_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    label TEXT,
    embedding BYTEA, -- for cross-run matching
    article_count INTEGER,
    distinctiveness_score REAL,
    included_in_cap BOOLEAN
);

-- One row per source domain (reputation cache, independent of any single run)
CREATE TABLE IF NOT EXISTS domain_reputation (
    domain TEXT PRIMARY KEY,
    tier TEXT, -- 'trusted' | 'unknown' | etc.
    base_score REAL,
    heuristic_adjustment REAL,
    final_score REAL,
    computed_at TIMESTAMPTZ
);

-- One row per article (metadata + summary only; NEVER full body text)
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    subtopic_id TEXT REFERENCES subtopics(subtopic_id),
    url TEXT,
    domain TEXT REFERENCES domain_reputation(domain),
    title TEXT,
    published_at TIMESTAMPTZ,
    reputation_score_at_fetch REAL
);

-- One row per claim cluster within a subtopic
CREATE TABLE IF NOT EXISTS claim_clusters (
    cluster_id TEXT PRIMARY KEY,
    subtopic_id TEXT REFERENCES subtopics(subtopic_id),
    summary TEXT,
    framing_label TEXT,
    sentiment_avg REAL
);

-- Many-to-many: which articles assert / omit which claim clusters
CREATE TABLE IF NOT EXISTS claim_cluster_articles (
    cluster_id TEXT REFERENCES claim_clusters(cluster_id),
    article_id TEXT REFERENCES articles(article_id),
    relation TEXT CHECK(relation IN ('asserts','omits')),
    claim_text TEXT
);

-- One row per subtopic per run: the synthesized briefing
CREATE TABLE IF NOT EXISTS briefings (
    subtopic_id TEXT PRIMARY KEY REFERENCES subtopics(subtopic_id),
    consensus_summary TEXT,
    disputed_summary TEXT,
    notable_omissions TEXT
);

-- Cross-run subtopic matches, for timeline/drift tracking
CREATE TABLE IF NOT EXISTS subtopic_matches (
    current_subtopic_id TEXT REFERENCES subtopics(subtopic_id),
    prior_subtopic_id TEXT REFERENCES subtopics(subtopic_id),
    similarity_score REAL
);

-- Per-stage LLM cost/token logging (Cross-Cutting Concerns, NFR-1), written
-- by observability/cost_callback.py on every top-level graph.invoke() call.
CREATE TABLE IF NOT EXISTS run_costs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost REAL,
    created_at TIMESTAMPTZ
);

-- Single-row table tracking the applied schema version, for lightweight
-- future migrations (bump manually when DDL changes; no Alembic at this scale).
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

INSERT INTO schema_version (version)
SELECT 1
WHERE NOT EXISTS (SELECT 1 FROM schema_version);
