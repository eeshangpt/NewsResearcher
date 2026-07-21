# NewsResearch

Multi-Agent News Research & Bias-Aware Briefing System.

See `PRD.md`, `TRD.md`, `news_research_arch.md`, and `EXECUTION_PLAN.md` for product/technical context.

## Local dev prerequisite: bring up infra with Docker Compose

Before running anything else locally, start the app database and the self-hosted
Langfuse tracing stack:

```bash
docker compose up -d
```

This brings up:

- `postgres` — the app's own Postgres database (`newsresearch`), used for pipeline
  persistence (subtopics, sourcing/reputation cache, run costs, checkpoints, etc.).
- The full self-hosted Langfuse stack, defined in `deploy/langfuse/docker-compose.yml`
  and wired into the root `docker-compose.yml`: `langfuse-postgres`, `clickhouse`,
  `redis`, `minio`, `langfuse-web`, `langfuse-worker`. This is a separate, isolated
  Postgres instance/volume from the app's own — Langfuse's internal migrations never
  touch application schema.

Once everything is healthy, the Langfuse UI is reachable at
[http://localhost:3000](http://localhost:3000).

Verify a clean bring-up:

```bash
docker compose ps
```

Every service should show a `healthy` status (this can take up to ~30-60 seconds on
first start while `langfuse-web`/`langfuse-worker` run their internal migrations).

To bring up only the app database (e.g. for a quick `psql` session or a task that
doesn't need tracing):

```bash
docker compose up -d postgres
```

To tear everything down (add `-v` to also drop volumes/data):

```bash
docker compose down
```

All Langfuse services and volumes are named/prefixed distinctly from the app's own
`postgres` service (`langfuse-*` vs. `postgres`/`newsresearch_postgres_data`), so
`docker compose config` will always show them as separate services and volumes.

All secrets used by the Langfuse stack in `deploy/langfuse/docker-compose.yml` have
hardcoded local-dev-only defaults so a fresh clone works out of the box with no
`.env` setup required. Override any of them via a root-level `.env` (or exported
shell variables) if you need different values.

Application-level configuration (`.env` for `OPENAI_API_KEY`,
`NEWSRESEARCH_DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, etc.) is
documented separately in `.env.example` once the config system lands.
