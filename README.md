# NewsResearch

Multi-Agent News Research & Bias-Aware Briefing System.

See `PRD.md`, `TRD.md`, `news_research_arch.md`, and `EXECUTION_PLAN.md` for product/technical context.

## What works today (Phases 0-3 of 6)

Give it a topic and it will, end to end, via one CLI command:

1. Source articles for the topic (DuckDuckGo + Tavily web search, plus trusted-outlet RSS
   and a Google News backfill if coverage is thin) and score each source's reputation.
2. Propose subtopic candidates from that coverage and pause for human approval
   (**Gate 1** — approve as-is or drop candidates).
3. For each approved subtopic: source + cluster its own coverage, extract claims, cluster
   claims across articles, tag sentiment, and summarize each cluster — then pause again
   (**Gate 2** — review the per-subtopic cluster report and continue).
4. Persist claim clusters and summaries to Postgres.

**Not built yet (Phase 4-6): bias/framing labeling, the Briefing Agent, the final
per-subtopic snapshot JSON artifact, trend/drift tracking across runs, and the Streamlit
UI.** The CLI below is the only way to drive the pipeline right now, and a run currently
ends after Phase 3's summarization step, not a finished bias-aware briefing.

## Local dev prerequisite: bring up infra with Docker Compose

Before running anything else locally, start the app database and the self-hosted
Langfuse tracing stack:

```bash
docker compose up -d
```

This brings up:

- `postgres` — the app's own Postgres database (`newsresearch`), used for pipeline
  persistence (subtopics, sourcing/reputation cache, run costs, checkpoints, claim
  clusters, etc.).
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

### `.env` setup

Copy `.env.example` to `.env` and fill in:

- `NEWSRESEARCH_DATABASE_URL` — the app Postgres connection string; the `.env.example`
  default already matches `docker compose up -d`'s `postgres` service, usually no
  change needed.
- `OPENAI_API_KEY` — required for any real pipeline run (subtopic proposal, claim
  extraction, summarization all call an OpenAI chat model via `llm/models.py`).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — a disposable API key pair generated
  from your own local Langfuse instance (log into `http://localhost:3000`, create a
  project, generate a key pair). Without a real key here, LLM-call traces silently
  don't show up in Langfuse — see the "Langfuse" section below.
- `LANGFUSE_HOST` — defaults to `http://localhost:3000`, matches the Docker Compose
  stack above.
- `MLFLOW_TRACKING_URI` — defaults to `./mlruns` (local file-store), no setup needed.
- `TAVILY_API_KEY` — optional. Sourcing's discovery step runs DuckDuckGo search
  unconditionally and adds a Tavily search leg on top of it if this key is set (free
  tier, no card required to sign up). If unset, sourcing just logs it and continues on
  DuckDuckGo (+ RSS, + Google News backfill if coverage is thin) — it never fails the
  run.

## Running the pipeline

With `docker compose up -d` running and `.env` populated:

```bash
uv run newsresearch run "<topic>"
```

This drives the compiled LangGraph pipeline for `<topic>` through sourcing, subtopic
proposal, Gate 1, per-subtopic sourcing/clustering, claim extraction/clustering/
summarization, and Gate 2, with the observability stack (cost logging + Langfuse +
MLflow) attached to the single top-level `graph.invoke()` call.

What you'll see on screen:

- A `run_id=run-<uuid>` printed immediately — this is also the LangGraph checkpoint
  `thread_id`, the Postgres `run_costs.run_id`, the Langfuse trace tag, and the MLflow
  run tag, so it's the one identifier to hang onto if you want to cross-reference a run
  across all three places later.
- **Gate 1**: the proposed subtopic candidates (label + article count), plus any excess
  candidates detected beyond the configured max. Prompted `Approve as-is or edit? [a/e]`
  — `a` (default) proceeds with everything shown; `e` lets you type a comma-separated
  list of candidate indices to keep, dropping the rest.
- **Gate 2**, once per approved subtopic: a cluster report (`cluster_sizes`,
  `source_spread`, sample headlines) for that subtopic's claim clusters. Prompted to
  press enter to continue — this gate is not currently editable from the CLI, just a
  review checkpoint.
- A final `run_id=... topic='...' completed.` line once every gate has been resumed.

### Resuming a killed run

If the process is killed (e.g. Ctrl-C) while parked at a gate prompt, the run's state
is durably checkpointed in Postgres (LangGraph `PostgresSaver`) — nothing is lost.
Resume it with the `run_id` printed at the start of the original run:

```bash
uv run newsresearch run "<topic>" --thread-id=run-<uuid>
```

`<topic>` is still required as a CLI argument but is ignored on resume — the original
run's persisted topic is used. An unknown or already-completed `thread_id` fails with a
clear error rather than silently starting a new run.

### Manual sourcing spot-check

To exercise the Sourcing Agent alone (web search + RSS + reputation scoring), without
running the full graph or hitting any gate:

```bash
uv run newsresearch dev sourcing-test "<keywords>" [--lookback-days N]
```

`<keywords>` is a whitespace-separated string (e.g. `"climate policy"` →
`["climate", "policy"]`); `--lookback-days` defaults to 7. Prints each surviving
article's URL, domain, reputation score, and tier.

## Checking observability

### Langfuse (per-LLM-call tracing)

Open [http://localhost:3000](http://localhost:3000), open the project matching your
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, and filter/search traces by the `run_id`
printed by the CLI. Each trace shows one span per LLM call made during that run
(subtopic proposal, claim extraction, summarization, etc.) with prompt/response,
latency, and token/cost detail.

**Known gotcha:** if `LANGFUSE_SECRET_KEY`/`LANGFUSE_PUBLIC_KEY` are blank or invalid,
the pipeline still runs and exits `0` — Langfuse export fails soft by design (NFR-3) —
but no trace ever appears for that run. If you expect a trace and don't see one, the
first thing to check is whether your Langfuse key pair is real: generate a fresh one
from the Langfuse UI itself (a manual, one-time setup step; nothing else can do this
for you).

### MLflow (per-pipeline-run tracking)

MLflow tracks something different from Langfuse: one run per pipeline invocation
(config snapshot as params — model choices, `max_subtopics`, etc. — plus artifacts),
not per-LLM-call traces. Data is written locally under `./mlruns` (no server needed to
generate it). To browse it:

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Run this from the repo root (where `./mlruns` lives) and open the URL it prints
(defaults to `http://127.0.0.1:5000`). Each run is tagged with the same `run_id` the
CLI prints, so you can find the MLflow run matching a specific Langfuse trace or
`run_costs` Postgres row.

### Cost/token logging (Postgres)

Every LLM call also writes a row to the app Postgres `run_costs` table (`run_id`,
`stage`, `model`, `input_tokens`, `output_tokens`, `estimated_cost`, `latency_ms`),
independent of whether Langfuse is reachable. Query it directly against
`NEWSRESEARCH_DATABASE_URL` if you want per-run cost totals without opening either UI.

## Tests

```bash
uv run pytest                              # full suite, excludes live-API tests by default
uv run pytest -m live tests/live/... -v    # opt-in real-API smoke tests, run deliberately
```
