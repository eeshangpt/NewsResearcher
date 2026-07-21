# Execution Plan
## Multi-Agent News Research & Bias-Aware Briefing System

**Companion documents:** PRD.md, TRD.md, news_research_arch.md
**Repo state at time of writing:** blank scaffold — stub `main.py`, empty `pyproject.toml` (no deps), empty `README.md`, `uv`-managed `.venv`. No source code, tests, or config exist yet.
**Package manager:** `uv` (add deps via `uv add`, dev deps via `uv add --group dev`).
**Embeddings default:** local `sentence-transformers` (via `langchain-huggingface`), chosen over OpenAI embeddings API for cost — configurable/swappable.
**Local infrastructure:** Docker Compose is now required for local dev (Postgres for app persistence + a self-hosted Langfuse stack). This is a deliberate departure from the TRD's original "SQLite: zero-ops" rationale, traded for local LLM-call tracing/debugging — see Cross-Cutting Concerns below.

---

## Cross-Cutting Concerns (established in Phase 0, threaded through every phase)

These aren't phases themselves — they're foundations every later phase depends on, so they must exist before Phase 1 starts and stay consistent afterward.

- **Config system (NFR-5)**: one `Settings` object (`pydantic-settings` `BaseSettings`) loaded from `.env` (secrets/connection strings) + optional `config.yaml` (tunables), env vars always override file. Structure as nested settings, not a flat bag: `Settings.pipeline.max_subtopics`, `Settings.reputation.staleness_days` / `min_score_threshold`, `Settings.clustering.similarity_threshold` / `subtopic_match_threshold`, `Settings.models.{subtopic,claim_extraction,summarization,bias_framing,briefing}`, `Settings.embeddings.backend` (`"local"` default | `"openai"`). Every agent/module reads from this object — never hardcodes a constant that's listed as configurable.

- **LLM layer — full LangChain idioms (NFR-6)**: `llm/models.py` holds factory functions returning LangChain objects — `get_chat_model(stage) -> BaseChatModel` (`ChatOpenAI` via `langchain-openai`, model name from `Settings.models.<stage>`) and `get_embeddings() -> Embeddings` (`HuggingFaceEmbeddings` for the local default, `OpenAIEmbeddings` as the swappable alternate — both behind the same `langchain_core.embeddings.Embeddings` interface). `llm/prompts/*.txt` stay plain template files (vendor-agnostic content), loaded into `ChatPromptTemplate.from_template()` at call time. `llm/schemas.py` holds pydantic output schemas; call sites use `model.with_structured_output(Schema)` instead of hand-rolled JSON parsing. LangGraph nodes call these LangChain runnables directly — LangGraph orchestrates control flow (gates, fan-out, sequencing), LangChain handles individual model/prompt construction; LangChain does not replace LangGraph as the orchestrator. Porting vendors later means changing `llm/models.py`'s factories, nothing else.

- **Cost/token logging (NFR-1)**: a custom `observability/cost_callback.py` (`BaseCallbackHandler` subclass) is attached at every top-level `graph.invoke(state, config={"callbacks": [...]})` call — LangChain callbacks propagate automatically through every nested call in every node, so no per-agent instrumentation code is needed. It captures token usage and writes `{run_id, stage, model, input_tokens, output_tokens, estimated_cost, latency_ms}` to a Postgres `run_costs` table, independent of whether Langfuse is reachable (fails soft — same graceful-degradation pattern as the Google News RSS backfill, NFR-3). `persistence/queries.py::cost_summary(run_id)` answers the PRD's "cost per run" success metric directly from the app's own database, without depending on Langfuse's API.

- **Observability / agent tracing & debugging — MLflow + Langfuse, distinct roles (no overlap):**
  - **Langfuse** (self-hosted) is the LLM-call-level tracing/debugging tool — a `langfuse` `CallbackHandler` attached alongside the cost callback at the same top-level graph invocation, so every agent's prompts/responses/latency are traced automatically. Traces are tagged with `run_id` + `subtopic_id` so a Langfuse trace maps back to a specific pipeline run. Self-hosted via Docker Compose (see Phase 0) rather than Langfuse Cloud, per your preference — accessible locally at `http://localhost:3000`.
  - **MLflow** is pipeline/experiment-level tracking, not raw call tracing (avoids duplicating Langfuse's job). Local file-store backend (`./mlruns`), zero extra infra. One MLflow run per pipeline run, keyed by `run_id`: logs config snapshot as params (model choices, `max_subtopics`, reputation weights, clustering thresholds) and the final snapshot JSON as an artifact — useful for comparing config changes across runs, and the natural home for the PRD's flagged v2 eval-harness work (`mlflow.evaluate()`) once a golden dataset exists.

- **Testing approach**: given GDELT/RSS/OpenAI/Postgres/Langfuse are external, non-deterministic, and/or stateful —
  - **Unit-testable now**: reputation scoring formula, dedup logic (URL normalize + title similarity), config loading/precedence, DB schema/queries, clustering functions given fixed embedding vectors, topic canonicalization/hashing, subtopic cross-run matching math, cost-logging arithmetic.
  - **Needs live calls / manual inspection**: GDELT/RSS response shape drift, WHOIS reliability, LLM prompt quality (subtopic proposals, claims, framing labels, briefing text), end-to-end gate flow, Langfuse trace correctness.
  - Tooling: `pytest`, `respx` (mocks `httpx` for GDELT/RSS/WHOIS-HTTP — record real responses once into `tests/fixtures/*.json`, replay offline), `freezegun` (staleness-window tests need controllable "now"), `pytest-mock`, `testcontainers[postgres]` (spins up an ephemeral Postgres container per test session — hermetic, doesn't depend on the dev Docker Compose stack being up, works in CI later). Keep `tests/live/` marked `@pytest.mark.live` for real-API smoke tests — opt-in only, never CI-blocking. `ruff` for lint/format.

- **Dev-time pipeline harness**: Streamlit (Phase 6) is deliberately last, but Phases 2–5 all involve blocking human gates — without a UI there's no way to approve/edit Gate 1 or view the Gate 2 report while building them. Resolve by repurposing `main.py` into a `typer` CLI from Phase 0 onward: `uv run newsresearch run "<topic>"` drives the graph, prints Gate 1 candidates to stdout, accepts approve/edit via stdin, prints the Gate 2 report per subtopic, prompts to continue. Throwaway-quality UX (replaced by Streamlit in Phase 6) but load-bearing for manual validation in every phase between.

---

## Phase 0 — Project Scaffolding

**Goal:** everything later phases assume exists actually exists.

**Directory layout:**
```
docker-compose.yml            # postgres (app db) + Langfuse self-host stack — all in this repo
deploy/
  langfuse/                   # Langfuse self-host compose/config, committed in this repo, not fetched externally
newsresearch/
  __init__.py
  cli.py                     # typer CLI (dev harness + eventual real entrypoint)
  config.py                  # Settings (pydantic BaseSettings)
  llm/
    models.py                 # LangChain chat-model + embeddings factories
    schemas.py                # pydantic models for structured LLM outputs
    prompts/                  # template files per agent
  observability/
    cost_callback.py           # LangChain callback handler -> run_costs (Postgres)
    langfuse_setup.py          # Langfuse CallbackHandler factory, run_id/subtopic_id tagging
    mlflow_setup.py            # MLflow run lifecycle helpers, keyed by run_id
  persistence/
    db.py                    # psycopg connection/pool, init_db(), schema_version check
    schema.sql                # Postgres DDL: TRD section 5 tables + run_costs + schema_version
    queries.py                 # typed read/write helpers, no raw SQL scattered in agents
  graph/
    state.py                  # LangGraph shared state schema (TypedDict/pydantic)
    build.py                  # graph assembly, empty passthrough nodes to start
    nodes/                    # one module per pipeline node
  agents/                     # one module per agent, added phase by phase
  sourcing/
  reputation/
  clustering/
  scheduling/
  data/
    trusted_outlets.yaml       # seed whitelist tier data (Phase 1 needs this)
tests/
  fixtures/
  live/
main.py                        # thin: `from newsresearch.cli import app; app()`
```

**Tasks:**
1. `uv add langgraph langgraph-checkpoint-postgres langchain langchain-core langchain-openai langchain-huggingface pydantic pydantic-settings python-dotenv typer psycopg[binary] psycopg_pool mlflow langfuse` — core deps.
2. `uv add --group dev pytest respx freezegun pytest-mock testcontainers[postgres] ruff` — dev deps.
3. `docker-compose.yml`: a `postgres` service for the app's own database (`newsresearch`), plus Langfuse's self-host services defined in `deploy/langfuse/` (Postgres, ClickHouse, Redis, MinIO, `langfuse-web`, `langfuse-worker`) — all committed directly in this repo, nothing referenced or pulled from outside it. Kept as Langfuse's own isolated stack, not sharing the app's database, so its internal migrations never touch app schema. Document `docker compose up -d` as the required pre-step for local dev.
4. `config.py`: define `Settings` with every NFR-5 tunable stubbed in, even before consumers exist.
5. `persistence/schema.sql`: transcribe TRD section 5 as Postgres DDL (`BYTEA` for `subtopics.embedding`, `TIMESTAMPTZ` for all timestamps) + `run_costs` table + a `schema_version(version INTEGER)` single-row table for lightweight future migrations (`CREATE TABLE IF NOT EXISTS`, bump `schema_version` manually when DDL changes — no Alembic needed at this scale).
6. `persistence/db.py`: `init_db(database_url) -> Connection`/pool via `psycopg_pool`, idempotent schema application. (No WAL-mode workaround needed — Postgres handles concurrent per-subtopic writes during fan-out natively.)
7. `graph/state.py`: top-level state (topic, canonical_topic, run_id, subtopics: list, approved: bool) + per-subtopic sub-state (used by `Send`-based fan-out in Phase 2).
8. `graph/build.py`: wire the full node topology from the TRD diagram as no-op passthrough nodes (Subtopic → Gate1 → fan-out → Sourcing → Clustering → Gate2 → Claims → Summarize → Bias → Briefing → Snapshot → Timeline), compiled with a `PostgresSaver` checkpointer — **not** an in-memory saver, since gates must survive process restarts (this matters again in Phase 6, where Streamlit reruns the script per interaction).
9. `llm/models.py`: chat-model and embeddings factories per the LLM layer design above.
10. `observability/cost_callback.py`, `langfuse_setup.py`, `mlflow_setup.py`: implement per the Cross-Cutting Concerns design; wire both callbacks into `cli.py`'s top-level `graph.invoke(...)` call.
11. `cli.py`: `typer` app, `run <topic>` command invoking the compiled graph end-to-end through the no-op nodes, with callbacks attached.
12. `.env.example`: `OPENAI_API_KEY`, `NEWSRESEARCH_DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (default `http://localhost:3000`), `MLFLOW_TRACKING_URI` (default `./mlruns`).

**Done when:**
- `docker compose up -d` brings up Postgres (app db) + the full local Langfuse stack.
- `uv run newsresearch run "test topic"` executes the full no-op graph start to finish and exits cleanly.
- `init_db()` applies the schema against Postgres, producing every TRD table + `run_costs` + `schema_version`.
- A stub LLM call through the graph produces a `run_costs` row in Postgres, a visible trace in the local Langfuse UI (`localhost:3000`), and an MLflow run under `./mlruns` — the full observability stack verified end-to-end before any real agent exists.
- `Settings()` loads without error from `.env.example`-shaped input.
- `pytest` (Postgres-backed tests via `testcontainers`) runs and `ruff check` passes.

### Phase 0 — Story/Task Breakdown (PM decomposition)

Repo-state check confirmed: no `newsresearch/` package, no `docker-compose.yml`, no `deploy/`, no `tests/` — nothing below has been built yet. The numbered Tasks above are preserved as-is except where a task bundled more than one independently-verifiable outcome; those are split below and cross-referenced back to their original task number for traceability. Splits: original Task 3 (docker-compose.yml) → app-Postgres service and Langfuse stack are separately verifiable; original Task 9 (`llm/models.py`) → `get_chat_model` and `get_embeddings` are separately verifiable; original Task 10 (observability) → three distinct modules plus a separate wiring step.

**Ownership split (per `devops-engineer`'s own scope boundary):** Story 0.2 (`docker-compose.yml` + `deploy/langfuse/`) is `devops-engineer`-owned — infra/containers only, no application code. Every other story (0.1, 0.3–0.9) is `backend-engineer`-owned — this includes the observability *code* (`cost_callback.py`, `langfuse_setup.py`, `mlflow_setup.py`) even though it integrates with devops-provisioned infra; devops owns the containers those modules talk to, not the modules themselves. Several backend tasks have a **runtime** (not build-time) dependency on Story 0.2 actually being up (`docker compose up -d`) rather than merely written — called out explicitly at each task below, since `testcontainers`-backed unit/integration tests are deliberately hermetic and do NOT carry this dependency. CI (running `ruff`/`pytest` in a pipeline) is not itself a Phase 0 task in the list below — `devops-engineer`'s own scope doc flags CI as unscoped in this plan and asks `project-manager` to scope it explicitly rather than have it freelanced; this is an open item, not a silent gap, and is called out again at the end of this breakdown.

**Story 0.1 — Dependency baseline established**
Acceptance: `uv sync` installs core + dev groups with no resolution errors.
- [ ] Task 0.1.1 (orig. Task 1): `uv add langgraph langgraph-checkpoint-postgres langchain langchain-core langchain-openai langchain-huggingface pydantic pydantic-settings python-dotenv typer psycopg[binary] psycopg_pool mlflow langfuse`.
      Acceptance: `pyproject.toml`/`uv.lock` list every named package; `uv sync` exits 0; each imports cleanly in `.venv`.
      Depends on: none
- [ ] Task 0.1.2 (orig. Task 2): `uv add --group dev pytest respx freezegun pytest-mock testcontainers[postgres] ruff`.
      Acceptance: dev group lists all named packages; `uv run pytest --version` and `uv run ruff --version` succeed.
      Depends on: none

**Story 0.2 — Local infra (Postgres + self-hosted Langfuse) via Docker Compose**
Acceptance: `docker compose up -d` brings up app Postgres and the full Langfuse stack, UI reachable at `localhost:3000`.
- [ ] Task 0.2.1 (orig. Task 3, split a): `docker-compose.yml` `postgres` service for the app DB (`newsresearch`).
      Acceptance: `docker compose up -d postgres` starts the container; `pg_isready` against it succeeds.
      Depends on: none
- [ ] Task 0.2.2 (orig. Task 3, split b): `deploy/langfuse/` self-host stack (Postgres, ClickHouse, Redis, MinIO, `langfuse-web`, `langfuse-worker`), wired into root `docker-compose.yml`, isolated from the app's own Postgres service/volume.
      Acceptance: `docker compose up -d` brings up all Langfuse services; `http://localhost:3000` loads; `docker compose config` shows Langfuse's Postgres as a distinct service/volume from Task 0.2.1's.
      Depends on: none
- [ ] Task 0.2.3 (orig. Task 3, doc note): document `docker compose up -d` as the required local-dev pre-step.
      Acceptance: README states the pre-step; `docker compose ps` on a fresh clone shows all services healthy.
      Depends on: 0.2.1, 0.2.2

**Story 0.3 — Config system (`Settings`) loads every NFR-5 tunable**
Acceptance: `Settings()` instantiates from `.env.example`-shaped input and exposes every nested field named in Cross-Cutting Concerns.
- [ ] Task 0.3.1 (orig. Task 4): `config.py` — nested `Settings(BaseSettings)` with every NFR-5 tunable stubbed in.
      Acceptance: unit test instantiates `Settings` from an in-memory env + optional `config.yaml`, asserts every named nested field is present and env vars override yaml.
      Depends on: none
- [ ] Task 0.3.2 (orig. Task 12): `.env.example` — `OPENAI_API_KEY`, `NEWSRESEARCH_DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `MLFLOW_TRACKING_URI`.
      Acceptance: copying to `.env` and loading via `Settings()` succeeds with no missing-required-field error.
      Depends on: 0.3.1

**Story 0.4 — Postgres schema applies cleanly and idempotently**
Acceptance: `init_db()` run twice produces every TRD table + `run_costs` + `schema_version`, second run is a no-op.
- [ ] Task 0.4.1 (orig. Task 5): `persistence/schema.sql` — TRD section 5 DDL (`BYTEA` embeddings, `TIMESTAMPTZ` timestamps) + `run_costs` + `schema_version`, all `CREATE TABLE IF NOT EXISTS`.
      Acceptance: `psql -f persistence/schema.sql` against an empty DB creates every table with no errors; re-running is a no-op.
      Depends on: none
- [ ] Task 0.4.2 (orig. Task 6): `persistence/db.py::init_db(database_url)` via `psycopg_pool`, idempotent schema application.
      Acceptance: `testcontainers`-backed test calls `init_db()`, asserts all tables + `schema_version` row exist; second call doesn't raise.
      Depends on: 0.4.1

**Story 0.5 — LangGraph skeleton compiles and executes as a no-op pipeline with durable checkpointing**
Acceptance: full TRD node topology exists as passthrough nodes, compiled with `PostgresSaver`, `graph.invoke()` completes without error.
- [ ] Task 0.5.1 (orig. Task 7): `graph/state.py` — top-level state + per-subtopic sub-state for `Send`-based fan-out.
      Acceptance: state schema importable; instance constructs with all named fields.
      Depends on: none
- [ ] Task 0.5.2 (orig. Task 8): `graph/build.py` — wire full topology (Subtopic→Gate1→fan-out→Sourcing→Clustering→Gate2→Claims→Summarize→Bias→Briefing→Snapshot→Timeline) as no-op nodes, compiled with `PostgresSaver` (not in-memory).
      Acceptance: `graph.invoke(initial_state, config={"configurable": {"thread_id": "test"}})` runs through every node and returns; a checkpoint row exists in Postgres afterward.
      Depends on: 0.5.1, 0.4.2
      Runtime note (backend-engineer, cross-track): the Postgres row check in this acceptance criterion requires the real app Postgres from devops Story 0.2 Task 0.2.1 to be up via `docker compose up -d` — this is distinct from 0.4.2's `testcontainers` tests, which are hermetic and need nothing from Story 0.2.

**Story 0.6 — LLM layer factories exist and satisfy the LangChain interface contract**
Acceptance: `get_chat_model(stage)` returns `BaseChatModel` for every `Settings.models.*` stage; `get_embeddings()` returns an `Embeddings`-interface object for both backends.
- [ ] Task 0.6.1 (orig. Task 9, split a): `llm/models.py::get_chat_model(stage) -> BaseChatModel`, model name from `Settings.models.<stage>`, backed by `ChatOpenAI`.
      Acceptance: calling it per stage name returns a `BaseChatModel` constructed with the `Settings`-sourced model name (constructor-arg inspection, no network call in test).
      Depends on: 0.3.1
- [ ] Task 0.6.2 (orig. Task 9, split b): `llm/models.py::get_embeddings() -> Embeddings` — `HuggingFaceEmbeddings` for `"local"` (default), `OpenAIEmbeddings` for `"openai"`, both `isinstance(..., langchain_core.embeddings.Embeddings)`.
      Acceptance: backend switch via `Settings.embeddings.backend` returns the correct implementation in each case.
      Depends on: 0.3.1
- [ ] Task 0.6.3 (convention scaffold, not separately numbered above): `llm/prompts/` directory + `llm/schemas.py` stub, establishing the convention before Phase 1+ agents add real content.
      Acceptance: `llm/prompts/` exists with at least one example `.txt` template loadable via `ChatPromptTemplate.from_template()`; `llm/schemas.py` exists and imports cleanly.
      Depends on: none

**Story 0.7 — Observability stack verified end-to-end through a stub LLM call**
Acceptance: one stub LLM call through the graph produces a `run_costs` row, a visible Langfuse trace, and an MLflow run — simultaneously, from one top-level `graph.invoke()`.
- [ ] Task 0.7.1 (orig. Task 10, split a): `observability/cost_callback.py` — `BaseCallbackHandler` capturing `{run_id, stage, model, input_tokens, output_tokens, estimated_cost, latency_ms}`, writes to `run_costs`, fails soft independent of Langfuse reachability.
      Acceptance: unit test with a mocked LLM call asserts a fully-populated `run_costs` row is written; simulated DB-write failure logs-and-continues, never raises.
      Depends on: 0.4.2, 0.6.1
- [ ] Task 0.7.2 (orig. Task 10, split b): `observability/langfuse_setup.py` — `CallbackHandler` factory, tags traces with `run_id` + `subtopic_id`.
      Acceptance: a stub call attached with this handler produces a trace visible at `localhost:3000` tagged with the given `run_id`.
      Depends on: 0.2.2, 0.3.2
- [ ] Task 0.7.3 (orig. Task 10, split c): `observability/mlflow_setup.py` — run lifecycle helpers keyed by `run_id`, `./mlruns` file-store backend.
      Acceptance: start/end helpers around a stub invocation produce exactly one MLflow run under `./mlruns` tagged with `run_id`.
      Depends on: 0.3.2
- [ ] Task 0.7.4 (orig. Task 10, wiring clause + Task 11's callback half): wire all three callbacks into `cli.py`'s top-level `graph.invoke(state, config={"callbacks": [...]})`.
      Acceptance: `uv run newsresearch run "test topic"` (with a real/stubbed chat-model call in one node) produces all three artifacts from Story 0.7's acceptance in one command.
      Depends on: 0.7.1, 0.7.2, 0.7.3, 0.8.1
      Runtime note (backend-engineer, cross-track): checking this acceptance for real requires both devops Task 0.2.1 (app Postgres, for the `run_costs` row) and Task 0.2.2 (Langfuse stack, for the visible trace) up via `docker compose up -d` first — this is the single task where all of Story 0.2's infra must be live simultaneously.

**Story 0.8 — CLI dev harness runs the compiled no-op graph end-to-end**
Acceptance: `uv run newsresearch run "test topic"` executes the full no-op graph and exits 0.
- [ ] Task 0.8.1 (orig. Task 11, minus callback wiring which moved to 0.7.4): `cli.py` — `typer` app, `run <topic>` command invoking the compiled graph; `main.py` reduced to `from newsresearch.cli import app; app()`.
      Acceptance: `uv run newsresearch run "test topic"` invokes `graph.invoke()` on the Phase-0 no-op graph and exits 0.
      Depends on: 0.5.2
      Runtime note (backend-engineer, cross-track): since the compiled graph uses a `PostgresSaver` checkpointer, this command needs devops Task 0.2.1's app Postgres reachable at `NEWSRESEARCH_DATABASE_URL`, not merely built.

**Story 0.9 — Testing/lint baseline passes**
Acceptance: `pytest` (testcontainers-backed) and `ruff check` both pass against everything built in Phase 0.
- [ ] Task 0.9.1: baseline tests — `Settings` load/precedence, schema idempotency, cost-callback correctness, graph compile+invoke.
      Acceptance: `uv run pytest` exits 0, using `testcontainers[postgres]` (no dependency on the dev compose stack being up).
      Depends on: 0.3.1, 0.4.1, 0.4.2, 0.7.1, 0.5.2
- [ ] Task 0.9.2: `ruff` configuration + clean pass over `newsresearch/` as it stands.
      Acceptance: `uv run ruff check .` exits 0.
      Depends on: 0.1–0.8 code tasks

**Story 0.10 — Phase 0 Done-when verified end-to-end (integration/demo)**
Acceptance: all five bullets in the phase-level "Done when" block above hold simultaneously on a clean checkout.
- [ ] Task 0.10.1: manual end-to-end walkthrough of the exact Done-when command sequence.
      Acceptance: each bullet confirmed in order; any failure is filed against the specific task above, not against "Phase 0" generally.
      Depends on: 0.1–0.9 all complete
      Cross-track note: this is the one task that requires both tracks' outputs simultaneously and live — devops Story 0.2 (`docker compose up -d`, both app Postgres and Langfuse stack healthy) plus every backend-engineer story 0.1/0.3–0.9 already merged. Whoever runs this walkthrough needs both agents' work done first; it is not solely a backend-engineer or solely a devops-engineer task.

**Open item flagged for `project-manager` scoping (not a Phase 0 task, do not silently add it here):** `devops-engineer`'s own scope doc calls out that CI (a pipeline running `uv run ruff check .` and `uv run pytest` on PRs, `testcontainers`-backed, `@pytest.mark.live` excluded) is not yet scoped as its own story anywhere in this plan. Phase 0's "Done when" only requires these commands to pass locally, not that CI exists — so this isn't blocking Phase 0 completion, but it should be decided explicitly (as its own Phase 0 story, or deferred to a later phase) rather than left implicit.

---

## Phase 1 — Sourcing Agent + Reputation Scoring + Caching

**Depends on: Phase 0 complete.** Every task below assumes `Settings` (Task 0.3.1), `init_db`/`persistence/db.py` (Task 0.4.2), the compiled no-op graph (Task 0.5.2), and the `cli.py` skeleton (Task 0.8.1) already exist — Phase 1 tasks name the specific Phase 0 task they build on, not just "Phase 0" generally.

**Rationale (per TRD, preserved intentionally):** validate source quality/coverage before investing in subtopic-discovery/clustering machinery on top of it. Runtime order differs — the Subtopic Agent runs first in the actual graph — this is a deliberate build-order choice to retire sourcing risk first, not a graph reordering.

**Modules:**
- `sourcing/gdelt.py` — GDELT DOC 2.0 client: query by keyword + date range, JSON mode. **Handle the API's 250-record-per-call cap explicitly** — page via repeated queries over sub-windows of the lookback range if a subtopic's article count is expected to exceed that; basic backoff on 429s.
- `sourcing/rss.py` — direct outlet RSS fetch via `feedparser`, filtered by keyword/date.
- `sourcing/google_news_backfill.py` — kept as a structurally separate module (not just a conditionally-called function) so it's obvious from the module boundary that it's non-load-bearing per NFR-3.
- `sourcing/dedup.py` — URL normalization (strip tracking params, trailing slash, scheme-normalize) + exact-match drop; `rapidfuzz` title similarity for cross-source wire-story dupes.
- `reputation/scorer.py` — TRD 4.2 formula exactly: `base_tier_score + w1*norm(domain_age) + w2*norm(backlink_proxy) + w3*norm(gdelt_rss_presence_freq) + w4*legitimacy_flags`, weights loaded from `Settings`, bounded adjustment range enforced (clip, don't sum unbounded).
- `reputation/signals.py` — signal collectors, each independently mockable:
  - GDELT/RSS presence frequency — cheap, derivable from Phase 1's own fetch results, no external dependency — **build and trust this one first**.
  - domain age via `python-whois` — **feasibility risk**: WHOIS is frequently rate-limited/blocked for bulk automated lookups. Fail soft (return `None`/neutral-normalize) rather than raising, and cache aggressively so it's a rare call, not per-run.
  - backlink proxy — **feasibility risk**: there is no simple live-query API for this. The realistic v1 approach is a small pre-downloaded/curated rank list (e.g. Tranco list or Common Crawl host-rank snapshot) shipped as a static data file, not a live network call. Decide this explicitly before writing the collector.
  - HTTPS + about-page heuristic via a plain HEAD/GET (`httpx`).
- `reputation/cache.py` — `domain_reputation` table read/write, staleness-window check (`Settings.reputation.staleness_days`), manual-invalidation function.
- `agents/sourcing_agent.py` — orchestrates: query → backfill-if-below-min-count → dedup → score → filter by `Settings.reputation.min_score_threshold`. Signature: `(keywords: list[str], lookback_days: int) -> list[ScoredArticle]`.
- `data/trusted_outlets.yaml` — seed whitelist (domain → tier), curated manually (Reuters, AP, BBC, Guardian, etc. per TRD example).

**Libraries:** `feedparser`, `httpx`, `python-whois`, `rapidfuzz`, `pyyaml`.

**Tasks (dependency order):**
1. `data/trusted_outlets.yaml` seed list.
2. GDELT client + pagination/backoff handling.
3. RSS client (trusted feeds) + isolated Google News backfill module.
4. Dedup pass over combined results.
5. GDELT/RSS-presence-frequency signal (no external dep — do first).
6. WHOIS + backlink-proxy-snapshot + HTTPS/about-page signals (each with soft-fail).
7. Scoring formula + `domain_reputation` cache read/write (Postgres).
8. Backfill trigger logic: invoke Google News RSS only below `Settings.sourcing.min_primary_article_count`; wrap in try/except that logs-and-continues, never raises (NFR-3).
9. Wire `agents/sourcing_agent.py`.
10. CLI command: `newsresearch dev sourcing-test "<keywords>"` for manual inspection.

**Done when:** given a hardcoded sample subtopic (no Subtopic Agent yet), `agents/sourcing_agent.py` returns a deduplicated, reputation-scored article list from real GDELT/RSS calls; manually spot-check score sanity against 3–5 known-good and known-bad domains; confirm Google News backfill failure (simulate by blocking it) doesn't crash the agent.

### Phase 1 — Story/Task Breakdown (PM decomposition)

The numbered Tasks above are preserved as the task-level breakdown; grouped into stories below with sharper per-task acceptance criteria and explicit cross-phase dependencies. No split was needed — each original task was already a single verifiable unit, apart from adding the Phase-1-library `uv add` step which the original list omitted.

**Story 1.0 — Phase-1 library additions**
Acceptance: `pyproject.toml` gains the Phase 1 libraries and `uv sync` succeeds.
- [ ] Task 1.0.1: `uv add feedparser httpx python-whois rapidfuzz pyyaml`.
      Acceptance: all five packages present in `pyproject.toml`; `uv sync` exits 0.
      Depends on: Phase 0 Task 0.1.1 (uv/pyproject baseline)

**Story 1.1 — Trusted-outlet whitelist seed data exists**
Acceptance: `data/trusted_outlets.yaml` exists with domain→tier entries, loadable as a dict.
- [ ] Task 1.1.1 (orig. Task 1): author `data/trusted_outlets.yaml` (Reuters, AP, BBC, Guardian, etc. per TRD example).
      Acceptance: file parses via `pyyaml`; each entry maps a domain to a tier string; loader test confirms structure.
      Depends on: none

**Story 1.2 — GDELT DOC 2.0 client fetches and paginates beyond the 250-record cap**
Acceptance: a query expected to exceed 250 raw results returns a combined list larger than 250 via sub-window paging, with 429 backoff.
- [ ] Task 1.2.1 (orig. Task 2, part a): `sourcing/gdelt.py` — single-window keyword+date-range query, JSON mode.
      Acceptance: returns parsed article dicts (title, url, domain, published date) from a real GDELT call under 250 results.
      Depends on: Task 1.0.1 (httpx)
- [ ] Task 1.2.2 (orig. Task 2, part b): pagination over sub-windows when expected volume exceeds 250, plus 429 backoff/retry.
      Acceptance: a lookback range known to exceed 250 raw hits returns >250 combined results via internal sub-window splitting; a simulated 429 triggers backoff-and-retry rather than immediate failure.
      Depends on: 1.2.1

**Story 1.3 — RSS sourcing and Google News backfill exist as structurally separate modules**
Acceptance: `sourcing/rss.py` fetches trusted-feed articles by keyword/date; `sourcing/google_news_backfill.py` is independently callable and structurally isolated (NFR-3).
- [ ] Task 1.3.1 (orig. Task 3, part a): `sourcing/rss.py` — fetch official RSS feeds for trusted-tier outlets from `data/trusted_outlets.yaml`, filter by keyword/date.
      Acceptance: given a keyword + lookback window, returns article dicts from at least 2 real trusted-outlet RSS feeds, correctly filtered.
      Depends on: 1.1.1, Task 1.0.1 (feedparser)
- [ ] Task 1.3.2 (orig. Task 3, part b): `sourcing/google_news_backfill.py` — structurally separate module wrapping Google News RSS fetch.
      Acceptance: has its own public function callable independently of `rss.py`/`gdelt.py`; import-graph check confirms `gdelt.py`/`rss.py` never import from it (only the orchestrator does).
      Depends on: Task 1.0.1 (feedparser/httpx)

**Story 1.4 — Deduplication removes exact-URL and cross-source near-duplicate articles**
Acceptance: `sourcing/dedup.py` collapses normalized-identical URLs and rapidfuzz-similar cross-domain titles (wire-story dupes).
- [ ] Task 1.4.1 (orig. Task 4, part a): URL normalization + exact-match drop (strip tracking params, trailing slash, scheme-normalize).
      Acceptance: unit test with `http://x.com/a?utm_source=y` and `https://x.com/a/` confirms only one survives.
      Depends on: Task 1.0.1 (no new dep beyond stdlib/httpx already added)
- [ ] Task 1.4.2 (orig. Task 4, part b): `rapidfuzz` title-similarity cross-source dedup.
      Acceptance: unit test with two near-identical titles from different domains above threshold confirms one is dropped; a below-threshold pair both survive.
      Depends on: 1.4.1, Task 1.0.1 (rapidfuzz)

**Story 1.5 — GDELT/RSS presence-frequency reputation signal (build and trust first, no external dependency)**
Acceptance: `reputation/signals.py`'s presence-frequency collector computes a per-domain frequency score from Phase 1's own fetch results, zero additional network calls.
- [ ] Task 1.5.1 (orig. Task 5): presence-frequency signal collector.
      Acceptance: given a fixed list of (domain, source_type) fetch results, returns a normalized per-domain frequency score deterministically; covered by a unit test with fixed input, no live call.
      Depends on: 1.2.2 (GDELT results shape), 1.3.1 (RSS results shape), 1.4.2 (operates on the deduped set, per stated task order)

**Story 1.6 — Domain-age, backlink-proxy, and HTTPS/about-page signals, each soft-failing independently**
Acceptance: three collectors in `reputation/signals.py` each return a normalized value or neutral/`None` on failure, never raising.

*FLAGGED FEASIBILITY RISKS (carried forward from EXECUTION_PLAN / PRD risk table, not footnotes):*
- **WHOIS rate-limiting**: WHOIS is frequently rate-limited/blocked for bulk automated lookups. The soft-fail path is the primary defense, not a corner case — Task 1.6.1 must not assume WHOIS reliability, and must cache aggressively (feeds Story 1.7's cache) so it's a rare call, not per-run.
- **Backlink-proxy data source**: no live-query API exists for this signal. This is an explicitly unresolved open decision per EXECUTION_PLAN's "Flagged open decisions" — a concrete static snapshot (Tranco list or Common Crawl host-rank) must be chosen and recorded *before* Task 1.6.2 starts, not decided implicitly during implementation.

- [ ] Task 1.6.1 (orig. Task 6, part a): WHOIS domain-age collector (`python-whois`), fail-soft, cache-friendly.
      Acceptance: unit test with mocked `python-whois` raising/timing out confirms the collector returns `None`, never an exception; a live/manual spot-check against 2–3 real domains returns a plausible age.
      Depends on: Task 1.0.1 (python-whois)
- [ ] Task 1.6.2 (orig. Task 6, part b): backlink-proxy signal from a static pre-downloaded rank snapshot (Tranco or Common Crawl host-rank), shipped as a data file under `data/` — not a live network call.
      Acceptance: the snapshot-source decision is explicitly recorded (which source, file location) before implementation; collector loads the file and returns a normalized rank-derived score, or neutral if the domain is absent from the snapshot.
      Depends on: the open decision above being resolved first (blocking — do not start without it)
- [ ] Task 1.6.3 (orig. Task 6, part c): HTTPS + about-page heuristic via HEAD/GET (`httpx`).
      Acceptance: unit test mocks an `httpx` response confirming HTTPS presence + about-page hit both contribute a legitimacy flag; a timeout/error case fails soft (flag = neutral, no raise).
      Depends on: Task 1.0.1 (httpx)

**Story 1.7 — Reputation scoring formula + `domain_reputation` cache implement TRD 4.2 exactly**
Acceptance: `reputation/scorer.py` computes the TRD 4.2 formula with `Settings`-sourced weights, clipped to a bounded adjustment range; `reputation/cache.py` respects `Settings.reputation.staleness_days`.
- [ ] Task 1.7.1 (orig. Task 7, part a): `reputation/scorer.py` — scoring formula, weights from `Settings.reputation`, bounded/clipped adjustment (not summed unbounded).
      Acceptance: unit test with fixed signal inputs + known weights asserts exact formula output; an adversarial extreme-signal combination is confirmed clipped rather than pushing score outside the configured bound.
      Depends on: 1.5.1, 1.6.1, 1.6.2, 1.6.3, Phase 0 Task 0.3.1 (`Settings.reputation.*` weights)
- [ ] Task 1.7.2 (orig. Task 7, part b): `reputation/cache.py` — `domain_reputation` table read/write, staleness-window check, manual-invalidation function.
      Acceptance: `testcontainers`-backed integration test writes a score, reads it back within the staleness window (cache hit, no recompute), confirms manual invalidation forces recompute; a `freezegun`-controlled clock verifies the staleness-window boundary precisely.
      Depends on: Phase 0 Task 0.4.2 (`init_db`/schema for the `domain_reputation` table), 1.7.1

**Story 1.8 — Backfill trigger logic invokes Google News RSS only as a non-load-bearing fallback**
Acceptance: Google News RSS is invoked only when primary (GDELT+RSS) results fall below `Settings.sourcing.min_primary_article_count`; a simulated backfill failure never raises out of the sourcing path (NFR-3).
- [ ] Task 1.8.1 (orig. Task 8): trigger-threshold logic + try/except wrap that logs-and-continues, never raises.
      Acceptance: unit test with primary-count below threshold confirms backfill is invoked; at/above threshold confirms it is NOT invoked; a test forcing `google_news_backfill` to raise confirms the sourcing path logs and continues with primary-only results.
      Depends on: 1.3.2, Phase 0 Task 0.3.1 (`Settings.sourcing.min_primary_article_count`)

**Story 1.9 — Sourcing Agent orchestrates the full pipeline end-to-end**
Acceptance: `agents/sourcing_agent.py(keywords: list[str], lookback_days: int) -> list[ScoredArticle]` — given a hardcoded sample subtopic, returns a deduplicated, reputation-scored, threshold-filtered list from real GDELT/RSS calls.
- [ ] Task 1.9.1 (orig. Task 9): wire `agents/sourcing_agent.py`: query → backfill-if-below-min-count → dedup → score → filter by `Settings.reputation.min_score_threshold`.
      Acceptance: calling it against real GDELT/RSS returns `ScoredArticle` objects each above `Settings.reputation.min_score_threshold`, no duplicate URLs/near-dupe titles present. This is the Phase 1 phase-level Done-when target itself.
      Depends on: 1.2.2, 1.3.1, 1.3.2, 1.4.2, 1.7.1, 1.7.2, 1.8.1

**Story 1.10 — Manual dev CLI for sourcing inspection**
Acceptance: `newsresearch dev sourcing-test "<keywords>"` runs the sourcing agent and prints a human-readable article+score listing.
- [ ] Task 1.10.1 (orig. Task 10): add `dev sourcing-test` typer subcommand.
      Acceptance: `uv run newsresearch dev sourcing-test "<keywords>"` prints each returned article's URL, domain, and reputation score to stdout; exits 0.
      Depends on: 1.9.1, Phase 0 Task 0.8.1 (typer CLI app must exist to extend)

**Story 1.11 — Phase 1 Done-when verified end-to-end (integration/demo)**
Acceptance: matches the phase-level Done-when block above.
- [ ] Task 1.11.1: manual spot-check pass — run `sourcing_agent` against 3–5 known-good domains (Reuters, AP, BBC, etc.) and 3–5 known-bad/low-quality domains.
      Acceptance: a recorded good-vs-bad score comparison shows good domains consistently scoring higher; any inversion is filed as a defect against Task 1.7.1 (formula) or the relevant Task 1.6.x (signal), not against "Phase 1" generally.
      Depends on: 1.9.1
- [ ] Task 1.11.2: backfill-failure resilience check — force/block `google_news_backfill` and confirm `sourcing_agent` still returns primary-source results without crashing.
      Acceptance: with backfill forced to raise/timeout, `sourcing_agent(...)` still returns a non-crashing result set built from GDELT+RSS alone — directly re-verifies NFR-3/Task 1.8.1 under real (not just unit-mocked) conditions.
      Depends on: 1.9.1, 1.8.1

---

## Phase 2 — Subtopic Agent + Gate 1 + Topical Clustering Agent + Gate 2

**Rationale:** validate human-in-the-loop UX and clustering quality before adding expensive LLM agents downstream.

**Modules:**
- `clustering/embeddings.py` — thin wrapper over `llm/models.py::get_embeddings()`; `embed(texts: list[str]) -> np.ndarray` convenience for clustering code.
- `clustering/cluster.py` — `cluster(vectors) -> labels`, HDBSCAN primary, KMeans fallback when article count < `Settings.clustering.kmeans_fallback_threshold`.
- `agents/subtopic_agent.py` — LLM proposes N candidates (`llm/prompts/subtopic_propose.txt`, via `get_chat_model("subtopic").with_structured_output(...)`) → broad topic-scoped fetch (reuses Phase 1 `sourcing_agent`, not yet subtopic-filtered) → embed+cluster → reconcile (merge candidates mapping to same cluster, split clusters spanning multiple candidates, drop unsupported candidates) → rank by volume/distinctiveness → truncate to `Settings.pipeline.max_subtopics` → retain excess for "also detected."
- `agents/topical_clustering_agent.py` — coarse per-subtopic clustering, reuses `clustering/cluster.py` and Phase 1's sourcing.
- `graph/nodes/gate1.py`, `graph/nodes/gate2.py` — LangGraph `interrupt()`-based nodes; resume via `Command(resume=...)`. Requires the `PostgresSaver` checkpointer from Phase 0 — verify gate state actually survives a process restart, not just an in-process pause.
- `reports/gate2_report.py` — pure aggregation (cluster sizes, sample headlines, source spread), zero LLM calls.

**Libraries:** `hdbscan`, `scikit-learn`.

**Tasks:**
1. `clustering/cluster.py` HDBSCAN + KMeans fallback.
2. Subtopic-propose prompt + pydantic schema for candidate list.
3. Broad fetch (topic-scoped) via Phase 1 sourcing agent.
4. Embed + cluster broad set; reconciliation logic (merge/split/drop).
5. Rank/cap/excess-retention.
6. `graph/nodes/gate1.py`: interrupt, present candidates + excess, accept approve/edit, re-run reconciliation on edit.
7. Fan-out wiring in `graph/build.py` using LangGraph `Send` per approved subtopic (concurrent branches).
8. `agents/topical_clustering_agent.py` per subtopic.
9. `reports/gate2_report.py` + `graph/nodes/gate2.py` interrupt, per-subtopic blocking.
10. Extend CLI harness: render Gate 1 candidates, accept stdin approve/edit, render Gate 2 report per subtopic.

**Done when:** a real topic string run via the CLI harness produces a reviewable subtopic list at Gate 1; approving proceeds to concurrent per-subtopic sourcing+clustering; Gate 2 shows a correct zero-LLM-cost report per subtopic; kill and restart the process mid-gate to confirm the `PostgresSaver` checkpoint actually resumes correctly (not just that functions work in isolation); confirm the Subtopic Agent's LLM calls appear as traces in the local Langfuse UI.

### Phase 2 — Story/Task Breakdown (PM decomposition)

**Depends on: Phase 0 complete, Phase 1 complete.** Tasks below name the specific prior-phase deliverable they build on (e.g. Phase 1 Task 1.9.1's `sourcing_agent`, Phase 0 Task 0.5.1's state schema), not just "Phase 0/1" generally.

The numbered Tasks above are preserved as the task-level backbone. One prerequisite named only in Modules (the `clustering/embeddings.py` wrapper) is promoted to its own task since Task 1 (`cluster.py`) depends on it but the original list didn't number it separately. Original Task 9 is split (report generation vs. the interrupt node itself), matching the Phase 0/1 precedent of splitting only where a task bundles more than one independently-verifiable outcome.

**Story 2.0 — Phase-2 library additions**
Acceptance: `pyproject.toml` gains `hdbscan` + `scikit-learn`, `uv sync` succeeds.
- [ ] Task 2.0.1: `uv add hdbscan scikit-learn`.
      Acceptance: both packages present; `uv sync` exits 0.
      Depends on: Phase 0 Task 0.1.1

**Story 2.1 — Clustering primitives: embeddings wrapper + HDBSCAN/KMeans**
Acceptance: given fixed embedding vectors, `cluster()` returns correct labels via HDBSCAN, falling back to KMeans below `Settings.clustering.kmeans_fallback_threshold` — this is one of the plan's explicitly unit-testable-now items (clustering functions given fixed vectors).
- [ ] Task 2.1.1 (named in Modules, not separately numbered above): `clustering/embeddings.py` — thin wrapper, `embed(texts: list[str]) -> np.ndarray` over Phase 0's `get_embeddings()`.
      Acceptance: returns a correctly-shaped array using the `Settings.embeddings.backend`-selected implementation from Phase 0 Task 0.6.2.
      Depends on: Phase 0 Task 0.6.2
- [ ] Task 2.1.2 (orig. Task 1): `clustering/cluster.py` — `cluster(vectors) -> labels`, HDBSCAN primary, KMeans fallback.
      Acceptance: unit test with a fixed multi-cluster embedding fixture confirms correct HDBSCAN grouping; a fixture below `kmeans_fallback_threshold` confirms KMeans is used instead.
      Depends on: 2.0.1, 2.1.1

**Story 2.2 — Subtopic Agent proposes, fetches, clusters, and reconciles candidates**
Acceptance: given a topic string, `agents/subtopic_agent.py` returns a ranked, capped subtopic list plus a retained "also detected" excess set.
- [ ] Task 2.2.1 (orig. Task 2): subtopic-propose prompt (`llm/prompts/subtopic_propose.txt`) + pydantic candidate-list schema, via `get_chat_model("subtopic").with_structured_output(...)`.
      Acceptance: given a sample topic, returns N schema-conformant candidate labels; call is traced in the local Langfuse UI.
      Depends on: Phase 0 Task 0.6.1, 0.6.3
- [ ] Task 2.2.2 (orig. Task 3): broad topic-scoped fetch reusing Phase 1's `sourcing_agent` (not yet subtopic-filtered).
      Acceptance: calling Phase 1 Task 1.9.1's agent with topic-level keywords returns an article set spanning multiple candidate subtopics.
      Depends on: Phase 1 Task 1.9.1
- [ ] Task 2.2.3 (orig. Task 4): embed+cluster the broad set; reconciliation logic (merge candidates mapping to the same cluster, split clusters spanning multiple candidates, drop unsupported candidates).
      Acceptance: unit test with a fixed embedding fixture triggers each of merge/split/drop independently under its respective condition.
      Depends on: 2.1.2, 2.2.1, 2.2.2
- [ ] Task 2.2.4 (orig. Task 5): rank by volume/distinctiveness, truncate to `Settings.pipeline.max_subtopics`, retain excess as "also detected."
      Acceptance: given more reconciled clusters than the cap, output length equals the cap exactly; every excess cluster is recorded in a separate field, none silently dropped.
      Depends on: 2.2.3, Phase 0 Task 0.3.1

**Story 2.3 — Gate 1: human approves/edits the subtopic list, durably**
Acceptance: `graph/nodes/gate1.py` interrupts with candidates + excess, accepts approve/edit via `Command(resume=...)`, re-runs reconciliation on edit, and the interrupt state survives a process restart.
- [ ] Task 2.3.1 (orig. Task 6): `graph/nodes/gate1.py` — interrupt node; approve/edit handling; edit triggers re-reconciliation.
      Acceptance: an approve-resume proceeds with the original list unchanged; an edit-resume (e.g. removing one candidate) re-triggers Task 2.2.3's reconciliation on the edited set.
      Depends on: 2.2.4
- [ ] Task 2.3.2 (implied by the phase Done-when, not separately numbered above): kill-and-restart durability check specifically for Gate 1's interrupt state.
      Acceptance: killing the process mid-Gate-1-interrupt and restarting, then resuming via the same `thread_id`, produces the correct pending-approval state read back from Postgres — not merely an in-process pause.
      Depends on: 2.3.1, Phase 0 Task 0.5.2

**Story 2.4 — Concurrent per-subtopic fan-out**
Acceptance: approving Gate 1 fans out to one concurrently-running branch per approved subtopic.
- [ ] Task 2.4.1 (orig. Task 7): fan-out wiring in `graph/build.py` using LangGraph `Send` per approved subtopic.
      Acceptance: approving N subtopics produces N concurrent branches, each carrying its own subtopic sub-state per Phase 0 Task 0.5.1's schema.
      Depends on: 2.3.1, Phase 0 Task 0.5.1

**Story 2.5 — Topical clustering per subtopic**
Acceptance: `agents/topical_clustering_agent.py` produces coarse per-subtopic clusters, reusing Phase 1 sourcing and Story 2.1's clustering primitives.
- [ ] Task 2.5.1 (orig. Task 8): per-subtopic sourcing (Phase 1) + coarse clustering (2.1.2).
      Acceptance: given one approved subtopic, returns topical clusters with plausible article groupings on a real subtopic — manually spot-checked, since clustering quality is this phase's flagged risk.
      Depends on: 2.1.2, 2.4.1, Phase 1 Task 1.9.1

**Story 2.6 — Gate 2: zero-LLM-cost cluster report, per-subtopic blocking**
Acceptance: `reports/gate2_report.py` aggregates cluster sizes/sample headlines/source spread with zero LLM calls; `graph/nodes/gate2.py` blocks each subtopic branch independently until that subtopic's report is reviewed.
- [ ] Task 2.6.1 (orig. Task 9, part a): `reports/gate2_report.py` — pure aggregation, zero LLM calls.
      Acceptance: given Story 2.5's cluster output, produces cluster-size/sample-headline/source-spread fields with zero calls to `get_chat_model` (asserted via a call-count test double).
      Depends on: 2.5.1
- [ ] Task 2.6.2 (orig. Task 9, part b): `graph/nodes/gate2.py` — interrupt node, blocking per subtopic branch independently.
      Acceptance: one subtopic branch paused at Gate 2 does not block a sibling branch's own Gate 2 from being reviewed and resumed independently.
      Depends on: 2.6.1, 2.4.1

**Story 2.7 — CLI harness renders both gates interactively**
Acceptance: the CLI harness renders Gate 1 candidates, accepts stdin approve/edit, then renders the Gate 2 report per subtopic and accepts continue.
- [ ] Task 2.7.1 (orig. Task 10): extend `cli.py` — render Gate 1 candidates/excess, accept stdin approve/edit; render Gate 2 report per subtopic, prompt to continue.
      Acceptance: `uv run newsresearch run "<topic>"` pauses at Gate 1 with a readable candidate list, accepts a stdin edit, proceeds to per-subtopic Gate 2 output, and resumes on confirmation.
      Depends on: 2.3.1, 2.6.2, Phase 0 Task 0.8.1

**Story 2.8 — Phase 2 Done-when verified end-to-end (integration/demo)**
Acceptance: all bullets in the phase-level "Done when" block above hold simultaneously.
- [ ] Task 2.8.1: full CLI run on a real topic through Gate 1 → fan-out → Gate 2, confirming per-subtopic zero-cost report correctness.
      Depends on: 2.7.1
- [ ] Task 2.8.2: kill/restart mid-gate durability re-check under the full real pipeline (not just Task 2.3.2's isolated Gate 1 check) — verify at both Gate 1 and Gate 2.
      Depends on: 2.3.2, 2.6.2
- [ ] Task 2.8.3: confirm the Subtopic Agent's LLM calls appear as traces in the local Langfuse UI.
      Depends on: 2.2.1, Phase 0 Task 0.7.2

---

## Phase 3 — Claim Extraction + Framing Clustering + Summarization

**Rationale:** validate claim-cluster quality manually before wiring in the highest-risk bias/briefing stage.

**Modules:**
- `sourcing/fulltext.py` — `trafilatura`-based fetch, strictly in-memory (returns a plain string, no path to any `persistence/` write function — enforce via module boundary, not just discipline).
- `agents/claim_extraction_agent.py` — small model via `get_chat_model("claim_extraction").with_structured_output(Claim)`, structured output `{claim_text, subject, attributed_source}` per article.
- `clustering/claim_clustering.py` — embed `claim_text`, cluster across *all* articles in the subtopic (not per-article) so assert/omit membership falls out naturally.
- `agents/sentiment.py` — auxiliary per-article/claim sentiment (small model call or lightweight lexicon — explicitly not the clustering axis, per FR-14/NFR risk table).
- `agents/summarization_agent.py` — small/medium model, one call per claim cluster.

**Libraries:** `trafilatura`.

**Tasks:**
1. Full-text fetch per article in a topical cluster, passed directly into extraction — no persistence path exists (schema already excludes body text).
2. Claim extraction prompt + schema; small model.
3. Claim clustering across the subtopic's full article set.
4. Sentiment attached as an attribute, not fed into clustering.
5. Persist `claim_clusters` + `claim_cluster_articles` (`asserts`/`omits` relation, `claim_text`) to Postgres.
6. Summarization Agent per cluster.

**Done when:** for a Gate-2-cleared subtopic, Phase 3 end-to-end produces claim clusters with correct assert/omit article membership and a per-cluster summary; manually spot-check several clusters that grouping reflects real agreement/disagreement rather than embedding noise; confirm claim-extraction traces are visible and inspectable in Langfuse.

### Phase 3 — Story/Task Breakdown (PM decomposition)

**Depends on: Phase 0–2 complete.** Every task assumes a Gate-2-cleared topical cluster (Phase 2 Task 2.6.2) already exists as input.

**Story 3.0 — Phase-3 library additions**
Acceptance: `pyproject.toml` gains `trafilatura`, `uv sync` succeeds.
- [ ] Task 3.0.1: `uv add trafilatura`.
      Acceptance: package present; `uv sync` exits 0.
      Depends on: Phase 0 Task 0.1.1

**Story 3.1 — In-memory full-text fetch, no persistence path**
Acceptance: `sourcing/fulltext.py` returns article body text purely in-memory, with no importable path to any `persistence/` write function — enforced via module boundary, not just discipline.
- [ ] Task 3.1.1 (orig. Task 1): `trafilatura`-based fetch per article in a Gate-2-cleared topical cluster.
      Acceptance: given an article URL, returns extracted body text as a plain string; an import-graph check confirms `sourcing/fulltext.py` has zero imports from `persistence/`.
      Depends on: Phase 2 Task 2.6.2, Task 3.0.1

**Story 3.2 — Claim extraction produces structured claims per article**
Acceptance: `agents/claim_extraction_agent.py` returns `{claim_text, subject, attributed_source}` per article via a small model with structured output.
- [ ] Task 3.2.1 (orig. Task 2): claim-extraction prompt + pydantic `Claim` schema, `get_chat_model("claim_extraction").with_structured_output(Claim)`.
      Acceptance: given full text from 3.1.1, returns one or more schema-conformant `Claim` objects per article; call is traced and inspectable in Langfuse.
      Depends on: 3.1.1, Phase 0 Task 0.6.1, 0.6.3

**Story 3.3 — Claims cluster across the whole subtopic, not per-article**
Acceptance: `clustering/claim_clustering.py` embeds `claim_text` and clusters across *all* articles in the subtopic, so assert/omit membership falls out of cluster membership itself.
- [ ] Task 3.3.1 (orig. Task 3): embed `claim_text` across the subtopic's full claim set; cluster (reusing Phase 2 Task 2.1.2's `cluster()`).
      Acceptance: given claims from multiple articles asserting/omitting the same fact, they land in the same cluster with correct per-article assert/omit membership recoverable from cluster assignment.
      Depends on: 3.2.1, Phase 2 Task 2.1.2

**Story 3.4 — Sentiment is an auxiliary attribute, never a clustering axis**
Acceptance: `agents/sentiment.py` computes per-article/claim sentiment and attaches it as metadata; Story 3.3's cluster assignments are provably unaffected by it (FR-14 / the plan's own risk table on sentiment-as-bias-proxy).
- [ ] Task 3.4.1 (orig. Task 4): sentiment collector (small model call or lightweight lexicon), attached as an attribute on claims/articles.
      Acceptance: a test confirms cluster assignments from 3.3.1 are identical whether or not sentiment has been computed — sentiment is never read by the clustering step.
      Depends on: 3.3.1

**Story 3.5 — Claim clusters persisted with assert/omit article membership**
Acceptance: `claim_clusters` + `claim_cluster_articles` persist `claim_text` and the `asserts`/`omits` relation per article.
- [ ] Task 3.5.1 (orig. Task 5): schema/write path for `claim_clusters` + `claim_cluster_articles`.
      Acceptance: `testcontainers`-backed test writes a subtopic's clusters and confirms each article's `asserts`/`omits` relation is queryable and matches 3.3.1's in-memory cluster assignment exactly.
      Depends on: 3.3.1, 3.4.1, Phase 0 Task 0.4.2

**Story 3.6 — Per-cluster summarization**
Acceptance: `agents/summarization_agent.py` produces one summary per persisted claim cluster using a small/medium model.
- [ ] Task 3.6.1 (orig. Task 6): summarization agent, one call per cluster.
      Acceptance: given a persisted claim cluster (3.5.1), returns a summary whose content is manually spot-checked against the cluster's actual claims — LLM prompt-quality is explicitly a needs-manual-inspection item per the plan's testing strategy, not an automated assertion.
      Depends on: 3.5.1, Phase 0 Task 0.6.1

**Story 3.7 — Phase 3 Done-when verified end-to-end (integration/demo)**
Acceptance: matches the phase-level Done-when block above.
- [ ] Task 3.7.1: manual spot-check of several real claim clusters on a Gate-2-cleared subtopic for correct assert/omit membership and summary plausibility.
      Acceptance: reviewer confirms grouping reflects real agreement/disagreement rather than embedding noise for at least several inspected clusters; any failure is filed against Task 3.3.1 (clustering) or 3.6.1 (summarization), not against "Phase 3" generally.
      Depends on: 3.6.1
- [ ] Task 3.7.2: confirm claim-extraction traces are visible and inspectable in the local Langfuse UI.
      Depends on: 3.2.1, Phase 0 Task 0.7.2

---

## Phase 4 — Bias & Framing Agent + Briefing Agent

**Rationale (per TRD):** highest-risk, highest-value stage — budget the most manual review time.

**Modules:**
- `agents/bias_framing_agent.py` — large model; per-cluster + per-source framing/stance label (descriptive, not a left/center/right scale per PRD risk table), claim-level include/omit/emphasis comparison.
- `agents/briefing_agent.py` — large model, sequential after bias agent (consumes its structured output directly, not re-derived from raw clusters).
- `agents/snapshot_assembly.py` — pure aggregation, no LLM call; reads back from Postgres, assembles one JSON run artifact.

**Tasks:**
1. Bias/framing prompt: descriptive labels, operates over `claim_clusters` + article/domain metadata already persisted.
2. Claim-level comparison: largely a re-presentation of `claim_cluster_articles` plus LLM-generated framing text — not a new clustering step.
3. Briefing Agent: sequential dependency enforced at the graph level (edge, not just prompt instruction), consumes bias agent's structured output.
4. Persist `claim_clusters.framing_label`, `briefings` table (consensus/disputed/omissions).
5. Snapshot Assembly reads and assembles final JSON artifact per subtopic.
6. Use Langfuse's session view to review a full subtopic's trace chain (bias agent → briefing agent) during manual QA — this is the main practical payoff of the tracing investment for this phase.

**Done when:** a chained Phase 1–4 run on a real subtopic produces a briefing a human finds plausible on manual review against source articles — validation is inherently qualitative here (PRD's flagged eval gap), so "done" = operator review across at least 2–3 real topics, not an automated check. Log each reviewed run as an MLflow run so config-vs-quality observations accumulate over time.

### Phase 4 — Story/Task Breakdown (PM decomposition)

**Depends on: Phase 0–3 complete.** Highest-risk, highest-value stage per the TRD — budget the most manual review time; validation here is inherently qualitative (PRD's flagged eval gap), not a substitute for writing sharp criteria.

**Story 4.1 — Bias/Framing Agent labels clusters and sources descriptively**
Acceptance: `agents/bias_framing_agent.py` produces a descriptive (never a left/center/right scale, per the PRD risk table) framing/stance label per cluster and per source, using a large model over already-persisted `claim_clusters` + article/domain metadata.
- [ ] Task 4.1.1 (orig. Task 1): bias/framing prompt + structured output, operating over Phase 3 Task 3.5.1's persisted data.
      Acceptance: given a persisted subtopic's claim clusters, returns a descriptive label per cluster and per source; the output schema itself rejects any left/center/right-style scalar field; call is traced in Langfuse.
      Depends on: Phase 3 Task 3.5.1, Phase 0 Task 0.6.1

**Story 4.2 — Claim-level include/omit/emphasis comparison**
Acceptance: for each cluster, a claim-level comparison shows what's included/omitted/emphasized differently relative to other clusters — built from existing `claim_cluster_articles` relations plus LLM framing text, not a new clustering pass.
- [ ] Task 4.2.1 (orig. Task 2): claim-level comparison generator.
      Acceptance: comparison output for a subtopic re-presents Phase 3 Task 3.5.1's `asserts`/`omits` relation per cluster with attached LLM framing text; no new embedding/clustering call is made in this path (verified by absence of a clustering-function invocation here).
      Depends on: 4.1.1

**Story 4.3 — Briefing Agent synthesizes consensus vs. disputed, sequentially after bias agent**
Acceptance: `agents/briefing_agent.py` runs only after the bias/framing agent completes for that subtopic, consuming its structured output directly (not re-derived from raw clusters) — enforced as a graph edge, not a prompt instruction.
- [ ] Task 4.3.1 (orig. Task 3): briefing agent + graph-level sequential edge (bias_framing_agent → briefing_agent).
      Acceptance: graph-topology inspection confirms an explicit edge enforcing bias-before-briefing; the briefing agent's input parameter is the bias agent's structured output object, not raw `claim_clusters`.
      Depends on: 4.1.1, Phase 0 Task 0.5.2

**Story 4.4 — Framing labels and briefings persisted**
Acceptance: `claim_clusters.framing_label` is populated; a `briefings` table stores consensus/disputed/omissions content.
- [ ] Task 4.4.1 (orig. Task 4): persist `claim_clusters.framing_label` + `briefings` table schema/write path.
      Acceptance: `testcontainers`-backed test confirms `framing_label` is written per cluster and a `briefings` row exists per subtopic with consensus/disputed/omissions fields populated.
      Depends on: 4.1.1, 4.3.1, Phase 0 Task 0.4.2

**Story 4.5 — Snapshot assembly produces the final per-subtopic JSON artifact**
Acceptance: `agents/snapshot_assembly.py` performs pure aggregation (no LLM call), reading back from Postgres to produce one structured JSON artifact per subtopic (sub-topic → clusters → summaries → bias labels → briefing).
- [ ] Task 4.5.1 (orig. Task 5): snapshot assembly reader/assembler.
      Acceptance: given a fully-processed subtopic, output JSON contains every named layer sourced entirely from Postgres reads; zero calls to `get_chat_model` during assembly (verified via a call-count test double).
      Depends on: 4.4.1, Phase 3 Task 3.6.1

**Story 4.6 — Phase 4 Done-when verified via qualitative operator review (flagged eval gap)**
Acceptance: matches the phase-level Done-when block — operator finds the briefing plausible against source articles across at least 2–3 real topics; each reviewed run logged as an MLflow run.
- [ ] Task 4.6.1 (orig. Task 6 + Done-when): use Langfuse's session view to review a full subtopic's bias→briefing trace chain during manual QA, across at least 2–3 real topics.
      Acceptance: for each reviewed topic, the operator records a plausible/not-plausible verdict against real source articles — this is explicitly qualitative, not an automatable pass/fail (per the PRD's flagged eval-harness gap); each reviewed run has a corresponding MLflow run for later config-vs-quality comparison.
      Depends on: 4.5.1, Phase 0 Task 0.7.3
      Note for the acceptance-verifier agent: this task's core verdict is inherently **UNVERIFIABLE HERE** for an automated check — mark it BLOCKED pending explicit operator sign-off rather than substituting an automated judgment for the required qualitative review.

---

## Phase 5 — Timeline Agent + Scheduling + Cross-Run Matching

**Modules:**
- `agents/timeline_agent.py` — triggered only on scheduled non-first runs for a topic hash.
- `persistence/topic_identity.py` — canonicalize (lowercase, whitespace-normalize, optional LLM-normalize call) → hash, keys the `topics` table.
- `clustering/subtopic_matching.py` — cosine similarity of current vs. prior subtopic embeddings against `Settings.clustering.subtopic_match_threshold`.
- `scheduling/scheduler.py` — APScheduler job or documented cron entrypoint invoking `newsresearch/scheduling/run_scheduled.py`.

**Libraries:** `apscheduler`.

**Tasks:**
1. Topic canonicalization → hash pipeline.
2. Scheduler entrypoint: looks up due topics by cadence, invokes the full graph headlessly. **Open design decision the TRD doesn't resolve** — Gate 1/2 need a non-interactive policy for scheduled runs: auto-approve on scheduled reruns, or fail-and-notify if gates can't be cleared unattended. Decide before building this task.
3. Timeline Agent: fetch prior runs for topic hash, embed current subtopic labels, match by threshold, compute drift (sentiment shift, volume change, new/dropped claim clusters).
4. Persist `subtopic_matches`, trend report as part of run artifact.
5. Verify NFR-3 specifically under the scheduled/unattended path (not just Phase 1's interactive path) — simulate Google News RSS failure during an actual scheduled run and confirm graceful degradation.
6. Confirm unattended scheduled runs still produce Langfuse traces and MLflow runs — no gaps in observability just because a human isn't watching.

**Done when:** running the same topic twice (simulated second scheduled run) correctly matches unchanged subtopics, flags a deliberately-varied one as newly emerged, produces a sane drift report; a real APScheduler/cron job fires an actual unattended run at least once, fully traced.

### Phase 5 — Story/Task Breakdown (PM decomposition)

**Depends on: Phase 0–4 complete.** Primary risk retired here: cross-run identity/drift correctness and unattended scheduled-run reliability, plus a gate policy the TRD never specified — that policy is promoted to its own blocking task rather than left implicit inside the scheduler task.

**Story 5.0 — Phase-5 library additions**
Acceptance: `pyproject.toml` gains `apscheduler`, `uv sync` succeeds.
- [ ] Task 5.0.1: `uv add apscheduler`.
      Acceptance: package present; `uv sync` exits 0.
      Depends on: Phase 0 Task 0.1.1

**Story 5.1 — Topic canonicalization and identity hashing**
Acceptance: `persistence/topic_identity.py` canonicalizes (lowercase, whitespace-normalize, optional LLM-normalize) and hashes a topic string, keying the `topics` table so trivial rephrasings resolve to the same tracked topic (FR-21).
- [ ] Task 5.1.1 (orig. Task 1): canonicalization → hash pipeline.
      Acceptance: unit test confirms `"Ukraine War"`, `" ukraine war "`, and an LLM-normalized rephrasing all hash identically — one of the plan's explicitly unit-testable-now items.
      Depends on: none

**Story 5.2 — Gate policy for scheduled/unattended runs is decided and implemented (open design decision)**
Acceptance: an explicit, documented policy exists for what happens when Gate 1/2 can't be cleared unattended (auto-approve vs. fail-and-notify) — the TRD/plan does not resolve this, so it must be decided here, not implicitly during coding.
- [ ] Task 5.2.1 (blocking open decision, named in orig. Task 2's text and the "Flagged open decisions" summary but not previously its own task): decide and record the gate policy (auto-approve vs. fail-and-notify) for scheduled reruns.
      Acceptance: the decision is recorded in this document (or a config default) before Task 5.2.2 begins; it names one of the two options explicitly, not left ambiguous.
      Depends on: none — must precede 5.2.2
- [ ] Task 5.2.2 (orig. Task 2): scheduler entrypoint — looks up due topics by cadence, invokes the full graph headlessly, applying the Task 5.2.1 policy at Gate 1/2.
      Acceptance: a due topic runs headlessly through both gates per the decided policy with zero interactive stdin/UI involvement.
      Depends on: 5.2.1, 5.1.1, Phase 2 Task 2.3.1, Phase 2 Task 2.6.2

**Story 5.3 — Cross-run subtopic matching and drift computation**
Acceptance: `clustering/subtopic_matching.py` matches current vs. prior subtopic embeddings by cosine similarity against `Settings.clustering.subtopic_match_threshold`; `agents/timeline_agent.py` computes drift (sentiment shift, volume change, new/dropped claim clusters) for matched and unmatched subtopics.
- [ ] Task 5.3.1 (orig. Task 3, part a): `subtopic_matching.py` — cosine-similarity matcher against the `Settings`-sourced threshold.
      Acceptance: given fixed embedding vectors for a matched pair and an unmatched pair, correctly classifies each with no live LLM call — one of the plan's explicitly unit-testable-now items (subtopic cross-run matching math).
      Depends on: Phase 2 Task 2.1.1, Phase 0 Task 0.3.1
- [ ] Task 5.3.2 (orig. Task 3, part b): `agents/timeline_agent.py` — triggered only on scheduled non-first runs for a topic hash; fetches prior runs, computes drift.
      Acceptance: given two runs for the same topic hash with one deliberately-varied subtopic, the drift report correctly separates matched-continuing subtopics from the newly-emerged one, with sentiment/volume deltas attached.
      Depends on: 5.3.1, 5.1.1

**Story 5.4 — Trend report persisted as part of the run artifact**
Acceptance: `subtopic_matches` + a trend report are persisted and included in the run's assembled artifact.
- [ ] Task 5.4.1 (orig. Task 4): persist `subtopic_matches` table + trend report attached to the run artifact (extends Phase 4's snapshot assembly).
      Acceptance: `testcontainers`-backed test confirms `subtopic_matches` rows exist per matched pair and the trend report is retrievable alongside the Phase 4 snapshot for the same `run_id`.
      Depends on: 5.3.2, Phase 4 Task 4.5.1

**Story 5.5 — NFR-3 graceful degradation re-verified under the unattended path**
Acceptance: a simulated Google News RSS failure during an actual scheduled (not interactive) run degrades gracefully — the same guarantee as Phase 1, now proven headless.
- [ ] Task 5.5.1 (orig. Task 5): force/block `google_news_backfill` during a real scheduled-run invocation and confirm no crash, primary-source results still returned.
      Acceptance: identical guarantee to Phase 1 Task 1.11.2, re-run specifically through the Task 5.2.2 scheduler entrypoint rather than the interactive CLI path.
      Depends on: 5.2.2, Phase 1 Task 1.8.1

**Story 5.6 — Observability holds under unattended runs**
Acceptance: a scheduled run with no human watching still produces Langfuse traces and an MLflow run, identical in completeness to an interactive run.
- [ ] Task 5.6.1 (orig. Task 6): confirm the Phase 0 Story 0.7 callback wiring is attached identically in the scheduler entrypoint's `graph.invoke()` call, not just `cli.py`'s.
      Acceptance: an unattended scheduled run produces the same three observability artifacts (a `run_costs` row, a Langfuse trace, an MLflow run) as Phase 0 Task 0.7.4's interactive-path check.
      Depends on: 5.2.2, Phase 0 Task 0.7.4

**Story 5.7 — Phase 5 Done-when verified end-to-end (integration/demo)**
Acceptance: matches the phase-level Done-when block above.
- [ ] Task 5.7.1: simulated second scheduled run of the same topic — confirm unchanged subtopics matched, a deliberately-varied one flagged as newly emerged, a sane drift report produced.
      Depends on: 5.4.1
- [ ] Task 5.7.2: at least one real APScheduler/cron job fires an actual unattended run, fully traced end-to-end.
      Depends on: 5.6.1, 5.2.2

---

## Phase 6 — Streamlit + Jupyter Front Ends

**Modules:**
- `app/streamlit_app.py` — topic input, Gate 1/2 interactive UI, snapshot/briefing/timeline views, calling `newsresearch.graph` + `newsresearch.persistence` only (no pipeline logic in the UI layer).
- `notebooks/exploration.ipynb` — same core module, exploratory/ad hoc charting.

**Libraries:** `streamlit`, `jupyter`/`ipykernel`.

**Tasks:**
1. Streamlit: topic form → invoke graph to Gate 1 → render/approve/edit → resume → Gate 2 per subtopic → resume → final views.
2. State handling: Streamlit re-executes the script per interaction — rely on the `PostgresSaver` checkpointer + `run_id` lookup, not `st.session_state` alone, so gate state is durable (this is exactly why Phase 0 chose a Postgres-backed saver over an in-memory one).
3. Jupyter notebook: thin wrapper over the same graph/core functions, auto-approve or manual cell-stepping through gates, charting over `articles`/`claim_clusters`/`briefings`.
4. Add a link/deep-link from the Streamlit run view to the corresponding Langfuse trace (`run_id`-tagged) for one-click debugging during operation, not just during development.

**Done when:** a full run is executable start-to-finish through Streamlit including both gates; the same run is inspectable from the notebook against the same Postgres database; the Langfuse trace for that run is reachable from the Streamlit UI.

### Phase 6 — Story/Task Breakdown (PM decomposition)

**Depends on: Phase 0–5 complete.** Usability only — deliberately last, no new pipeline logic; both front ends must call `newsresearch.graph` + `newsresearch.persistence` only, never reimplement pipeline behavior in the UI layer.

**Story 6.0 — Phase-6 library additions**
Acceptance: `pyproject.toml` gains `streamlit`, `jupyter`/`ipykernel`, `uv sync` succeeds.
- [ ] Task 6.0.1: `uv add streamlit jupyter ipykernel`.
      Acceptance: all three present; `uv sync` exits 0.
      Depends on: Phase 0 Task 0.1.1

**Story 6.1 — Streamlit UI drives a full run through both gates, calling only graph + persistence**
Acceptance: `app/streamlit_app.py`'s topic form invokes the graph through Gate 1 approve/edit, resumes to per-subtopic Gate 2, resumes again to final views — with zero pipeline logic living in the UI layer.
- [ ] Task 6.1.1 (orig. Task 1): Streamlit flow — topic form → Gate 1 render/approve/edit → resume → Gate 2 per subtopic → resume → final views.
      Acceptance: a manual walkthrough completes a real topic run through both gates in the browser; an import/code-review check confirms `streamlit_app.py` contains no direct sourcing/clustering/agent logic, only `graph.invoke`/`Command(resume=...)` calls and persistence reads.
      Depends on: Phase 2 Task 2.3.1, Phase 2 Task 2.6.2, Phase 4 Task 4.5.1
- [ ] Task 6.1.2 (orig. Task 2): durable gate-state handling via the `PostgresSaver` checkpointer + `run_id` lookup, not `st.session_state` alone.
      Acceptance: reloading the Streamlit page mid-gate (simulating its per-interaction script re-execution) still resumes correctly from Postgres, not from lost in-memory session state.
      Depends on: 6.1.1, Phase 0 Task 0.5.2

**Story 6.2 — Jupyter notebook provides exploratory access to the same core**
Acceptance: `notebooks/exploration.ipynb` wraps the same graph/core functions as the Streamlit app, supports auto-approve or manual cell-stepping through gates, and charts over `articles`/`claim_clusters`/`briefings` from the same Postgres database.
- [ ] Task 6.2.1 (orig. Task 3): notebook wrapper — auto-approve or manual cell-stepping, charting.
      Acceptance: running the notebook against the same Postgres instance as a prior Streamlit run reproduces/inspects that run's data (the same `run_id` queryable from both).
      Depends on: 6.1.1

**Story 6.3 — One-click Langfuse trace access from the Streamlit run view**
Acceptance: a link/deep-link from the Streamlit run view opens the corresponding `run_id`-tagged Langfuse trace.
- [ ] Task 6.3.1 (orig. Task 4): deep-link from the run view to the Langfuse trace URL.
      Acceptance: clicking the link from a completed run's Streamlit view opens the correct trace at `localhost:3000` for that exact `run_id`.
      Depends on: 6.1.1, Phase 0 Task 0.7.2

**Story 6.4 — Phase 6 Done-when verified end-to-end (integration/demo)**
Acceptance: matches the phase-level Done-when block above.
- [ ] Task 6.4.1: full run executed start-to-finish through Streamlit including both gates.
      Depends on: 6.1.2
- [ ] Task 6.4.2: the same run inspected from the notebook against the same Postgres database.
      Depends on: 6.2.1
- [ ] Task 6.4.3: Langfuse trace for that run reachable from the Streamlit UI.
      Depends on: 6.3.1

---

## Suggested Build Order Summary

| Phase | Depends on | Primary risk being retired |
|---|---|---|
| 0 | — | Nothing exists yet; every later phase blocked without it — now includes standing up Docker Compose infra (Postgres + Langfuse) alongside the code scaffold |
| 1 | 0 | Source quality/coverage, and feasibility of two reputation signals (WHOIS, backlink proxy) |
| 2 | 0, 1 | Human-gate UX (interrupt/resume durability against Postgres) + clustering quality |
| 3 | 0–2 | Claim-cluster quality — foundation for bias/briefing |
| 4 | 0–3 | Bias/framing + briefing quality — highest value, least automatable validation |
| 5 | 0–4 | Cross-run identity/drift correctness, unattended scheduled-run reliability, gate policy for headless runs |
| 6 | 0–5 | Usability only — deliberately last, no new pipeline logic |

**Flagged open decisions to resolve before/during the relevant phase:**
- Phase 1: concrete backlink-proxy data source (static Tranco/Common Crawl snapshot, not a live API) — resolve before writing `reputation/signals.py`.
- Phase 5: gate policy for scheduled/unattended runs (auto-approve vs. fail-and-notify) — the TRD doesn't specify this; needs a decision, not just implementation.
- Future (not v1): `pgvector` for in-database embedding similarity search — a natural fit now that the app is on Postgres, but not adopted now; the plan keeps the original Python/numpy-side similarity computation to avoid re-architecting clustering/matching beyond what was asked.