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

**Status as of last update: Phase 0 code is complete and merged; one Done-when bullet is BLOCKED pending a human/UI step.** Stories 0.1–0.9 are implemented and merged to `master` (Story 0.7 via PR #6, Story 0.8 + Task 0.7.4 together via PR #7, branch `story/phase0-cli-dev-harness`). `acceptance-verifier` ran the Story 0.10 walkthrough independently (re-deriving from live repo/infra state, not trusting prior self-reports) and returned a **BLOCKED** verdict:
- `docker compose ps` — all 7 services (`postgres`, `langfuse-postgres`, `clickhouse`, `redis`, `minio`, `langfuse-web`, `langfuse-worker`) healthy. **MET.**
- `uv run newsresearch run "..."` exits 0 (re-confirmed independently by the verifier with its own fresh run). **MET.**
- `\dt` against the app Postgres shows all 9 TRD tables + `run_costs` + `schema_version` (`version=2`, per Story 0.7's migration) + the 4 LangGraph checkpoint tables. **MET.**
- `Settings()` loads cleanly from `.env.example`-shaped env vars. **MET.**
- `uv run ruff check .` and `uv run pytest` both pass clean on `master` (30 passed, 1 deselected live test). **MET.**
- The observability triad (one stub LLM call → `run_costs` row + Langfuse trace + MLflow run, all from the *same* run_id): **BLOCKED, not met.** Correction to an inaccurate claim in a prior version of this note: no single run_id has actually been shown to produce all three artifacts together. Every run made from this environment (mine and the verifier's) used an invalid/placeholder Langfuse secret key — no `.env` is committed and Postgres only stores hashed secrets, so no valid key is retrievable headlessly — meaning the Langfuse export correctly soft-fails (`401`, exit 0 anyway) but no trace is produced for those run_ids. Separately, two *real* Langfuse traces do exist in ClickHouse (from earlier agent runs made with a genuine, now-discarded disposable API key), but neither of those run_ids has a matching MLflow run, and their timestamps predate the final merged CLI wiring. The soft-fail behavior itself is correctly verified (twice, independently); the triad-in-one-run claim is not.
  **Action needed to close this out:** log into the Langfuse UI at `http://localhost:3000` (existing accounts: `eeshangpt+langfuse@gmail.com`, or the disposable `backend-agent@example.com`), generate a fresh API key pair, export `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, run `uv run newsresearch run "<topic>"` once, and confirm that one run_id has a `run_costs` row, a Langfuse trace, and an MLflow run all together. This is a manual/UI step no agent can do headlessly.

The CI-scoping decision below remains a separate, explicitly non-blocking open item, deferred to whenever `project-manager`/`tech-lead` picks it up.

Repo-state check confirmed: no `newsresearch/` package, no `docker-compose.yml`, no `deploy/`, no `tests/` — nothing below has been built yet. The numbered Tasks above are preserved as-is except where a task bundled more than one independently-verifiable outcome; those are split below and cross-referenced back to their original task number for traceability. Splits: original Task 3 (docker-compose.yml) → app-Postgres service and Langfuse stack are separately verifiable; original Task 9 (`llm/models.py`) → `get_chat_model` and `get_embeddings` are separately verifiable; original Task 10 (observability) → three distinct modules plus a separate wiring step.

**Ownership split (per `devops-engineer`'s own scope boundary):** Story 0.2 (`docker-compose.yml` + `deploy/langfuse/`) is `devops-engineer`-owned — infra/containers only, no application code. Every other story (0.1, 0.3–0.9) is `backend-engineer`-owned — this includes the observability *code* (`cost_callback.py`, `langfuse_setup.py`, `mlflow_setup.py`) even though it integrates with devops-provisioned infra; devops owns the containers those modules talk to, not the modules themselves. Several backend tasks have a **runtime** (not build-time) dependency on Story 0.2 actually being up (`docker compose up -d`) rather than merely written — called out explicitly at each task below, since `testcontainers`-backed unit/integration tests are deliberately hermetic and do NOT carry this dependency. CI (running `ruff`/`pytest` in a pipeline) is not itself a Phase 0 task in the list below — `devops-engineer`'s own scope doc flags CI as unscoped in this plan and asks `project-manager` to scope it explicitly rather than have it freelanced; this is an open item, not a silent gap, and is called out again at the end of this breakdown.

**Story 0.1 — Dependency baseline established** ✅ MERGED (`master`)
Acceptance: `uv sync` installs core + dev groups with no resolution errors.
- [x] Task 0.1.1 (orig. Task 1): `uv add langgraph langgraph-checkpoint-postgres langchain langchain-core langchain-openai langchain-huggingface pydantic pydantic-settings python-dotenv typer psycopg[binary] psycopg_pool mlflow langfuse`.
      Acceptance: `pyproject.toml`/`uv.lock` list every named package; `uv sync` exits 0; each imports cleanly in `.venv`.
      Depends on: none
- [x] Task 0.1.2 (orig. Task 2): `uv add --group dev pytest respx freezegun pytest-mock testcontainers[postgres] ruff`.
      Acceptance: dev group lists all named packages; `uv run pytest --version` and `uv run ruff --version` succeed.
      Depends on: none

**Story 0.2 — Local infra (Postgres + self-hosted Langfuse) via Docker Compose** ✅ MERGED (`master`)
Acceptance: `docker compose up -d` brings up app Postgres and the full Langfuse stack, UI reachable at `localhost:3000`.
- [x] Task 0.2.1 (orig. Task 3, split a): `docker-compose.yml` `postgres` service for the app DB (`newsresearch`).
      Acceptance: `docker compose up -d postgres` starts the container; `pg_isready` against it succeeds.
      Depends on: none
- [x] Task 0.2.2 (orig. Task 3, split b): `deploy/langfuse/` self-host stack (Postgres, ClickHouse, Redis, MinIO, `langfuse-web`, `langfuse-worker`), wired into root `docker-compose.yml`, isolated from the app's own Postgres service/volume.
      Acceptance: `docker compose up -d` brings up all Langfuse services; `http://localhost:3000` loads; `docker compose config` shows Langfuse's Postgres as a distinct service/volume from Task 0.2.1's.
      Depends on: none
- [x] Task 0.2.3 (orig. Task 3, doc note): document `docker compose up -d` as the required local-dev pre-step.
      Acceptance: README states the pre-step; `docker compose ps` on a fresh clone shows all services healthy.
      Depends on: 0.2.1, 0.2.2

**Story 0.3 — Config system (`Settings`) loads every NFR-5 tunable** ✅ MERGED (`master`)
Acceptance: `Settings()` instantiates from `.env.example`-shaped input and exposes every nested field named in Cross-Cutting Concerns.
- [x] Task 0.3.1 (orig. Task 4): `config.py` — nested `Settings(BaseSettings)` with every NFR-5 tunable stubbed in.
      Acceptance: unit test instantiates `Settings` from an in-memory env + optional `config.yaml`, asserts every named nested field is present and env vars override yaml.
      Depends on: none
- [x] Task 0.3.2 (orig. Task 12): `.env.example` — `OPENAI_API_KEY`, `NEWSRESEARCH_DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `MLFLOW_TRACKING_URI`.
      Acceptance: copying to `.env` and loading via `Settings()` succeeds with no missing-required-field error.
      Depends on: 0.3.1

**Story 0.4 — Postgres schema applies cleanly and idempotently** ✅ MERGED (`master`)
Acceptance: `init_db()` run twice produces every TRD table + `run_costs` + `schema_version`, second run is a no-op.
- [x] Task 0.4.1 (orig. Task 5): `persistence/schema.sql` — TRD section 5 DDL (`BYTEA` embeddings, `TIMESTAMPTZ` timestamps) + `run_costs` + `schema_version`, all `CREATE TABLE IF NOT EXISTS`.
      Acceptance: `psql -f persistence/schema.sql` against an empty DB creates every table with no errors; re-running is a no-op.
      Depends on: none
- [x] Task 0.4.2 (orig. Task 6): `persistence/db.py::init_db(database_url)` via `psycopg_pool`, idempotent schema application.
      Acceptance: `testcontainers`-backed test calls `init_db()`, asserts all tables + `schema_version` row exist; second call doesn't raise.
      Depends on: 0.4.1

**Story 0.5 — LangGraph skeleton compiles and executes as a no-op pipeline with durable checkpointing** ✅ MERGED (`master`)
Acceptance: full TRD node topology exists as passthrough nodes, compiled with `PostgresSaver`, `graph.invoke()` completes without error.
- [x] Task 0.5.1 (orig. Task 7): `graph/state.py` — top-level state + per-subtopic sub-state for `Send`-based fan-out.
      Acceptance: state schema importable; instance constructs with all named fields.
      Depends on: none
- [x] Task 0.5.2 (orig. Task 8): `graph/build.py` — wire full topology (Subtopic→Gate1→fan-out→Sourcing→Clustering→Gate2→Claims→Summarize→Bias→Briefing→Snapshot→Timeline) as no-op nodes, compiled with `PostgresSaver` (not in-memory).
      Acceptance: `graph.invoke(initial_state, config={"configurable": {"thread_id": "test"}})` runs through every node and returns; a checkpoint row exists in Postgres afterward.
      Depends on: 0.5.1, 0.4.2
      Runtime note (backend-engineer, cross-track): the Postgres row check in this acceptance criterion requires the real app Postgres from devops Story 0.2 Task 0.2.1 to be up via `docker compose up -d` — this is distinct from 0.4.2's `testcontainers` tests, which are hermetic and need nothing from Story 0.2.
      Verified: confirmed against the real running dev-compose Postgres (not just testcontainers) — 14 checkpoint rows observed via `psql` for the test thread.

**Story 0.6 — LLM layer factories exist and satisfy the LangChain interface contract** ✅ MERGED (`master`)
Acceptance: `get_chat_model(stage)` returns `BaseChatModel` for every `Settings.models.*` stage; `get_embeddings()` returns an `Embeddings`-interface object for both backends.
- [x] Task 0.6.1 (orig. Task 9, split a): `llm/models.py::get_chat_model(stage) -> BaseChatModel`, model name from `Settings.models.<stage>`, backed by `ChatOpenAI`.
      Acceptance: calling it per stage name returns a `BaseChatModel` constructed with the `Settings`-sourced model name (constructor-arg inspection, no network call in test).
      Depends on: 0.3.1
- [x] Task 0.6.2 (orig. Task 9, split b): `llm/models.py::get_embeddings() -> Embeddings` — `HuggingFaceEmbeddings` for `"local"` (default), `OpenAIEmbeddings` for `"openai"`, both `isinstance(..., langchain_core.embeddings.Embeddings)`.
      Acceptance: backend switch via `Settings.embeddings.backend` returns the correct implementation in each case.
      Depends on: 0.3.1
- [x] Task 0.6.3 (convention scaffold, not separately numbered above): `llm/prompts/` directory + `llm/schemas.py` stub, establishing the convention before Phase 1+ agents add real content.
      Acceptance: `llm/prompts/` exists with at least one example `.txt` template loadable via `ChatPromptTemplate.from_template()`; `llm/schemas.py` exists and imports cleanly.
      Depends on: none
      Note: added `sentence-transformers` as an explicit dependency beyond Story 0.1's original list — required at construction time by `HuggingFaceEmbeddings`, not pulled in transitively as assumed.

**Story 0.7 — Observability stack verified end-to-end through a stub LLM call** ✅ CODE MERGED (`master`, PR #6) — ⚠️ end-to-end acceptance not yet actually demonstrated (see Task 0.7.4)
Acceptance: one stub LLM call through the graph produces a `run_costs` row, a visible Langfuse trace, and an MLflow run — simultaneously, from one top-level `graph.invoke()`.
Note: this was originally built unrequested/out of sequence (ahead of Story 0.8, which Task 0.7.4 depends on); it was reviewed and merged as-is. `cost_callback.py`'s per-token pricing table still has real rates for only two models, placeholders otherwise — not yet trustworthy for real cost figures, tracked as a known gap rather than a blocker.
- [x] Task 0.7.1 (orig. Task 10, split a): `observability/cost_callback.py` — `BaseCallbackHandler` capturing `{run_id, stage, model, input_tokens, output_tokens, estimated_cost, latency_ms}`, writes to `run_costs`, fails soft independent of Langfuse reachability.
      Acceptance: unit test with a mocked LLM call asserts a fully-populated `run_costs` row is written; simulated DB-write failure logs-and-continues, never raises.
      Depends on: 0.4.2, 0.6.1
- [x] Task 0.7.2 (orig. Task 10, split b): `observability/langfuse_setup.py` — `CallbackHandler` factory, tags traces with `run_id` + `subtopic_id`.
      Acceptance: a stub call attached with this handler produces a trace visible at `localhost:3000` tagged with the given `run_id`.
      Depends on: 0.2.2, 0.3.2
- [x] Task 0.7.3 (orig. Task 10, split c): `observability/mlflow_setup.py` — run lifecycle helpers keyed by `run_id`, `./mlruns` file-store backend.
      Acceptance: start/end helpers around a stub invocation produce exactly one MLflow run under `./mlruns` tagged with `run_id`.
      Depends on: 0.3.2
- [x] Task 0.7.4 (orig. Task 10, wiring clause + Task 11's callback half): wire all three callbacks into `cli.py`'s top-level `graph.invoke(state, config={"callbacks": [...]})`.
      Acceptance: `uv run newsresearch run "test topic"` (with a real/stubbed chat-model call in one node) produces all three artifacts from Story 0.7's acceptance in one command.
      Depends on: 0.7.1, 0.7.2, 0.7.3, 0.8.1
      Status: code merged to `master` via PR #7 (branch `story/phase0-cli-dev-harness`), together with Story 0.8. **Acceptance criterion not yet actually demonstrated** — `acceptance-verifier` ran this independently and found that no single run_id has produced `run_costs` + Langfuse trace + MLflow run together: every headless run in this environment lacks a retrievable valid Langfuse secret key (soft-fails correctly, `401`, exit 0, but no trace), while the two real traces that do exist in ClickHouse (from an earlier disposable-key run) have no matching MLflow run and predate this PR's merge. The wiring code itself is verified correct (soft-fail behavior confirmed twice independently, `run_costs`+MLflow confirmed together from one run); only the three-artifacts-from-one-run acceptance check needs a human to close, via a real Langfuse API key generated through the UI. See the section-level status note above for the exact steps.

**Story 0.8 — CLI dev harness runs the compiled no-op graph end-to-end** ✅ MERGED (`master`, PR #7)
Acceptance: `uv run newsresearch run "test topic"` executes the full no-op graph and exits 0.
- [x] Task 0.8.1 (orig. Task 11, minus callback wiring which moved to 0.7.4): `cli.py` — `typer` app, `run <topic>` command invoking the compiled graph; `main.py` reduced to `from newsresearch.cli import app; app()`.
      Acceptance: `uv run newsresearch run "test topic"` invokes `graph.invoke()` on the Phase-0 no-op graph and exits 0.
      Depends on: 0.5.2
      Note: this task's implementation also touches the already-merged `graph/build.py` (Story 0.5) — the `subtopic` node became a small in-file stub `BaseChatModel` (not `llm.models.get_chat_model`, deliberately, so Phase 0 needs no real `OPENAI_API_KEY`) so there's an actual LLM call for Task 0.7.4's callbacks to capture. Flagged explicitly by the implementing agent rather than silently bundled; reviewed and merged as-is.

**Story 0.9 — Testing/lint baseline passes** ✅ VERIFIED (`master`)
Acceptance: `pytest` (testcontainers-backed) and `ruff check` both pass against everything built in Phase 0.
- [x] Task 0.9.1: baseline tests — `Settings` load/precedence, schema idempotency, cost-callback correctness, graph compile+invoke, CLI end-to-end.
      Acceptance: `uv run pytest` exits 0, using `testcontainers[postgres]` (no dependency on the dev compose stack being up).
      Depends on: 0.3.1, 0.4.1, 0.4.2, 0.7.1, 0.5.2
      Verified: 30 passed, 1 deselected (`@pytest.mark.live`) on `master` post-merge.
- [x] Task 0.9.2: `ruff` configuration + clean pass over `newsresearch/` as it stands.
      Acceptance: `uv run ruff check .` exits 0.
      Depends on: 0.1–0.8 code tasks
      Verified: clean on `master` post-merge.

**Story 0.10 — Phase 0 Done-when verified end-to-end (integration/demo)** ⚠️ BLOCKED — 5 of 6 bullets MET, 1 BLOCKED
Acceptance: all six bullets in the phase-level "Done when" block above hold simultaneously on a clean checkout.
- [ ] Task 0.10.1: manual end-to-end walkthrough of the exact Done-when command sequence.
      Acceptance: each bullet confirmed in order; any failure is filed against the specific task above, not against "Phase 0" generally.
      Depends on: 0.1–0.9 all complete
      Verdict (`acceptance-verifier`, run independently against live repo/infra state): docker-compose-healthy, CLI-exits-0, schema-completeness, `Settings()`-loads, and pytest+ruff all **MET**. The observability-triad-in-one-run bullet is **BLOCKED** — see Task 0.7.4's status note and the section-level status note above for the exact gap and the human action needed to close it. **Phase 0 is not fully done** until that one bullet is demonstrated; everything else genuinely is.

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

**Story 1.2 — GDELT DOC 2.0 client fetches and paginates beyond the 250-record cap** ✅ tech-lead approved, ready to merge (branch `task/gdelt-doc-client`)
Acceptance: a query expected to exceed 250 raw results returns a combined list larger than 250 via sub-window paging, with 429 backoff.
- [ ] Task 1.2.1 (orig. Task 2, part a): `sourcing/gdelt.py` — single-window keyword+date-range query, JSON mode.
      Acceptance: returns parsed article dicts (title, url, domain, published date) from a real GDELT call under 250 results.
      Depends on: Task 1.0.1 (httpx)
      **POST-MERGE DEFECT FOUND — FIXED (tech-lead, 2026-07-22, discovered during Story 1.10's required live-run review, not a Story 1.2 relitigation; fix landed and re-verified 2026-07-23, commit `36e75a0` on branch `task/dev-sourcing-cli`):** `_build_query()` unconditionally quotes every keyword and OR-joins them with no enclosing parens (`" OR ".join(f'"{keyword}"' for keyword in keywords)`). Verified directly against the real GDELT DOC 2.0 endpoint (bypassing `gdelt.py` entirely, to isolate this from the already-known rate-limiting issue): a single quoted keyword (`"Iran"`) gets a real HTTP 200 with a non-JSON body `"The specified phrase is too short."`; two-plus quoted keywords OR-joined without wrapping parens (`"Iran" OR "nuclear"`) gets a real HTTP 200 with a non-JSON body `"Queries containing OR'd terms must be surrounded by ()"`. Both are genuine GDELT DOC 2.0 query-grammar rejections, not rate-limiting, and `query_window`'s existing `except ValueError` (JSON-parse failure) correctly turns each into a `GDELTError` — which then propagates uncaught out of `sourcing_agent`, per Story 1.2's own approved raise-on-exhaustion design. This has been silently masked in every "verified against real GDELT" review to date (Story 1.9's live test, Task 1.11.2) because their only live coverage used a single-keyword list (`KEYWORDS = ["Iran"]`) wrapped in `except GDELTError: return []` — which cannot distinguish "genuinely rate-limited," "genuinely 0 JSON matches," and "query-grammar rejected" from each other; all three previously-reported "0 GDELT results" observations are equally consistent with this defect having fired instead of a real empty match. Not reopening those approvals (their RSS-side results were independently real and sufficient for what they were verifying), but flagging so the record isn't misread as "GDELT single/multi-keyword queries have been proven to work against the live API" — they haven't. **Required fix:** only quote a keyword when it contains whitespace (quoting exists to force phrase-matching for multi-word phrases; a lone token doesn't need it and GDELT rejects a quoted lone token as too short a "phrase"), and wrap the full OR-joined expression in `(...)` whenever there's more than one term. Re-verify against the real endpoint directly (a scratch script hitting `api.gdeltproject.org` is sufficient) before trusting a mocked/`respx`-based unit test again, since respx fixtures can't catch a query the real API itself rejects.
      **FIX CONFIRMED (tech-lead re-review, 2026-07-23):** `_build_query` now quotes a keyword only when it contains whitespace and wraps the OR-joined expression in `(...)` only when there's more than one term, matching the required fix exactly. Independently re-verified directly against the real GDELT DOC 2.0 endpoint (not the `respx`-mocked unit tests): `Iran`, `"climate change"`, `(Iran OR Russia)`, and `("climate change" OR wildfire)` each returned a real HTTP 200 + genuine parseable JSON body — no "phrase too short" or "must be surrounded by ()" rejection for any of the four single/multi, bare/phrase keyword-list shapes. `tests/test_gdelt.py` replaced the one stale case that encoded the original bug with three cases covering bare-single-token, single-phrase, and multi-term-wrapped behavior. The characterization above (that prior "verified against real GDELT" reviews using single-keyword `except GDELTError: return []` couldn't distinguish rate-limiting from query-grammar rejection from genuine zero-matches) still stands as historical record and is not retroactively changed by this fix — it explains why the defect went undetected for as long as it did.
- [ ] Task 1.2.2 (orig. Task 2, part b): pagination over sub-windows when expected volume exceeds 250, plus 429 backoff/retry.
      Acceptance: a lookback range known to exceed 250 raw hits returns >250 combined results via internal sub-window splitting; a simulated 429 triggers backoff-and-retry rather than immediate failure.
      Depends on: 1.2.1

**Story 1.3 — RSS sourcing and Google News backfill exist as structurally separate modules** ✅ tech-lead approved, ready to merge (branch `story/rss-google-news-backfill`)
Acceptance: `sourcing/rss.py` fetches trusted-feed articles by keyword/date; `sourcing/google_news_backfill.py` is independently callable and structurally isolated (NFR-3).
- [ ] Task 1.3.1 (orig. Task 3, part a): `sourcing/rss.py` — fetch official RSS feeds for trusted-tier outlets from `data/trusted_outlets.yaml`, filter by keyword/date.
      Acceptance: given a keyword + lookback window, returns article dicts from at least 2 real trusted-outlet RSS feeds, correctly filtered.
      Depends on: 1.1.1, Task 1.0.1 (feedparser)
- [ ] Task 1.3.2 (orig. Task 3, part b): `sourcing/google_news_backfill.py` — structurally separate module wrapping Google News RSS fetch.
      Acceptance: has its own public function callable independently of `rss.py`/`gdelt.py`; import-graph check confirms `gdelt.py`/`rss.py` never import from it (only the orchestrator does).
      Depends on: Task 1.0.1 (feedparser/httpx)

**Story 1.4 — Deduplication removes exact-URL and cross-source near-duplicate articles** ✅ tech-lead approved, ready to merge (branch `task/dedup-sourcing`)
Acceptance: `sourcing/dedup.py` collapses normalized-identical URLs and rapidfuzz-similar cross-domain titles (wire-story dupes).
- [x] Task 1.4.1 (orig. Task 4, part a): URL normalization + exact-match drop (strip tracking params, trailing slash, scheme-normalize).
      Acceptance: unit test with `http://x.com/a?utm_source=y` and `https://x.com/a/` confirms only one survives.
      Depends on: Task 1.0.1 (no new dep beyond stdlib/httpx already added)
- [x] Task 1.4.2 (orig. Task 4, part b): `rapidfuzz` title-similarity cross-source dedup.
      Acceptance: unit test with two near-identical titles from different domains above threshold confirms one is dropped; a below-threshold pair both survive.
      Depends on: 1.4.1, Task 1.0.1 (rapidfuzz)
      Reuses `Settings.clustering.similarity_threshold` (pre-provisioned for exactly this, per TRD.md line 88) rather than a new config field.

**Story 1.5 — GDELT/RSS presence-frequency reputation signal (build and trust first, no external dependency)** ✅ tech-lead approved, ready to merge (branch `task/gdelt-rss-presence-frequency`)
Acceptance: `reputation/signals.py`'s presence-frequency collector computes a per-domain frequency score from Phase 1's own fetch results, zero additional network calls.
- [ ] Task 1.5.1 (orig. Task 5): presence-frequency signal collector.
      Acceptance: given a fixed list of (domain, source_type) fetch results, returns a normalized per-domain frequency score deterministically; covered by a unit test with fixed input, no live call.
      Depends on: 1.2.2 (GDELT results shape), 1.3.1 (RSS results shape), 1.4.2 (operates on the deduped set, per stated task order)

**Story 1.6 — Domain-age, backlink-proxy, and HTTPS/about-page signals, each soft-failing independently** ✅ tech-lead approved, ready to merge (branch `story/domain-reputation-signals`)
Acceptance: three collectors in `reputation/signals.py` each return a normalized value or neutral/`None` on failure, never raising.

*FLAGGED FEASIBILITY RISKS (carried forward from EXECUTION_PLAN / PRD risk table, not footnotes):*
- **WHOIS rate-limiting**: WHOIS is frequently rate-limited/blocked for bulk automated lookups. The soft-fail path is the primary defense, not a corner case — Task 1.6.1 must not assume WHOIS reliability, and must cache aggressively (feeds Story 1.7's cache) so it's a rare call, not per-run.
- **Backlink-proxy data source — RESOLVED (`tech-lead`):** use the **Tranco list**, not Common Crawl host-rank. Reasoning: Tranco ships as a single canonical, already-`rank,domain`-shaped CSV download, whereas Common Crawl's host-graph product requires joining a separate vertex file against a harmonic-centrality/pagerank file per crawl snapshot with no fixed "top N" cut and no pre-trimmed small slice — an order of magnitude more preprocessing for a noisier signal. Both are free/permissively licensed (Tranco: CC BY/CC BY-SA, attribution required), satisfying the no-paid-data-license constraint.
  - **File**: commit the **top 100,000** rows (not the full 1M) at `newsresearch/data/tranco_top100k.csv` — plain `rank,domain` CSV, no header, Tranco's native format, no transformation needed beyond truncation. Full top-1M is ~22.7MB uncompressed (confirmed by direct download, not estimated); the top-100k slice is ~2.3MB, trivially committable/diffable, no Git LFS needed. Candidate news/outlet domains overwhelmingly fall within a global top-100k rank; the TRD 4.2 formula only needs enough resolution to distinguish "clearly-established" from "not clearly established," not long-tail precision. Add a one-line attribution note (source URL + license) alongside the file, e.g. a sibling `newsresearch/data/tranco_top100k.csv.README`.
  - **Freshness**: this is a frozen artifact, refreshed manually/occasionally by re-running the truncation step against a fresh Tranco download and committing the update — a separate axis from `Settings.reputation.staleness_days`, which governs per-domain *cached score* recomputation, not the snapshot file's own vintage. Task 1.6.2's implementer should note this distinction in `reputation/signals.py`'s docstring so the two aren't conflated.
  - **Normalization**: continuous log-scale decay, not discrete tier buckets (buckets create artificial score cliffs at boundaries): `normalized = clip(1 - log10(rank) / log10(100_000), 0.0, 1.0)` for a domain present in the snapshot (rank 1 → 1.0, rank 10 → 0.8, rank 1,000 → 0.4, rank 100,000 → 0.0). **Neutral value 0.5 (not 0) if the domain is absent from the snapshot** — absence from a global top-100k popularity list is expected/common for legitimate small/regional/niche outlets and shouldn't read as evidence of illegitimacy; this also matches the fail-open convention already used by the other two Story 1.6 signals (WHOIS, HTTPS/about-page), so `reputation/scorer.py` doesn't need per-signal special-casing. If Task 1.11.1's good-vs-bad empirical spot-check later shows this doesn't discriminate well enough, the documented fallback is discrete tiers (top-1k=1.0, top-10k=0.8, top-100k=0.5, absent=0.5) — file that as a tuning follow-up against this specific signal, not against "Phase 1" generally, only if an inversion actually shows up.
  - **Naming caveat**: Tranco is a popularity/traffic rank, not a literal backlink-graph rank. TRD 4.2's formula variable is named `backlink_proxy` — keep that field/variable name as-is (lower churn, `Settings`/scorer already assume it) but add a one-line doc comment clarifying "Tranco combined popularity rank, used as a backlink/authority proxy" so it isn't later "fixed" to expect literal backlink counts.

- [ ] Task 1.6.1 (orig. Task 6, part a): WHOIS domain-age collector (`python-whois`), fail-soft, cache-friendly.
      Acceptance: unit test with mocked `python-whois` raising/timing out confirms the collector returns `None`, never an exception; a live/manual spot-check against 2–3 real domains returns a plausible age.
      Depends on: Task 1.0.1 (python-whois)
- [ ] Task 1.6.2 (orig. Task 6, part b): backlink-proxy signal from the Tranco top-100k snapshot at `newsresearch/data/tranco_top100k.csv` (decision recorded above) — not a live network call.
      Acceptance: collector loads the committed CSV, returns `clip(1 - log10(rank)/log10(100_000), 0, 1)` for a present domain, `0.5` neutral for an absent one; unit test covers a known-high-rank domain, a known-absent domain, and the log-scale formula's exact output for a fixed rank.
      Depends on: none beyond the resolved decision above (no code dependency on Task 1.0.1 — CSV loading needs no new library)
- [ ] Task 1.6.3 (orig. Task 6, part c): HTTPS + about-page heuristic via HEAD/GET (`httpx`).
      Acceptance: unit test mocks an `httpx` response confirming HTTPS presence + about-page hit both contribute a legitimacy flag; a timeout/error case fails soft (flag = neutral, no raise).
      Depends on: Task 1.0.1 (httpx)

**Story 1.7 — Reputation scoring formula + `domain_reputation` cache implement TRD 4.2 exactly** ✅ tech-lead approved, ready to merge (branch `story/reputation-scoring-cache`)
Acceptance: `reputation/scorer.py` computes the TRD 4.2 formula with `Settings`-sourced weights, clipped to a bounded adjustment range; `reputation/cache.py` respects `Settings.reputation.staleness_days`.
- [ ] Task 1.7.1 (orig. Task 7, part a): `reputation/scorer.py` — scoring formula, weights from `Settings.reputation`, bounded/clipped adjustment (not summed unbounded).
      Acceptance: unit test with fixed signal inputs + known weights asserts exact formula output; an adversarial extreme-signal combination is confirmed clipped rather than pushing score outside the configured bound.
      Depends on: 1.5.1, 1.6.1, 1.6.2, 1.6.3, Phase 0 Task 0.3.1 (`Settings.reputation.*` weights)
      **Integration checkpoint (tech-lead review, Task 1.5.1):** `get_presence_frequency_scores` deliberately normalizes each domain's distinct-source-type count against the *batch's own* distinct-source-type count, not a fixed universe of 3 (`gdelt`/`rss`/`google_news_backfill`) — reasoned and correct for a single batch (see its docstring), but `domain_reputation` is a cross-run persisted cache (TRD 4.2/4.6: "one row per domain, independent of any single run"), and whether Google News backfill fired in the specific batch that happens to trigger a given domain's staleness-triggered recompute is incidental to that domain's legitimacy (it depends on Story 1.8's primary-article-count threshold for *that day's* subtopic query, not on the domain itself). Concretely: the same domain, with identical GDELT+RSS coverage, can score 1.0 in a batch where backfill never fired and ~0.67 in a batch where backfill fired for unrelated reasons — a scale shift in a cached, long-lived signal caused by batch composition rather than domain quality. Task 1.7.1's implementer must make an explicit choice here (don't let it pass through unexamined): (a) fix the denominator to the known 3-source-type universe so the same coverage always maps to the same value regardless of whether backfill fired, accepting that non-backfill batches structurally can't reach 1.0 on this one signal, or (b) keep the batch-relative denominator and document in `scorer.py` why the resulting run-to-run noise on this one weighted term (`w3`, bounded per the TRD 4.2 formula's ±0.3 adjustment range) is acceptable. Either is defensible; leaving it undecided is not — record whichever is chosen in `scorer.py`'s docstring so it isn't relitigated per-recompute.
      **RESOLVED (implementer's choice, tech-lead verified 2026-07-22):** kept the batch-relative denominator (option b), documented in `reputation/scorer.py`'s module docstring. Tech-lead verified the stated magnitude arithmetically against the actual configured defaults (`weight_presence_frequency=0.05`, `adjustment_bound=0.3`): worst-case swing (batch-relative 1.0 vs. 0.67 for the same domain) costs `0.05 * 0.33 ≈ 0.017` final-score points, ~5.7% of the ±0.3 bound — small and acceptable, as claimed. One wording nit (non-blocking): the docstring's "it self-corrects via `staleness_days`" overstates what actually happens — a stale row's next recompute lands in *some* batch composition, not necessarily a more "correct" one, so it bounds the *duration* of any single noisy value rather than correcting it toward truth. Re-word if this file is touched again; not worth a fix-only PR. If Task 1.11.1's spot-check later surfaces an actual threshold-flip, the fix belongs in `reputation/signals.py::get_presence_frequency_scores`'s denominator per option (a) above, filed against Task 1.5.1/1.6.x, not against this scorer.
- [ ] Task 1.7.2 (orig. Task 7, part b): `reputation/cache.py` — `domain_reputation` table read/write, staleness-window check, manual-invalidation function.
      Acceptance: `testcontainers`-backed integration test writes a score, reads it back within the staleness window (cache hit, no recompute), confirms manual invalidation forces recompute; a `freezegun`-controlled clock verifies the staleness-window boundary precisely.
      **Known gotcha (found during Story 1.3, verified by tech-lead 2026-07-22):** `freeze_time` triggers a fresh lazy import of `langfuse.api`'s Pydantic models while time is frozen, and pydantic-core's schema generation chokes on it — confirmed to reproduce only when the full suite runs (langfuse must already be loaded by an earlier test, e.g. `test_config.py`/`test_langfuse_setup.py`, before `freeze_time` does its module-attribute scan); running `tests/test_rss.py` alone does not trigger it, which is why this is easy to miss in isolation. Any test using `freeze_time` alongside langfuse-touching code needs `freeze_time(..., ignore=["langfuse"])`. Since Story 1.3 and this task are now two independent occurrences of the same paper cut, and `tests/conftest.py` does not yet exist: **Task 1.7.2's implementer should add one** with an autouse fixture (or `freezegun.config.configure(default_ignore_list=["langfuse"])` at collection time) so future `freeze_time` users don't each have to rediscover and repeat the `ignore=` kwarg.
      Depends on: Phase 0 Task 0.4.2 (`init_db`/schema for the `domain_reputation` table), 1.7.1

**Story 1.8 — Backfill trigger logic invokes Google News RSS only as a non-load-bearing fallback** ✅ tech-lead approved, ready to merge (branch `task/backfill-trigger-logic`)
Acceptance: Google News RSS is invoked only when primary (GDELT+RSS) results fall below `Settings.sourcing.min_primary_article_count`; a simulated backfill failure never raises out of the sourcing path (NFR-3).
- [ ] Task 1.8.1 (orig. Task 8): trigger-threshold logic + try/except wrap that logs-and-continues, never raises.
      Acceptance: unit test with primary-count below threshold confirms backfill is invoked; at/above threshold confirms it is NOT invoked; a test forcing `google_news_backfill` to raise confirms the sourcing path logs and continues with primary-only results.
      Depends on: 1.3.2, Phase 0 Task 0.3.1 (`Settings.sourcing.min_primary_article_count`)

**Story 1.9 — Sourcing Agent orchestrates the full pipeline end-to-end** ✅ tech-lead approved, ready to merge (branch `task/sourcing-agent-orchestrator`)
Acceptance: `agents/sourcing_agent.py(keywords: list[str], lookback_days: int) -> list[ScoredArticle]` — given a hardcoded sample subtopic, returns a deduplicated, reputation-scored, threshold-filtered list from real GDELT/RSS calls.
- [ ] Task 1.9.1 (orig. Task 9): wire `agents/sourcing_agent.py`: query → backfill-if-below-min-count → dedup → score → filter by `Settings.reputation.min_score_threshold`.
      Acceptance: calling it against real GDELT/RSS returns `ScoredArticle` objects each above `Settings.reputation.min_score_threshold`, no duplicate URLs/near-dupe titles present. This is the Phase 1 phase-level Done-when target itself.
      Depends on: 1.2.2, 1.3.1, 1.3.2, 1.4.2, 1.7.1, 1.7.2, 1.8.1
      **Integration checkpoint (tech-lead review, Task 1.4.2) — RESOLVED (tech-lead verified 2026-07-22):** `dedup.py`'s cross-source-only guard compares `article["domain"]` by plain string equality; confirm at wiring time that `gdelt.py` (1.2.x) and `rss.py`/`google_news_backfill.py` (1.3.x) emit `domain` in a consistent normalized form (lowercase, no `www.` prefix) for the same outlet — otherwise the same-domain exemption can silently misfire in either direction. Fix at the article-dict-construction site in the affected sourcing client, not by adding normalization logic to `dedup.py` (keep dedup's input contract a plain string-equality comparison). Also worth an empirical look once real titles flow through: `dedup_by_title_similarity` uses `rapidfuzz.fuzz.ratio` (character-edit-distance); real cross-outlet wire-story headlines often reorder clauses more than they edit characters, so `fuzz.token_sort_ratio`/`token_set_ratio` may catch more true dupes if 1.11.1's spot-check shows under-matching — file that as a tuning follow-up against Task 1.4.2 specifically, not against "Phase 1" generally, same convention as the Tranco signal note above.
      Confirmed: `gdelt.py` lowercases+strips at construction (already correct); `rss.py`'s `domain` values come straight from `OUTLET_RSS_FEEDS`'s dict keys, which are hand-authored already-lowercase/no-`www.` (already correct); `google_news_backfill.py`'s `_domain_from_url` was the one gap — it stripped `www.` but never lowercased the resolved host, so a mixed-case Google News `<source>` href for an outlet also seen via GDELT/RSS (e.g. `WWW.Reuters.COM`) could defeat `dedup.py`'s same-domain string-equality guard in either direction. This is a real, previously-latent bug (not a pretext to touch the file), correctly fixed at the sourcing-client construction site per this checkpoint's own instruction (not by adding normalization logic to `dedup.py`), and covered by a direct regression test (`test_domain_from_url_lowercases_mixed_case_host`).
      **Orchestration wiring — verified correct:** `sourcing_agent()` calls `gdelt.fetch` + `rss.fetch_trusted_rss` → `backfill_trigger.maybe_backfill` → `dedup.dedup` → cache-first reputation scoring (`cache.get_fresh_domain_reputation`, falling back to `signals.*` + `scorer.score_domain` + `cache.write_domain_reputation` on miss) → filter by `Settings.reputation.min_score_threshold`, in that order, and every call site matches the real signature of the already-merged module it calls (checked against source, not assumed). No parallel config/LLM-call/cost-logging path introduced; correctly makes zero LLM calls (Phase 1 has no agentic step) and touches no persistence beyond the already-approved `domain_reputation` cache. No `cli.py` scope creep (that's Story 1.10). The 6 unit tests in `tests/test_sourcing_agent.py` (hermetic `testcontainers` Postgres + mocked network collectors) have real, specific assertions — not "doesn't raise" — covering dedup (near-dupe AP article and the unknown-tier domain both correctly drop, only the wire-tier survivor remains), threshold-filtering, cache-hit-skips-recompute (asserts the three signal collectors are *not* called and the returned score is exactly the pre-written cached value), backfill-triggering (`assert_called_once` + backfilled domain present in the result), and NFR-3 backfill-failure resilience (backfill raises, result still returns the primary-source article).
      **REQUIRED FIX before merge:** `tests/live/test_sourcing_agent_live.py`'s assertions all hold vacuously on an empty result list (`for item in result: ...` and the dedup-uniqueness checks pass trivially when `result == []`) — it never asserts `len(result) > 0`. The implementer's one live run returned zero `ScoredArticle` objects, and as currently written the test would have passed identically whether that was "everything correctly scored below threshold" or "the pipeline is broken and returns nothing" — it demonstrates the code runs without crashing, not the stated acceptance criterion ("returns `ScoredArticle` objects... above threshold"). This is a real gap against this repo's own precedent: `tests/live/test_rss_live.py` deliberately picks a broad, near-certain-to-match keyword (`"the"`) specifically "so this doesn't flake on a quiet news day" and asserts `len(articles) > 0` — the sourcing-agent live test should follow the same convention for at least its RSS leg (its `KEYWORDS = ["artificial intelligence regulation"]` was tuned only for GDELT's single-window-cap safety per `test_gdelt_live.py`'s precedent, not for guaranteeing an RSS hit). Add an explicit non-emptiness assertion and a keyword choice that reliably clears it.
      Tech-lead independently re-ran the live path to check whether this is achievable at all before requiring it: real GDELT genuinely exhausted its 5-retry/155s backoff budget with real HTTP 429s in this sandbox on a narrow single-keyword 3-day window (`GDELTError` propagated, consistent with `gdelt.py`'s own documented rate-limiting behavior — an environment condition affecting GDELT specifically, not a Story 1.9 defect). With `gdelt.fetch` mocked to `[]` and keyword `"Iran"` (confirmed via direct feed inspection to overlap that day's actual BBC/Guardian/NPR/Al Jazeera headlines), the real remaining pipeline (real trusted-RSS fetch, real WHOIS/HTTPS signal collection, real reputation-cache write, real dedup, real threshold filter) returned 14 correctly-deduped, "major"-tier, threshold-clearing `ScoredArticle` objects (scores 0.929–0.955) with no duplicate URLs or (domain, title) pairs. This confirms the pipeline itself is correct and the acceptance criterion is genuinely achievable — the required fix is to the test's rigor, not the implementation.
      **Non-blocking forward note (file against Phase 5 / Story 5.5's NFR-3 re-verification, not "Phase 1" generally):** unlike `google_news_backfill.py`'s failure (wrapped by `backfill_trigger.maybe_backfill`'s soft-fail), a real GDELT retry-exhaustion (`GDELTError`) propagates uncaught out of `sourcing_agent` — and does so *before* `rss.fetch_trusted_rss` even runs, discarding any RSS results that would otherwise have succeeded independently. This matches GDELT's already-approved Story 1.2 design (raise-on-exhaustion) and sits outside NFR-3's literal scope (which names the *backfill-only* source, not GDELT, as what must degrade gracefully), so it is not a Story 1.9 blocker and not something this review reopens. But this review's own live run reproduced a real GDELT 429-exhaustion on a narrow, single-keyword, 3-day window — stronger evidence than a hypothetical that this will recur under a scheduled/unattended run. Worth Phase 5 explicitly deciding whether a scheduled run should tolerate a GDELT outage by degrading to RSS(+backfill-trigger)-only rather than failing the whole run, rather than discovering this the first time a scheduled run happens to land during a GDELT rate-limit window.
      **REQUIRED FIX — RESOLVED (tech-lead verified 2026-07-22, commit `478082e`):** the live test now asserts `len(result) > 0` with an explicit message tying it to the acceptance criterion, switched keyword to `"Iran"` (a broad sustained-topic choice matching `test_rss_live.py`'s own "the"-for-reliability precedent, confirmed via direct feed inspection to overlap real BBC/Guardian/NPR/Al Jazeera headlines), and added assertions that at least one surviving result is wire/major tier. GDELT is still called for real (not mocked) via `_bounded_real_gdelt_fetch`, a thin wrapper around the real `gdelt.fetch` with a smaller test-local retry budget (`max_retries=2, backoff_seconds=5.0`) that swallows only a genuine `GDELTError` (rate-limit exhaustion) to `[]` — a narrowly-scoped accommodation for this sandbox's observed GDELT throttling, not a canned/mocked response and not a silent regression of Story 1.2's raise-on-exhaustion design (the real design is unchanged; only this one test's tolerance for that already-known condition is scoped down). Independently re-ran the live test: real GDELT was genuinely attempted (two real HTTP 429s, backoff, then a real 200 that returned 0 matching articles for the query/window — not a mock), and 15 real RSS articles from BBC/Guardian/NPR/Al Jazeera survived dedup and threshold filtering (`0.929`–`0.955`-range scores), correctly deduped (no duplicate URLs or (domain, title) pairs) and clearing the wire/major-tier assertion. Full suite: 127 passed, 6 deselected; `ruff check .` clean. No regressions. Approved.

**Story 1.10 — Manual dev CLI for sourcing inspection** ✅ tech-lead approved, ready to merge (branch `task/dev-sourcing-cli`)
Acceptance: `newsresearch dev sourcing-test "<keywords>"` runs the sourcing agent and prints a human-readable article+score listing.
- [ ] Task 1.10.1 (orig. Task 10): add `dev sourcing-test` typer subcommand.
      Acceptance: `uv run newsresearch dev sourcing-test "<keywords>"` prints each returned article's URL, domain, and reputation score to stdout; exits 0.
      Depends on: 1.9.1, Phase 0 Task 0.8.1 (typer CLI app must exist to extend)
      **✅ tech-lead approved, ready to merge (branch `task/dev-sourcing-cli`)** (2026-07-23, commit `36e75a0`, supersedes the "approve-with-required-fixes" verdict below — both required fixes confirmed landed and re-verified live.)
      **Structurally sound:** `dev sourcing-test` follows `run`'s established conventions exactly — `Settings()` construction, `database_url` guard via `typer.BadParameter`, pool via `persistence/db.py::init_db`, and is in fact more careful than `run` about pool lifecycle (`run` never closes its pool; this command does, in a `finally`). The call site (`sourcing_agent(keywords.split(), lookback_days, pool=pool, settings=settings)`) matches `sourcing_agent`'s real, already-merged signature exactly. Naive whitespace keyword-splitting and a `--lookback-days` default of 7 are reasonable and honestly documented in `--help`; `lookback_days` isn't a `Settings`-managed constant elsewhere in the codebase (it's a required positional arg on `sourcing_agent` itself), so defaulting it in the CLI is not a violation of the "never hardcode a configurable constant" cross-cutting rule. The `dev` sub-app (`app.add_typer(dev_app, name="dev")`) is clean, idiomatic Typer with no bleed into `run`'s code path. `tests/test_cli.py`'s new tests are correctly scoped to what's actually CLI-layer (arg parsing/splitting, pool guard, output formatting, empty-result message) via a hermetic mock of `sourcing_agent` itself — re-testing `sourcing_agent`'s own internals here would duplicate `tests/test_sourcing_agent.py`, not add value.
      **Required fix 1 (blocking, scoped to `cli.py` only — not `sourcing_agent`):** independently ran `uv run newsresearch dev sourcing-test "Iran"` twice, live, in this branch's own checkout against the real dev Postgres + real GDELT/RSS (per this review's mandate — the implementer's own tests never did this). Both runs crashed with a raw traceback and a **confirmed non-zero exit code (1)**, never printing readable output or a clean "no results" message: run 1 hit the `_build_query` grammar defect above (real HTTP 200, non-JSON rejection body, raised as `GDELTError`); run 2 hit a genuine GDELT rate-limit exhaustion (5 real retries, real 429s, ~155s backoff, raised as `GDELTError`). `sourcing_test` has zero exception handling around the `sourcing_agent(...)` call, so any real GDELT failure — a known, already-accepted characteristic of `sourcing_agent`'s approved design per Story 1.9's forward note — surfaces as an unreadable crash rather than the "exits 0" this story's own acceptance criterion requires. Fix: a narrow `try/except GDELTError` around the `sourcing_agent(...)` call in `cli.py::sourcing_test` that prints a short, readable diagnostic and exits via `raise typer.Exit(code=1)` (or 0, implementer's call, as long as it isn't a bare traceback) — this is dev-UX polish for a manual inspection tool, not a change to `sourcing_agent`'s own soft-fail/hard-fail contract (GDELT's raise-on-exhaustion stays exactly as approved in Story 1.2/1.9; NFR-3 only ever covered the backfill source).
      **Required fix 2 (blocking, filed against Task 1.2.1, not this diff):** see the `_build_query` grammar defect documented above — the CLI is the first real caller to expose it because it's the first thing that forwards free-form, real single/multi-keyword input straight to `sourcing_agent` → `gdelt.fetch` without a test-only `GDELTError`-swallowing wrapper. This needs to land before re-verifying Story 1.10's live acceptance, since right now essentially any realistic keyword input trips it independent of rate-limiting.
      **Next step:** once both are fixed, re-run `uv run newsresearch dev sourcing-test "<keywords>"` live against real GDELT/RSS in this same worktree and confirm readable per-article output (or the clean empty-result message) and exit 0 before marking this "ready to merge."
      **RE-REVIEW (tech-lead, 2026-07-23, commit `36e75a0`): both required fixes confirmed, approved.** `_build_query` now quotes a keyword only when it contains whitespace and wraps the OR-joined expression in `(...)` only when there's more than one term (verified correct by direct code inspection for all five cases: zero keywords still raises `ValueError`; a bare single-word keyword returns unquoted/unwrapped; a single multi-word phrase returns quoted, unwrapped; multiple single-word keywords and mixed multi-word/single-word keywords both return `(...)`-wrapped OR joins). Independently re-verified directly against the real GDELT DOC 2.0 endpoint (not mocked): `Iran`, `"climate change"`, `(Iran OR Russia)`, and `("climate change" OR wildfire)` each got a real HTTP 200 + genuine parseable JSON body (5 articles each) — the "phrase too short" / "must be surrounded by ()" rejections are gone. `cli.py::sourcing_test` now wraps the `sourcing_agent(...)` call in `try/except GDELTError`, printing a readable one-line diagnostic to stderr and exiting via `raise typer.Exit(code=1) from exc` instead of a raw traceback; `sourcing_agent`/`gdelt.py`'s own raise-on-exhaustion design (GDELT is a primary source, outside NFR-3's literal backfill-only scope) is unchanged — this is CLI-layer presentation only. `tests/test_gdelt.py` now has three `_build_query` cases (single bare token, single multi-word phrase, multi-term OR-wrap) replacing the one stale case that encoded the bug; `tests/test_cli.py` adds a `GDELTError`-side-effect case asserting non-zero exit, the diagnostic message, and no `Traceback` in output, using the real `GDELTError` type via a mocked `sourcing_agent`. Full suite: 134 passed, 6 deselected; `ruff check .` clean. No scratch/debug files in the diff (only the four expected files touched). Approved as-is, ready to merge.

**Story 1.11 — Phase 1 Done-when verified end-to-end (integration/demo)**
Acceptance: matches the phase-level Done-when block above.
- [x] Task 1.11.1: manual spot-check pass — run `sourcing_agent` against 3–5 known-good domains (Reuters, AP, BBC, etc.) and 3–5 known-bad/low-quality domains.
      Acceptance: a recorded good-vs-bad score comparison shows good domains consistently scoring higher; any inversion is filed as a defect against Task 1.7.1 (formula) or the relevant Task 1.6.x (signal), not against "Phase 1" generally.
      Depends on: 1.9.1
      **PASS (acceptance-verifier):** good domains 0.904–0.992 (reuters.com, apnews.com, bbc.com, theguardian.com, npr.org — all wire/major tier), bad domains 0.464–0.492 (5 known-low-quality domains, all unknown-tier). Clean separation, no inversions; driven mostly by `base_tier_score`. Non-blocking follow-up filed: reuters.com's HTTPS/about-page check (Task 1.6.3) got a false `https_present=False` because Reuters returns HTTP 401 specifically on `HEAD` requests (anti-bot behavior) — didn't affect ordering here since base tier dominates, but worth a future fix (fall back to GET, or treat non-2xx-non-error as neutral rather than False).
- [x] Task 1.11.2: backfill-failure resilience check — force/block `google_news_backfill` and confirm `sourcing_agent` still returns primary-source results without crashing.
      Acceptance: with backfill forced to raise/timeout, `sourcing_agent(...)` still returns a non-crashing result set built from GDELT+RSS alone — directly re-verifies NFR-3/Task 1.8.1 under real (not just unit-mocked) conditions.
      Depends on: 1.9.1, 1.8.1
      **CLEAN PASS, caveat resolved (re-verified 2026-07-23, post GDELT query-builder fix):** re-run with `min_primary_article_count` forced high (100000) to guarantee backfill triggering, keywords `["glacier", "permafrost"]`, `fetch_google_news_backfill` patched at its real import site (`newsresearch.sourcing.backfill_trigger.fetch_google_news_backfill`) to raise `RuntimeError`. Result: `BACKFILL_CALLED: True`, `sourcing_agent(...)` returned **67 real `ScoredArticle`s, all `source_type: "gdelt"`** (`BY_SOURCE_TYPE: {"gdelt": 67}`) — no crash, no stub. This supersedes the prior "stubbed GDELT" caveat: GDELT's own contribution is now confirmed live and non-empty in a genuine backfill-failure scenario, closing out NFR-3/Task 1.8.1 resilience under fully real conditions with no shortcuts remaining. **Phase 1 is now completely done.**

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
- ~~Phase 1: concrete backlink-proxy data source~~ **RESOLVED**: Tranco top-100k snapshot at `newsresearch/data/tranco_top100k.csv`, log-scale normalization, neutral 0.5 on absence — see Story 1.6's decision note.
- Phase 5: gate policy for scheduled/unattended runs (auto-approve vs. fail-and-notify) — the TRD doesn't specify this; needs a decision, not just implementation.
- Future (not v1): `pgvector` for in-database embedding similarity search — a natural fit now that the app is on Postgres, but not adopted now; the plan keeps the original Python/numpy-side similarity computation to avoid re-architecting clustering/matching beyond what was asked.