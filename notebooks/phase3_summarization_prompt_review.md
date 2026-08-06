# Task 3.6.1a — summarization prompt design + review

Owner: `data-scientist`. Design/analysis deliverable, not production code.
Handoff target: `backend-engineer`, Task 3.6.1b (wire via
`get_chat_model("summarization").invoke(...)` over `ChatPromptTemplate.
from_template()`, same pattern as `agents/claim_extraction_agent.py` and
`agents/subtopic_agent.py`).

Prompt: `newsresearch/llm/prompts/summarization.txt` (committed, ready to
lift as-is). No new pydantic schema — TRD §4.5's contract is "a concise
summary per cluster" (a string); `claim_clusters.summary` (schema.sql) is
already a plain `TEXT` column, and Phase 4's Bias & Framing agent input is
specified as "claim clusters + source metadata" (TRD §4.6), not the
summary — so nothing downstream consumes summarization output as
structured data. Adding a schema here would be an unrequested abstraction
for a contract that's a string end to end.

Script: `notebooks/phase3_summarization_review.py`. Real output:
`notebooks/phase3_summarization_samples.json` (9 real clusters, full
claim-cluster + generated summary).

**Real model calls against real data, one pass each, manually read — not a
scored/golden-dataset eval.** PRD §7 names that gap explicitly for v1;
judgments below are a manual read of real GPT-4.1-mini output against real
claim clusters, not a measured metric.

## Data

Reused Task 3.3.1a's real corpus (`tests/fixtures/phase3_claim_clustering_
corpus.json`, 6 real articles / 120 real claims on one real subtopic — a
DeepSeek/Qwen/Kimi AI-model-cost story) and ran `clustering/claim_
clustering.py::cluster_claims()` directly (the actual function `agents/
claim_extraction_agent.py` → `persistence/claim_clusters.py` production
path calls) to get the same 35 real clusters Task 3.3.1a's sweep
documented. No Postgres instance was up locally in this session (`docker
compose ps` empty) so no persisted `claim_clusters` rows existed to pull —
used the fixture + `cluster_claims()` directly instead, per this task's
stated fallback. No article full text was read, stored, or written
anywhere in this process — the fixture and every artifact here carry
`claim_text` + `domain` only, same no-full-text-storage scope the fixture
already had.

Same single-subtopic limitation Task 3.3.1a flagged applies here: one
real story shape (a wire-syndicated cost/benchmark story with heavy
cross-outlet duplication), not a representative sample across subtopic
types. In particular this fixture has **no `disputed`/`alleged`-certainty
claims** (all 120 claims are `confirmed`/`developing`, mostly `reported`/
`attributed`) — rule 6 below (preserve certainty/dispute language) is
untested against real contested claims; flagged, not silently assumed to
work.

## Design decision: sample against real false-merge clusters, not just clean ones

Task 3.3.1a's own finding is the load-bearing fact for this prompt:
**false-merge precision at the recommended clustering setting is 0.674 —
roughly 1 in 3 claims sharing a predicted cluster are two distinct facts
wrongly merged**, not the same fact restated. A summarization prompt that
only assumes "these claims are paraphrases of one fact, write one sentence"
would systematically misrepresent (or silently drop) claims in that ~1/3 —
a direct faithfulness violation of this task's own acceptance line ("no
invented or dropped claims"). So the 9 real clusters sampled for review
were deliberately chosen to cover the real cluster shapes this fixture
actually contains, not just the easy case:

- **Clean multi-source same-fact paraphrase** (cluster 21: 3 outlets, one
  wire-rewritten fact).
- **Shared fact + one genuinely extra fact merged in** (cluster 31: 3
  outlets restate one price figure, a 4th claim from one of those same
  outlets adds a distinct "3 cents per test" estimate — a partial
  false-merge).
- **Genuine false-merge, two distinct numbers, one source** (cluster 2:
  "80% price cut for Luna" and "20% price reduction for Terra" — different
  numbers, different products, embedded together by wording similarity).
- **Single-source, 5 genuinely distinct facts** (cluster 9: one article's
  five different benchmark scores, clustered together — HDBSCAN's
  `min_cluster_size=2` tuned for cross-source pairs, not within-article
  multi-fact runs, a related but distinct false-merge shape).
- **Single-source, 2 related-but-distinct claims** (clusters 34, 0).
- **3 single-assertion sources, each contributing one distinct fact**
  (cluster 15).
- **Single-source with a named-third-party quote mixed with the outlet's
  own framing** (cluster 1: one claim is the outlet's own reporting, one
  is a named individual's stated opinion).

## Prompt (iteration 1 — no iteration 2 needed)

`newsresearch/llm/prompts/summarization.txt`. Key rules, each tied to a
concrete real failure mode this cluster corpus actually presents:

1. **Grounding**: only use information stated in the listed claims, never
   add outside facts/interpretation.
2. **Unify true paraphrases**: if all claims restate one fact
   (cross-outlet wire-rewrite), write one sentence, not once per source.
3. **Never silently drop a distinct fact**: if the cluster actually
   contains 2+ genuinely different facts (Task 3.3.1a's ~1/3 false-merge
   rate makes this a real, expected case, not an edge case), cover every
   one, using multiple sentences if needed — directly targets this task's
   "nothing silently omitted" requirement.
4. **Never blend distinct facts into one vaguer merged claim** — e.g.
   never average/round two different numbers for two different subjects
   into one approximate statement. Directly targets cluster-2's real
   failure shape.
5. **Calibrate source-count language to what's actually listed** — never
   write "multiple outlets report" for a fact only one listed source
   states. This matters because the *cluster itself* already conflates
   "N claims" with "N sources" in the false-merge cases (e.g. cluster 9 is
   5 claims from 1 source) — the summary must not compound that by
   implying broader corroboration than exists.
6. **Preserve certainty/attribution distinctions** (confirmed vs.
   alleged/disputed) rather than flattening to one settled-sounding
   statement — untested against real contested claims on this corpus (see
   limitation above), included because TRD §4.5/PRD's non-scalar framing
   intent implies hedged claims shouldn't get flattened into false
   certainty once Phase 4 consumes cluster data.
7. Write for a reader, never mention "the cluster" or clustering
   mechanics.
8. No commentary/speculation beyond what the claims state.

## Real output, judged (manual spot-check, not scored)

Full output: `notebooks/phase3_summarization_samples.json`. Highlights:

- **Cluster 21** (clean 3-source paraphrase): "Three sources
  (asiaone.com, economictimes.indiatimes.com, and finance.yahoo.com)
  report that a version of Chinese startup DeepSeek's flagship AI model is
  by far the least expensive to run on benchmark tests among well-known
  models globally." — unified correctly, source count exactly matches the
  3 listed, nothing invented.
- **Cluster 31** (shared fact + 1 extra): correctly split into "four
  sources report [the $0.14/$0.28 pricing]... Additionally, one source
  states [the separate 3-cents-per-test estimate]" — did not blur the
  extra claim into the shared one, did not overclaim its source count.
- **Cluster 2** (genuine false-merge, two numbers): kept both numbers
  against their own products ("80% price cut for the lightweight GPT-5.6
  Luna model... 20% price reduction for the mid-tier GPT-5.6 Terra
  model") and correctly said "one source" for both, rather than averaging
  or picking one. This is the sharpest test of rule 4 and it held.
- **Cluster 9** (5 distinct facts, 1 source): all 5 benchmark-score facts
  present in the summary with their original numbers intact, nothing
  dropped to keep it "concise" — directly validates the "nothing silently
  omitted" acceptance requirement against the highest-risk real case in
  this corpus.
- **Cluster 1** (named third-party quote mixed with outlet's own
  reporting): correctly attributed the Trevor Koverko/SapienX claim to him
  by name rather than folding it into "siliconangle.com reports," even
  though `attribution_type` (from the `Claim` schema) is *not* passed to
  this prompt — the distinction is recoverable from `claim_text` alone
  because Task 3.2.1a's extraction prompt already keeps attribution
  language in the claim text itself. Confirms the summarization prompt's
  input contract (claim_text + domain only, no other `Claim` fields) is
  sufficient — no need to widen the input shape.
- **Cluster 0** (minor observed nuance, not a failure): two claims from
  one source describing the same PLA-unit distillation effort with
  slightly different framing ("process and summarize sensitive military
  source code" vs. "safely processing sensitive data... from distillation
  of GPT-3.5 outputs") were merged into one sentence. Defensible as the
  same underlying fact restated with different specificity (this is one
  article's own two sentences about one event, not two distinct facts),
  but the merged wording is tighter than either original claim — flagged
  as the one place iteration 1's output sits closest to the "unify vs.
  blend" line. Not treated as a failure requiring iteration 2: no number,
  entity, or fact was invented or contradicted, and rule 4's real target
  (distinct *numbers*/subjects) didn't apply here.

No invented facts and no silently dropped claims were found across any of
the 9 sampled clusters, including the four that were real false-merges
(31, 2, 9, and the shared+extra shape more generally). No second prompt
iteration was needed on this sample — flagging that as itself a
limitation of the sample (9 clusters, one subtopic, one model, one pass
each) rather than claiming the prompt is validated beyond what a
single-topic manual read supports.

## Input contract for `backend-engineer` (Task 3.6.1b)

Per-cluster input is a formatted list of `(domain, claim_text)` pairs for
the cluster's **asserting** rows only (`claim_cluster_articles.relation =
'asserts'`, joined to `articles.domain` for the source label) —
`{claims}` template variable, one line per claim: `- {domain}:
{claim_text}`. Omitting articles are not passed to this prompt (TRD §4.5's
contract is "each claim cluster," and non-assertion is Phase 4's
consensus/disputed/omissions job per TRD §4.7, not summarization's).
Output is `chain.invoke(...).content` (plain string, no
`with_structured_output`) written to `claim_clusters.summary`.

## Cost note (NFR-1)

`Settings.models.summarization` is already `gpt-4.1-mini` (small/medium
tier, matches TRD §4.5's "small/medium model" sizing) — this review used
that model as configured, no change recommended. Prompt is
~300 words + a per-cluster claim list (typically 2-5 short claims per the
real cluster-size distribution Task 3.3.1a documented) — well within a
small model's context and cost profile for a per-cluster call.
