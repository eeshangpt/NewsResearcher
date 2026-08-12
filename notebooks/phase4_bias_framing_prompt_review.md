# Task 4.1.1a — Bias & Framing Agent prompt + output schema

Owner: `data-scientist`. Design deliverable only — no production wiring.
`backend-engineer` owns Task 4.1.1b (wiring via
`get_chat_model("bias_framing").with_structured_output(...)`, Langfuse trace).

Deliverables:
- `newsresearch/llm/prompts/bias_framing.txt` — the prompt (final, iteration 5).
- `notebooks/phase4_bias_framing_schema_draft.py` — the schema, paste-ready for
  `newsresearch/llm/schemas.py`, plus two runnable structural self-checks.
- `notebooks/phase4_bias_framing_review.py` — the real-data harness.
- `notebooks/phase4_bias_framing_samples.json` — raw output of the final
  3-repeat run (93 cluster-outputs).

Scope note: this task is the **labels** only. The include/omit/emphasis
comparison *prose* is Task 4.2.1a, and the briefing synthesis is 4.3.1a. I
deliberately did not build either here.

## Architectural input I designed to (not re-litigated)

`tech-lead`, 2026-08-12: the agent batches a bounded number of claim clusters
per call — not one call per cluster, not one monolithic call per subtopic. New
`Settings.models.bias_framing_batch_size`, default 8. The schema below is
therefore a **list of per-cluster objects nested under one batch-level
completion** (`BiasFramingBatch.clusters`), so each batch's output merges
additively into the subtopic's full result set. All evaluation below was run
at that granularity (8 clusters per call; 31 clusters → batches of 8/8/8/7).

## Schema shape

Three nested models. Full field descriptions live in
`phase4_bias_framing_schema_draft.py`.

```
BiasFramingBatch
  clusters: list[ClusterFraming]           # one per cluster in this batch

ClusterFraming
  cluster_id:        str
  cluster_coherence: Literal[single_shared_fact | multiple_distinct_facts]
  shared_focus:      str
  divergence:        Literal[single_article_only | no_notable_divergence |
                             wording_or_emphasis_differs | context_differs |
                             accounts_conflict]
  divergence_note:   str
  per_article:       list[ArticleFraming]

ArticleFraming
  article_id:          str
  domain:              str
  epistemic_treatment: Literal[states_as_established | attributes_to_named_party |
                               hedges_as_uncertain | reports_as_contested]
  contextual_frame:    str    # the descriptive framing label, ~4-12 words
  evidence_quote:      str    # verbatim span from THIS article's claims
```

This is deliberately the same family of decision as `Claim.attribution_type` /
`Claim.certainty` and `agents/sentiment.py`'s 3-class label: closed enum
taxonomies, **no numeric score anywhere**, for the same documented reason — a
float invites the false-precision misread that FR-14 and the PRD risk table
warn about.

### Two deliberate departures from the obvious shape

**1. `per_article`, not `per_source`.** TRD §4.6 says "per cluster and per
source". I key on `article_id` and carry `domain` alongside, because in the
real data one domain contributes several articles with genuinely different
editorial character: `theguardian.com` supplies both a news report
(`leipzig-drone-1`) and an opinion column (`leipzig-drone-2`). Blending them
into one "theguardian.com" label manufactures an outlet-level bias finding out
of a genre difference. Domain is retained, so a domain-level roll-up remains
available downstream to anything that genuinely wants one — this is a strict
superset of the TRD's contract, not a narrowing of it.
**Flagged to `tech-lead`/`backend-engineer` as an I/O-contract nuance rather
than assumed**, per my standing instruction not to redesign an agent's
contract unilaterally.

**2. No omission field.** TRD §4.6's "which sources omit it" is *not* model
output here. `claim_cluster_articles` already stores `relation='omits'`
deterministically; asking the model to restate it is exactly the failure class
Task 3.7.1b diagnosed (the model reproducing a count/membership the DB already
knows, and getting it wrong). Omission is read from Postgres and joined in
downstream by 4.2.1b. The prompt explicitly forbids emitting absence entries.

Note for whoever consumes this: in the real data a single **domain** can appear
in both the `asserts` and `omits` relations for one cluster (Guardian's news
piece asserts, its column omits). Omission is an **article**-level fact and
must not be collapsed to domain level when presented.

## How the schema *structurally* prevents a left/center/right field

Not "the prompt asks nicely." Four independent structural properties, each of
which alone blocks the obvious encoding:

1. **There is no numeric field anywhere in the schema tree.** A
   left/center/right scale is by definition a position on an ordered axis, and
   its natural encoding is a number. The schema offers no `int`/`float` slot at
   any nesting level, so there is nowhere to put one. (This is the same
   rejection as `Claim`'s deliberate absence of a confidence score.)
2. **Every closed-vocabulary field is a `Literal`, and none of the vocabularies
   contain a political term.** Under `.with_structured_output()` these are
   emitted as JSON-Schema `enum`s and constrained at decode time — `"left"`,
   `"centre"`, `"conservative"` are *unrepresentable*, not merely discouraged.
   Pydantic rejects them a second time on validation. Verified by a runnable
   assertion, not by inspection (see below).
3. **The object graph is cluster-major, with no outlet-level object at all.**
   A political lean is a property claimed about an *outlet*. There is no
   `SourceProfile`, no per-domain summary, no top-level per-outlet node —
   nothing anywhere in the tree whose scope is "an outlet". Every single label
   is nested inside one specific cluster and, below that, one specific article.
   To emit "The Guardian is left-leaning" the model would have to re-state it
   independently inside every cluster, in a field whose contract is about that
   cluster's wording.
4. **Every free-text label is anchored to a required verbatim
   `evidence_quote`** copied from that article's claims in that cluster. The
   two free-text fields that could in principle host a lean label
   (`contextual_frame`, `divergence_note`) sit next to a field demanding the
   source wording the label is based on. A political-lean judgement has no
   quote to cite, because it is not derivable from any span of the claim text.

Properties 1 and 2 are machine-checked by
`phase4_bias_framing_schema_draft.py`'s `_assert_no_political_scalar_field()`
and `_assert_enums_are_closed()`, which walk the model tree and assert (a) no
`int`/`float` annotation, (b) no banned term in any field name or enum member,
(c) that `ArticleFraming` actually raises `ValidationError` on
`"left_leaning"`, `"centre"`, `"conservative"`, `"3"`. Run with
`uv run python notebooks/phase4_bias_framing_schema_draft.py`. This exists so
that a later well-meaning edit adding a `bias_score: float` fails a check
rather than passing review. Prompt rules 12–14 are the *secondary* defence;
they are not what this task's acceptance criterion rests on.

Properties 3 and 4 are design-level and not machine-checkable — stated here so
a reviewer can hold me to them.

## Method

Real data throughout, no fixtures:

- The real persisted subtopic `task371-leipzig-drone` in local Postgres
  (`NEWSRESEARCH_DATABASE_URL`): **31 claim clusters** with asserting claims,
  from 4 real articles — `bbc.com`, `theguardian.com` ×2 (a news report and an
  opinion column), `npr.org`.
- Clusters read via the production
  `persistence/claim_clusters.py::read_cluster_article_relations`, not re-typed.
- Real `get_chat_model("bias_framing")` (`gpt-4.1`) calls through
  `ChatPromptTemplate.from_template()` loading the real prompt file, with
  `.with_structured_output(BiasFramingBatch)` — the same path 4.1.1b will wire.
- Batched 8 per call. Final run repeated **3 independent times** (12 real LLM
  calls, 93 cluster-outputs) to measure sampling stability rather than trusting
  one draw.
- **No article full text touched, fetched, or written anywhere** — only
  `claim_text`, `article_id`, `domain` leave the DB (FR-25).
- Source metadata: `articles.reputation_score_at_fetch` is NULL for every row
  in this subtopic, so reputation was **not** evaluated. I would not feed it in
  anyway: a per-outlet scalar in the prompt invites exactly the outlet-level
  scoring this design excludes. Flagging rather than deciding silently.

## Iteration — each pass driven by an observed real failure

**Iteration 1 → 25 mechanical failures.** Two root causes.
*(a)* The model emitted one `per_article` entry **per claim line** instead of
per article — the same line-vs-entity confusion Task 3.7.1b found in
summarization, recurring in a new place. *(b)* Consequently, clusters 19 and 25
— each containing exactly **one** article — were labelled
`wording_or_emphasis_differs`, i.e. the agent reported a difference *between
sources* where only one source existed. That is the single worst failure mode
available to this agent: a fabricated bias signal, which would feed the
briefing agent a disputed claim that does not exist. Fixed by rewriting rule 4
(one entry per distinct article, combining its lines) and rule 8 (divergence
compares distinct articles; one article ⇒ `single_article_only` regardless of
line count). Also tightened rule 7 after 6 quotes used `...` elisions.

**Iteration 2 → 0 mechanical failures**, but manual review caught what the
automated check did not: cluster 1 has **two** distinct articles (both
Guardian) yet was labelled `single_article_only` in 3/3 runs. My check only
tested one direction. Widened it → **9/69 multi-article cluster-outputs
wrongly labelled `single_article_only`** (clusters 1, 2, 24 — every case the
"2 articles, 1 domain" shape, 3/3 runs each; fully systematic, not sampling
noise). The model was reasoning at outlet level.

**Iteration 3** sharpened rule 8's wording ("count article ids, never
domains"). Result: **9/69 → 6/69**. Prompt nagging alone was not fixing it.

**Iteration 4 — the actual fix.** Root cause: I was asking the model to
*count* distinct articles, which Task 3.7.1b already established as a
demonstrated model failure on this data. So I stopped asking. The harness now
renders a `Distinct articles in this cluster (N): id, id` roster line into each
cluster's block — the DB already knows it — and rule 4/8 declare that roster
authoritative and forbid re-deriving it. Result: **6/69 → 0/69**. Two residual
failures appeared in the opposite direction on cluster 7.

**Iteration 5 (final).** Cluster 7 is one BBC article carrying both a suspicion
of Moscow *and* Russia's denial; the model read that tension as
`accounts_conflict`/`no_notable_divergence`, overriding the roster on content
grounds. Added a clause: a tension *within* one article is
`epistemic_treatment=reports_as_contested`, not divergence. Result: cluster 7
is now `single_article_only` + `reports_as_contested` in 3/3 runs.

**Final: 0/93 divergence-rule violations, 0/93 per-article-membership errors.**

### Residual known defect (not fixed, stated plainly)

**3 of 186 `evidence_quote` values (1.6%)** are non-verbatim, all on cluster 22,
all the same pattern: the model elides the middle of a long compound claim with
`...` to quote the assertion-plus-denial arc (`'Police have warned of a
potential Russian involvement ... which Moscow has denied'`). Rule 7 forbids
this explicitly and it persisted anyway. These are *faithful elisions, not
fabrications* — no invented wording — so I stopped iterating rather than
distort the prompt further for a cosmetic defect.
**Recommendation to `backend-engineer`:** verify `evidence_quote` is a
substring of that article's claim text at wiring time and blank it if not,
soft-fail, per the house soft-fail convention. Deterministic and cheaper than
more prompt pressure.

## Acceptance dimension (a): labels descriptive, not scalar-political

**PASS.** Across the final run's **186 `contextual_frame` values** (155
distinct) and **93 `divergence_note` values**: zero political-lean, ideological,
partisan, or evaluative-ranking language. A crude substring scan flagged 5
strings; all 5 are false positives on the literal device name "**anti**-
explosives robot" taken from the source claims. No output rates, ranks, or
grades an outlet, and none compares outlets on trustworthiness or objectivity.

The descriptors describe *what the article's words do*:

| | |
|---|---|
| `relays differing media and tabloid reports on device composition` | attribution behaviour |
| `places drone claims within wider context of European sabotage accusations` | contextualisation |
| `keeps to procedural police response` / `describes procedural police response and outcome` | scope choice |
| `frames airport as strategic NATO and Ukrainian hub` | salience choice |
| `emphasises ongoing uncertainty and investigation` | epistemic posture |
| `highlights limited nature of the damage` | emphasis |

Worth noting: `russian`/`ukrainian` occur in descriptors, but as the *subject
matter* of the incident, never as an alignment attributed to an outlet — the
distinction a naive keyword filter would miss.

A meaningful negative signal: **`plain factual statement, no wider framing`
appears 29 times.** Rule 6's explicit escape hatch is being used, i.e. the
model declines to invent a frame where none exists rather than manufacturing
bias to fill the field. That false-positive resistance matters more here than
richness of description.

## Acceptance dimension (b): label consistency across sources on the same claim

**PASS on the per-article label; PARTIAL on the cluster-level degree-of-difference.**
Measured as agreement across 3 independent runs of the identical input:

| field | stable across 3/3 runs |
|---|---|
| `epistemic_treatment` (per article) | **60/62 (97%)** — multi-article clusters only: 52/54 (96%) |
| `divergence` (per cluster) | 26/31 (84%) — multi-article only: 18/23 (78%) |
| `cluster_coherence` | 23/31 (74%) |

The direct cross-source test — the same underlying fact reported by different
outlets:

- **Cluster 9** (Dobrindt's "hybrid threat scenario", asserted by bbc.com,
  theguardian.com, npr.org): `attributes_to_named_party` for all three outlets
  in all three runs — **9/9 identical**. No outlet singled out, no drift.
- **Cluster 21** (bbc.com "the ambassador *stated Russia could be responsible*"
  vs theguardian.com "the ambassador *blamed Moscow*"): both
  `attributes_to_named_party` 3/3, cluster `wording_or_emphasis_differs` 3/3,
  and the note correctly names the differing verbs. This is the textbook
  emphasis-divergence case and it is stable and correct.

**The instability is confined to the degree-of-difference boundary** —
`no_notable_divergence` ↔ `wording_or_emphasis_differs` ↔ `context_differs`.
Across all 93 outputs, no cluster ever drifted into or out of
`accounts_conflict` (it occurs exactly 3 times, all cluster 13, 3/3 runs).
That is the safety property that matters: the value the briefing agent will
read as "disputed" is the stable one; the churn is on how *big* a wording
difference is, which is a genuinely fuzzy judgement.

I am not claiming 78% is good. It is a real, honest ceiling on the cluster-level
divergence value and downstream (4.3.1a) should treat
`wording_or_emphasis_differs` vs `context_differs` as roughly interchangeable
rather than as a meaningful distinction.

## Handling the 0.674 claim-clustering ceiling

Task 3.3.1a measured claim-cluster false-merge precision at 0.674 — ~1 in 3
clustered pairs are distinct facts wrongly merged. The danger for *this* agent
is specific: **a false merge looks exactly like source disagreement**, so a
naive prompt converts a clustering defect into a fabricated bias finding.

Handled by prompt rules 1–3: decide coherence *before* divergence, and
`accounts_conflict` is reserved for assertions about the same subject that
cannot both be true — two articles covering different facts are explicitly
*not* in conflict.

Real test — **cluster 23**, a genuine false merge (bbc.com "the detonator was
removed... controlled explosion" + npr.org "police removed the detonator" +
theguardian.com's column "the bomb did not explode because the detonator was
faulty" — procedure vs. cause-of-failure, distinct facts):
**it never once returned `accounts_conflict`** across 3 runs, returning
`wording_or_emphasis_differs` with notes correctly separating the police-action
account from the faulty-detonator account. The guard holds on the case it was
written for.

`cluster_coherence=multiple_distinct_facts` fires on **37 of 93 outputs
(40%)**, in the same neighbourhood as the measured 0.674 false-merge precision,
and gives downstream an explicit flag for "this grouping is mixed" rather than
hiding it. I have not checked those 37 against hand-labelled ground truth —
the agreement in magnitude is a sanity check, not a validation.

## Honest caveats

- **This is a manual spot-check, not a scored eval.** PRD §7 / §8's named v1
  gap: there is no golden dataset and no automated quality metric. The
  stability percentages above are *self-consistency across repeated runs of one
  model on one subtopic* — they measure determinism, **not** correctness. A
  label can be stably wrong. The correctness judgements (dimensions a and b)
  are my reading of real output against real claim text, nothing more.
- **One subtopic, one event, one day, 4 articles, 3 domains.** The task brief
  says "several real persisted subtopics"; only this one exists in the DB with
  populated clusters. Narrow in an important way: it is a security-incident
  story where the outlets largely agree on facts. A contested political story
  would exercise `accounts_conflict` and the anti-lean rules far harder than
  this corpus does, and I have **not** tested that. This is the most important
  limitation of this write-up.
- **`gpt-4.1` only**, per `Settings.models.bias_framing`. I did not evaluate
  whether the large model is necessary here — the epistemic/framing judgement
  is the reasoning-heavy call NFR-1 explicitly budgets the large model for, so
  I left the assignment alone. Batching 8 clusters per call already cuts call
  count ~8× versus per-cluster, which is the main NFR-1 lever available.
- **Genre is invisible to the agent.** Cluster 13's `accounts_conflict`
  (theguardian.com's *opinion column* asserting a Russian-orchestrated plot vs
  npr.org's *news report* saying it is under investigation) is a correct
  description of the two claim texts, but a reader would want to know one is a
  column. There is no article-type column in `articles`, so the agent cannot
  know. **Flagged to `tech-lead` as a possible schema addition** — it is an
  infra/data-model call, not mine. Until then, 4.3.1a's briefing prompt should
  not treat a lone `accounts_conflict` as proof of an outlet-level factual
  dispute.
- Batch size 8 was given, not tuned. Structured output was reliable at 8
  (12/12 calls returned exactly the requested cluster count, 3 runs). I have
  no evidence about where it degrades.

## Handoff to `backend-engineer` (Task 4.1.1b)

1. **Prompt**: `newsresearch/llm/prompts/bias_framing.txt`, final, on
   `feat/datascientist`. Single `{clusters}` placeholder, loaded with
   `ChatPromptTemplate.from_template()`. No other placeholder.
2. **Schema**: copy the three models from
   `notebooks/phase4_bias_framing_schema_draft.py` into
   `newsresearch/llm/schemas.py` verbatim (drop the two `_assert_*` helpers, or
   port them to `tests/` — they are the machine-checkable half of this task's
   acceptance criterion and are worth keeping *somewhere*).
3. **Input formatting is load-bearing, not incidental.** Mirror
   `phase4_bias_framing_review.py::format_batch` exactly:
   ```
   Cluster <cluster_id>
   Distinct articles in this cluster (N): <id>, <id>
   Claims (article id | source domain | claim):
   - <article_id> | <domain> | <claim_text>
   ```
   The roster line is not decoration — removing it regressed
   `single_article_only` accuracy from 0/69 back to 6/69 in iteration 3.
   Only `relation='asserts'` rows go in; omitting rows must be excluded.
4. **Batch** at `Settings.models.bias_framing_batch_size` (default 8), one
   subtopic's clusters per batch, merging each call's `clusters` list
   additively.
5. Suggested cheap guard: drop/blank any `evidence_quote` that is not a
   substring of that article's claim text (the 1.6% ellipsis defect above),
   soft-fail.
6. `claim_clusters.framing_label` is a single `TEXT` column in the TRD schema,
   but this output is structured and per-article. **How it gets persisted is a
   storage-shape decision for `tech-lead`/Story 4.4, not something I should
   pre-empt** — flagging rather than assuming a JSON blob in that column.
