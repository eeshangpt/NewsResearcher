# Phase 2 Task 2.2.3a — Subtopic reconciliation (merge/split/drop) design

Owner: `data-scientist`. Design/validation deliverable per
`EXECUTION_PLAN.md`'s "Phase 2 — Role Ownership & Parallelization
Re-analysis," specifically Task 2.2.3a as corrected by `tech-lead` (the
distinctiveness-score formula must be written down here as a concrete
deliverable, not just referenced in prose — see the final section of this
doc). Not production code. Handoff target: `backend-engineer`, Task 2.2.3b
(`agents/subtopic_agent.py`'s reconciliation step).

Validation script: `notebooks/phase2_reconciliation_eval.py`.
Fixtures: `tests/fixtures/reconciliation_merge.json`,
`reconciliation_split.json`, `reconciliation_drop.json` (paired with the
already-committed `tests/fixtures/clustering_synthetic_topics.json`, whose
`sentences`/`embeddings`/`true_labels` arrays these three fixtures index
into by position — see "How to use these fixtures" below).

## Scope and what this depends on

Per Task 2.2.3's original acceptance context: given (a) a broad topic-scoped
article fetch (Task 2.2.2, merged), (b) that fetch embedded and clustered
into cluster labels (Task 2.1.2's `cluster()`, HDBSCAN primary / KMeans
fallback — approach settled, productionized code (2.1.2b) still in flight in
parallel and not needed here), and (c) a set of LLM-proposed candidate
subtopic labels (Task 2.2.1's `subtopic_propose.txt` prompt, label +
rationale each — designed in `notebooks/phase2-subtopic-prompt-design.md`),
this document specifies exactly how to reconcile (b) and (c) into a final
subtopic list, and how to score/rank that list for Task 2.2.4.

This design is validated against the real `clustering_synthetic_topics.json`
fixture's real embeddings (local `sentence-transformers/all-MiniLM-L6-v2`
backend, via the already-merged `clustering/embeddings.py::embed()`), using
its ground-truth `true_labels` as a stand-in for `cluster()`'s output. That
substitution is intentional and appropriate scope, not a shortcut: Task
2.1.2a already validated clustering quality on this exact fixture (HDBSCAN
recovers the 4 true clusters at ARI 0.781); re-deriving that here would
duplicate that task rather than validate reconciliation, which is what
2.2.3a is actually responsible for. **This is a manual, fixture-based
validation, not a golden-dataset eval** — PRD §7 names that gap explicitly
for v1; treat every number below as directional evidence from one
constructed fixture, not a certified benchmark.

## Core mechanism

Two embedding-space comparisons drive all three rules:

1. **candidate → cluster-centroid similarity**: for each LLM candidate,
   embed `f"{label}. {rationale}"` (label alone is often only 3-10 words;
   the rationale carries real topical signal the bare label lacks) via
   `clustering/embeddings.py::embed()`. For each cluster found by
   `cluster()` (excluding HDBSCAN noise, label `-1`), compute its centroid as
   the mean of its member article embedding vectors. Compute cosine
   similarity between every candidate and every cluster centroid — an
   `n_candidates x n_clusters` matrix. Each candidate's **best cluster** is
   `argmax` over this row; its **best-cluster similarity** is that row's max
   value.
2. **candidate → candidate similarity**: cosine similarity between two
   candidates' own embeddings (same `embed()` call, no article involvement).
   This is the key signal that disambiguates merge from split (see below) —
   a design correction made during this task's own validation, documented in
   "Design iteration" below.

### Rule 1 — Merge: when do two+ candidates map to the same cluster?

**Condition:** a candidate *claims* cluster `c` if `c` is its best cluster
AND its best-cluster similarity ≥ `MATCH_THRESHOLD` (0.60, derivation below).
If two or more candidates claim the same cluster `c`, group them by mutual
candidate-candidate similarity: any pair of claimants of `c` whose
candidate-candidate cosine similarity ≥ `CANDIDATE_DUP_THRESHOLD` (0.65,
derivation below) belong in the same "claim-group" (single-linkage grouping
via union-find over all claimant pairs). **If every claimant of `c` collapses
into one claim-group, merge them**: emit one subtopic for cluster `c`, whose
supporting article count is the cluster's full member count, and whose
canonical label is the claimant with the single highest candidate→centroid
similarity (others retained as `merged_from` aliases, useful for Gate 1
display — "also proposed as: ...").

**Validated (fixture: `reconciliation_merge.json`):** candidates "EU AI Act
enforcement actions" and "European Union AI Act compliance crackdown" (two
independently-worded near-duplicate proposals about the same real story)
both claim the real `eu_ai_act_enforcement` cluster (centroid similarities
0.866 and 0.861) and have candidate-candidate similarity 0.828 ≥ 0.65 →
correctly merge into one subtopic backed by that cluster's full 8 articles.
Two other candidates ("AI copyright lawsuits...", "US executive order...")
each singly and correctly claim their own distinct clusters. The fourth real
cluster (`state_level_ai_legislation`) has no claimant in this scenario —
see "Unclaimed clusters" below.

### Rule 2 — Split: when does one cluster span multiple genuinely distinct candidates?

**Condition:** same claim-group construction as above, but if two or more
claimants of cluster `c` land in **different** claim-groups (i.e. no
claimant pair reaches `CANDIDATE_DUP_THRESHOLD`), the cluster's real article
content is not actually distinguishing what the LLM proposed as separate
angles — **split**: for each member article of `c`, compute its cosine
similarity to every claimant candidate and assign it to whichever
claim-group contains its single best-matching claimant. Emit one subtopic
per resulting group, each backed by only its assigned article subset (not
the full original cluster), with the group's highest-centroid-similarity
claimant as canonical label.

**Validated (fixture: `reconciliation_split.json`):** to exercise a genuine
split (not merely simulate a hypothetical), the fixture deliberately
constructs the failure mode the rule exists to catch: the real
`us_executive_order_ai_safety` (8 articles) and `state_level_ai_legislation`
(8 articles) clusters are relabeled as one synthetic 16-article cluster (id
`99`), standing in for an upstream `cluster()` call that under-split two
genuinely distinct subtopics into one coarse blob. Candidates "White House
executive order on AI safety" and "State legislatures passing AI regulation
bills" both claim cluster `99` (centroid similarities 0.776 and 0.795) but
their candidate-candidate similarity is only 0.453 (< 0.65) → correctly
split: 8 articles assigned to each, matching the true 8/8 composition of the
synthetic merge exactly.

### Rule 3 — Drop: when does a candidate have no real supporting cluster?

**Condition:** a candidate is dropped if its best-cluster similarity <
`MATCH_THRESHOLD` (0.60) — i.e. even its single closest cluster centroid
isn't a real match. Dropped candidates are excluded entirely, never
force-merged into an unrelated cluster.

**Validated (fixture: `reconciliation_drop.json`):** two candidates sharing
only broad-domain vocabulary with the fixture's real subtopics but with no
actual supporting cluster — "AI chip export control negotiations" (best sim
0.525, against `state_level_ai_legislation`) and "AI voice assistant privacy
concerns" (best sim 0.422) — both fall under 0.60 and are dropped. A third,
genuinely matching candidate ("EU AI Act enforcement actions", sim 0.866)
survives as a single match. Both drop cases were deliberately chosen to be
*harder* than an obviously off-topic control (a basketball-headline
candidate scored 0.097 against every cluster in a side probe — trivially
droppable and not a useful validation case on its own).

### Threshold derivation (not guesses — real observed gaps on this fixture)

Both thresholds were set from an actual measured gap in cosine-similarity
values on this fixture, not chosen a priori:

| Comparison type | Observed cosine range |
|---|---|
| Genuine candidate ↔ its true cluster centroid | 0.776 – 0.908 |
| Same-broad-domain but unsupported candidate ↔ any cluster centroid | 0.363 – 0.525 |
| Off-domain control (e.g. basketball headline) ↔ any cluster centroid | 0.097 – 0.126 (side probe, not in committed fixtures) |
| Near-duplicate candidate pair (same real story, reworded) | 0.828 |
| Genuinely distinct candidate pair (different real subtopics) | 0.453, 0.483 |

`MATCH_THRESHOLD = 0.60` sits in the wide, clean gap between "genuine match"
(≥0.776) and "same-domain-but-unsupported" (≤0.525) observed on this
fixture. **Design-iteration note, stated plainly:** the first version of
this script used `MATCH_THRESHOLD = 0.42`, which was not derived from this
gap and incorrectly let the "AI chip export control negotiations" candidate
claim a cluster at 0.525 similarity — re-running the drop scenario surfaced
this failure directly, which is why the threshold was raised to 0.60 and
re-validated. `CANDIDATE_DUP_THRESHOLD = 0.65` sits between the observed
near-duplicate pair (0.828) and the observed genuinely-distinct pair (0.453,
0.483). Both thresholds are derived from **one constructed fixture's worth of
real embeddings** — a real measurement, but on synthetic data at one
specific set of hand-written candidate phrasings. They should be treated as
starting points, not final calibration, and revisited once Task 2.2.3b has
real Gate-1 candidate/reconciliation data to check them against (this
mirrors the same caveat Task 2.1.2a's clustering recommendation doc already
states for `kmeans_fallback_threshold`).

**Design-iteration note (why candidate-candidate similarity, not per-article
vote, decides merge vs. split):** an earlier draft of this algorithm decided
merge-vs-split purely by looking at how a claimed cluster's *member articles*
individually voted between claimant candidates (a "does one candidate
dominate >X% of articles" heuristic). Running that draft against real
embeddings showed it is fragile: the genuine near-duplicate EU AI Act pair
above (candidate-candidate similarity 0.828) still had its cluster's 8
member articles split 5/3 between the two candidates by per-article
similarity — well short of any reasonable "one candidate dominates"
threshold, which would have wrongly triggered a SPLIT for what is clearly
one story described two ways. Comparing the claimant candidates to *each
other* directly, instead of relying on noisy per-article voting, is the
robust signal and is what `notebooks/phase2_reconciliation_eval.py`
implements. Per-article similarity is still used, but only *after* the
merge-vs-split decision is made, to partition a split cluster's articles
between the resulting groups.

### Unclaimed clusters (a gap this task's scope does not resolve)

A cluster with a genuine article grouping but zero candidates claiming it
(observed in the merge fixture: `state_level_ai_legislation` had no
claimant) is out of Task 2.2.3's literal scope, which only defines
merge/split/drop over *candidates*. **Explicit design call, flagged for
`tech-lead`/`backend-engineer` rather than silently decided:** an unclaimed
cluster is dropped from the reconciled list entirely in this design (it
never becomes a subtopic, since there is no LLM-authored label for it), not
force-labeled with a generic/fallback name. This means a real,
article-supported subtopic can be silently lost at Gate 1 if
`n_candidates` (from Task 2.2.1a, recommended default 8) undershoots the
broad fetch's real cluster count. Recommended mitigation, not implemented
here since it's a `Settings`/telemetry decision: log a per-run count of
unclaimed clusters (with sizes) so an operator/future story can see how
often this happens and retune `n_candidates` or add an LLM re-prompt step if
the rate is high. This is a real, named gap, not an oversight.

## Distinctiveness-score formula (Task 2.2.3a's corrected deliverable)

Feeds Task 2.2.4's rank/cap/excess-retention step directly. Written out
explicitly, per `tech-lead`'s specific correction to this task (the original
draft only described this qualitatively in prose):

For a reconciled subtopic `s` with article count `count(s)`, embedding
centroid `centroid(s)` (the centroid of the cluster or cluster-subset backing
it after merge/split), and the full reconciled subtopic set `S` (all
surviving subtopics after merge/split/drop, `|S| = n`):

```
volume_norm(s)     = count(s) / total_articles_in_broad_fetch

avg_pairwise_distance(s) = mean over all s' in S, s' != s of:
                              1 - cosine_similarity(centroid(s), centroid(s'))
                           (0.0 if n == 1, i.e. no peer to compare against)

distinctiveness_score(s) = 0.5 * volume_norm(s) + 0.5 * avg_pairwise_distance(s)
```

Both `volume_norm` and `avg_pairwise_distance` are already bounded to
roughly `[0, 1]` (cosine similarity of embedding vectors from this model
rarely goes negative for topically-related short text, so `1 - cos_sim`
stays in a practical `[0, ~1]` range) — no additional rescaling needed before
combining with equal 0.5/0.5 weights.

**Why 0.5/0.5 equal weighting, stated as a starting point, not a tuned
result:** this weighting is chosen so that the single largest, most generic
cluster does not automatically dominate ranking purely on volume — a
subtopic that is both well-covered *and* clearly distinct from the other
surviving subtopics should rank highest, matching the PRD/TRD's intent that
`max_subtopics` truncation shouldn't just keep the biggest catch-all bucket
and starve genuinely distinct smaller angles. This 0.5/0.5 split has **not**
been tuned against real operator judgments of "which 5 subtopics should have
survived" (no such labeled data exists yet, per PRD §7's v1 eval gap) — it
is a defensible, symmetric starting point that should be revisited once
Task 2.2.4 has real capped/excess-retained output for an operator to
manually spot-check against.

**Demonstrated (see `phase2_reconciliation_eval.py`'s
`scenario_distinctiveness`, using the merge scenario's 4 surviving
subtopics):**

| Subtopic | article_count | volume_norm | avg_pairwise_distance | distinctiveness_score |
|---|---|---|---|---|
| AI copyright lawsuits over training data | 8 | 0.25 | 0.474 | **0.362** |
| US executive order on AI safety | 8 | 0.25 | 0.420 | 0.335 |
| EU AI Act enforcement actions (merged) | 8 | 0.25 | 0.381 | 0.316 |
| State legislatures passing AI regulation bills | 8 | 0.25 | 0.368 | 0.309 |

All four subtopics have equal volume here (fixture is intentionally
balanced 8/8/8/8), so the ranking is driven entirely by
`avg_pairwise_distance` — i.e. by which subtopic's centroid sits furthest
(in embedding space) from the other three. This is a plausibility
demonstration only (a manual read of one small fixture), not a validated
ranking benchmark.

## Settings-ready threshold names (for `backend-engineer`, Task 2.2.3b — not added to `config.py` by me)

None of these exist on `ClusteringSettings` today (only `similarity_threshold`
and `subtopic_match_threshold` are defined). Recommended new fields, named to
fit the existing `Settings.clustering.*` convention:

| Field | Recommended default | Used for |
|---|---|---|
| `Settings.clustering.reconciliation_match_threshold` | `0.60` | Rule 1/3: candidate→cluster-centroid cosine similarity to claim a cluster (or be dropped if below) |
| `Settings.clustering.reconciliation_dup_threshold` | `0.65` | Rule 1/2: candidate→candidate cosine similarity to treat two claimants as the same claim (merge) vs. distinct claims (split) |
| `Settings.clustering.distinctiveness_volume_weight` | `0.5` | Distinctiveness formula's volume term weight (must sum to 1.0 with the distance weight below) |
| `Settings.clustering.distinctiveness_distance_weight` | `0.5` | Distinctiveness formula's avg-pairwise-distance term weight |

## How to use these fixtures (for `backend-engineer`, Task 2.2.3b)

Each `tests/fixtures/reconciliation_{merge,split,drop}.json` contains:
- `candidate_labels` / `candidate_rationales`: the LLM-proposed candidates
  for that scenario.
- `candidate_embeddings`: pre-computed 384-dim vectors for each candidate
  (via `embed()`, local backend) — no live model call needed at test time.
- `cluster_ids`: the cluster label set in play for that scenario.
- `article_true_labels` (merge/drop scenarios) or `article_cluster_labels`
  (split scenario, since it synthetically relabels two real clusters as one):
  per-article cluster assignment, same order/length as
  `clustering_synthetic_topics.json`'s `sentences`/`embeddings` arrays — load
  that fixture's `embeddings` array alongside this one to get the actual
  article vectors these labels index into.
- `expected_outcome`: which candidates should merge / split / get dropped,
  for a deterministic assertion.

A unit test can load both fixture files together, run the reconciliation
logic (once ported into `agents/subtopic_agent.py`), and assert the action
taken for each candidate matches `expected_outcome` — no network/API call or
live embedding model needed, matching the pattern already established by
Task 2.1.2a's `clustering_synthetic_topics.json` fixture for
`backend-engineer`'s unit tests.

No full article text appears anywhere in these fixtures or in the script
that generated them — only short synthetic headline-style sentences (reused
from the already-committed clustering fixture) and short candidate
label/rationale strings, consistent with the no-full-text-persistence rule.
