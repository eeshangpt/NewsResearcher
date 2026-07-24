# Phase 2 Task 2.1.2a — HDBSCAN vs. KMeans clustering recommendation

Owner: `data-scientist`. Offline prototyping/evaluation deliverable per
`EXECUTION_PLAN.md`'s "Phase 2 — Role Ownership & Parallelization
Re-analysis." Not production code. Handoff target: `backend-engineer`,
Task 2.1.2b (`clustering/cluster.py`).

Analysis script: `notebooks/phase2_clustering_eval.py`.
Fixture: `tests/fixtures/clustering_synthetic_topics.json`.

## What "good" means here

This is a manual/analytical evaluation against a constructed fixture with
known ground truth, not a golden-dataset eval (PRD §7 names that gap
explicitly for v1). The metric used is Adjusted Rand Index (ARI) against the
fixture's known topic labels — a legitimate quantitative measure *for this
fixture*, but the fixture itself is small and synthetic, so treat the exact
numbers as directional, not as a certified benchmark of real-run clustering
quality. The real-data spot-check against the GDELT climate fixture (below)
is a manual qualitative read, explicitly labeled as such.

## Recommended defaults

| Setting | Recommended value | Where it's used |
|---|---|---|
| `min_cluster_size` | **4** | `HDBSCAN(min_cluster_size=..., min_samples=...)` |
| `min_samples` | **2** | same call (min_samples=1 scored identically on this fixture; 2 is the more conservative choice against noisy real headline text — see reasoning below) |
| `Settings.clustering.kmeans_fallback_threshold` | **20** (article count) | `clustering/cluster.py`'s HDBSCAN→KMeans fallback branch |

`Settings.clustering.kmeans_fallback_threshold` does not currently exist on
`ClusteringSettings` (only `similarity_threshold` and
`subtopic_match_threshold` are defined today) — `backend-engineer` needs to
add the field as part of 2.1.2b.

## Why these values: methodology

**Library note:** the standalone `hdbscan` package (the TRD's stated
library choice) is not yet installed in this environment — Task 2.0.1
(`uv add hdbscan scikit-learn`) hasn't landed. `scikit-learn` 1.9.0 *is*
already present and ships its own `sklearn.cluster.HDBSCAN` (added in
sklearn 1.3), which this evaluation used instead. The core algorithm and the
hyperparameters (`min_cluster_size`, `min_samples`) are the same concept in
both implementations, so the recommended values should transfer directly,
but `backend-engineer` should re-run a quick sanity check against the
committed fixture using the actual `hdbscan` package before locking these in
as final, in case of implementation differences (e.g. default `cluster_selection_method`,
distance metric handling) between the two libraries.

### 1. First fixture attempt was too easy — discarded

The initial fixture used four *unrelated* news topics (Fed rate policy, AI
chip export controls, Premier League transfers, Amazon deforestation).
Every `min_cluster_size`/`min_samples` combination tested (2-6 / 1-3) and
KMeans both scored a trivial ARI of 1.0 — the topics are so semantically
distant in embedding space that any reasonable clustering setting recovers
them perfectly. This doesn't exercise the actual decision the pipeline
needs, which is distinguishing subtopics that are all closely related angles
on one broad topic. Replaced with a harder fixture (below) before drawing
any conclusion.

### 2. Real fixture: four subtopics within one broad topic

`tests/fixtures/clustering_synthetic_topics.json` — 32 headline-style
sentences, 8 each across 4 subtopics of one broad topic ("AI regulation"):
`eu_ai_act_enforcement`, `us_executive_order_ai_safety`,
`ai_copyright_lawsuits`, `state_level_ai_legislation`. All embedded via the
real `newsresearch.llm.models.get_embeddings()` factory (local
`sentence-transformers/all-MiniLM-L6-v2` backend, 384-dim — the project
default, no `OPENAI_API_KEY` required).

Full sweep results (32 points, all 4 subtopics at 8 each):

| min_cluster_size | min_samples | clusters found | noise points | ARI |
|---|---|---|---|---|
| 2 | 1 or 2 | 6 | 6 | 0.553 |
| 3 | 1 or 2 | 5 | 5 | 0.712 |
| 3 | 3 | 3 | 12 | 0.737 |
| **4** | **1 or 2** | **4** | **8** | **0.781 (best)** |
| 4 | 3 | 2 | 2 | 0.324 |
| 5-6 | 1-3 | 2 | 1-2 | 0.31-0.32 |

`min_cluster_size=4` is the clear peak: it recovers the correct number of
clusters (4) with the highest ARI. Lower values over-split near-duplicate
phrasing into spurious extra clusters (5-6 clusters instead of 4); higher
values under-split, merging genuinely distinct subtopics together (down to
2 clusters). `min_samples=1` and `min_samples=2` were tied on this fixture;
`min_samples=2` is the recommended default anyway because it's the more
conservative choice (requires slightly denser local neighborhoods before
committing a point to a cluster core), which should generalize a bit better
against noisier real headline text than this fixture's clean synthetic
sentences.

For comparison, `KMeans(n_clusters=4)` (i.e. **told** the true k) scored a
perfect ARI of 1.0 on the same 32-point set. This is the central real
trade-off, not just a fixture quirk: KMeans wins when the target number of
clusters is already known (or well-estimated) and articles are being forced
into that many buckets regardless of density; HDBSCAN wins when the number
of true clusters is unknown up front and/or the data contains real noise
that shouldn't be force-assigned anywhere — at the cost of needing enough
points per cluster to detect density structure at all.

### 3. Subsample sweep — where HDBSCAN degrades as article count drops

Same fixture, subsampled to progressively fewer points per subtopic (using
the winning `min_cluster_size=4, min_samples=1` setting):

| total articles | per-subtopic | HDBSCAN ARI | HDBSCAN clusters found | HDBSCAN noise | KMeans (k=4) ARI |
|---|---|---|---|---|---|
| 28 | 7 | 0.554 | 3 | 2 | 1.0 |
| 20 | 5 | 0.242 | 2 | 1 | 1.0 |
| 12 | 3 | 0.0 | 0 | 12 (all noise) | 1.0 |
| 8 | 2 | 0.364 | 2 | 0 | 1.0 |
| 4 | 1 | 0.0 | 0 | 4 (all noise) | 1.0 |

HDBSCAN's quality drops sharply once the total article count falls below
roughly 20-28 for this 4-subtopic structure, and collapses entirely at 12
and below (it can't find *any* cluster structure once `min_cluster_size=4`
means the whole subsample barely covers one cluster's worth of points).
KMeans stays perfect throughout **because it was given the correct k=4**
— it never has to *discover* the cluster count, only partition. This is
exactly the trade-off the TRD's "KMeans fallback for low article counts"
rationale describes.

**Recommendation: `kmeans_fallback_threshold = 20`** (total article/vector
count below which `cluster()` should use KMeans instead of HDBSCAN). This is
picked at the point where HDBSCAN's ARI has already dropped to a
weak/unreliable 0.24-0.55 in this fixture, before it collapses to 0 at
12 and below — i.e. the fallback should trigger before HDBSCAN is merely
"a bit worse," at the point it's already unreliable. Note this value is
derived from one synthetic fixture with 4 fairly-distinct subtopics; the
real threshold may need retuning once `backend-engineer`/`data-scientist`
have real run data from Task 2.2.3/2.5.1's actual clustering calls (this is
a documented starting point, not a final calibration).

**Open question this fixture cannot resolve, flagged for `backend-engineer`
and `tech-lead`, not decided here:** KMeans requires a `k` up front,
HDBSCAN does not. When `cluster()` falls back to KMeans below the
threshold, where does `k` come from?
- For Task 2.2.3 (subtopic reconciliation over the broad fetch), the
  Subtopic Agent's own LLM-proposed candidate count is a natural `k` hint
  already available in that call path.
- For Task 2.5.1 (per-subtopic topical clustering), there's no equivalent
  upstream candidate count — `cluster()` would need either a caller-supplied
  default `k`, or an estimation step (e.g. a small silhouette-score sweep
  over `k=2..min(5, n-1)`) before falling back to KMeans. This needs a
  concrete decision in Task 2.1.2b/2.2.3b; it is not resolved by this
  prototyping task alone since it's part interface design (what does
  `cluster()`'s signature accept as an optional k-hint?) and part algorithm
  choice.

### 4. Ambiguous/cross-cutting input — qualitative check

Three additional headlines that plausibly span more than one subtopic
("Lawmakers around the world are racing to regulate artificial
intelligence.", etc.) were added to the full 32-point fixture and run
through both algorithms at the recommended settings. HDBSCAN placed 1 of 3
as noise (`-1`) and force-assigned the other 2 into an existing cluster;
KMeans, having no noise concept, force-assigned all 3. This is a real,
if modest, quality difference in HDBSCAN's favor for handling genuinely
ambiguous/cross-cutting input — but it's a soft signal from 3 hand-picked
sentences, not a robust measurement; don't over-read it.

### 5. Real-data spot-check (manual, qualitative — not a measured metric)

As a secondary, non-quantitative sanity check (task option (b)), the 61
English-language article titles from `tests/fixtures/gdelt_doc2_capped_250_climate.json`
(a real Phase 1 GDELT DOC 2.0 fixture, topic "climate") were embedded and
clustered at the recommended settings (`min_cluster_size=4, min_samples=2`).
No ground-truth labels exist for this real set, so this is a manual read,
not an ARI score:

- 4 clusters found, 26/61 titles marked noise (43%).
- One 20-member cluster grouped genuinely climate-disaster-relevant
  headlines together plausibly (heat waves, floods, wildfire-related
  pollen studies, Afghanistan flood search-and-rescue) — this looks like a
  real, coherent topical cluster on manual read.
- Two small 4-member clusters were each exact-duplicate wire-service
  headlines about the same event ("Tropical Storm Bertha ... landfall"),
  correctly grouped together and separated from each other by minor
  headline-text variation between two different phrasings of the same
  story — useful evidence this also functions as a light duplicate/near-
  duplicate detector.
- The noise bucket correctly rejected clearly off-topic GDELT keyword
  false-positives (e.g. "The 25 Greatest Crime Movies of All Time, Ranked",
  "The Lawyer Elite vs. the American People") that matched the "climate"
  query for unrelated reasons — HDBSCAN's noise concept is doing real,
  useful work filtering GDELT's known false-positive-prone keyword search
  here, which KMeans (forced-assignment, no noise concept) would not do.
- One weaker cluster (bus fares, a home appliance product, a car review, a
  "cool roofs" paint marketing piece) grouped together for a less obviously
  coherent reason — plausibly weak shared vocabulary ("smart", "cost",
  "global") rather than real topical relatedness. This is the kind of
  borderline case a `data-scientist` manual spot-check (Story 2.5.1's
  flagged risk item) should keep watching for on real per-subtopic runs.

**This is a manual spot-check of one real fixture, not a validated
benchmark** — stated plainly per the instruction not to overstate v1 rigor.
It supports the chosen `min_cluster_size`/`min_samples` as reasonable
starting defaults on real, noisy, multi-source headline text, but it is not
a substitute for reviewing real per-subtopic clustering output once Task
2.5.1 is wired (which the Phase 2 breakdown already flags as needing
ongoing `data-scientist` spot-checks, not a one-time sign-off).

## Fixture for `backend-engineer`'s Wave 2 unit test

`tests/fixtures/clustering_synthetic_topics.json` contains:
- `sentences`: the 32 synthetic headline strings (no real article full text
  — synthetic, short, never persisted full-text content; consistent with
  the no-full-text-storage rule even though that rule is really about
  Phase 3's `sourcing/fulltext.py`, not Phase 2).
- `true_labels`: ground-truth subtopic index (0-3) per sentence.
- `embeddings`: the actual 384-dim vectors produced by `get_embeddings()`
  (local backend) for each sentence — pre-computed so a unit test doesn't
  need to load a sentence-transformers model at test time.
- `topic_names`, `embedding_model`: metadata for reproducibility.

A deterministic unit test for `clustering/cluster.py` (Task 2.1.2b) can load
this fixture, call `cluster(np.array(embeddings))` with the recommended
settings, and assert the labels achieve at least the ARI recorded above
against `true_labels` (or, more simply, assert the correct number of
non-noise clusters is found) — exercising the "unit test with a fixed
multi-cluster embedding fixture confirms correct HDBSCAN grouping" acceptance
criterion from Task 2.1.2 without needing network/API access at test time.

A second, much smaller subsample of this same fixture (e.g. 8-12 points,
values already computed above) can similarly back the "a fixture below
`kmeans_fallback_threshold` confirms KMeans is used instead" acceptance
criterion.
