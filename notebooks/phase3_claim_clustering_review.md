# Task 3.3.1a — claim-text clustering hyperparameter recommendation

Owner: `data-scientist`. Design/analysis deliverable, not production code.
Handoff target: `backend-engineer`, Task 3.3.1b (wire into `Settings` +
`clustering/cluster.py`'s existing, unchanged `cluster()` per tech-lead's
architecture decision — dedicated `Settings.clustering.claim_min_cluster_size`/
`claim_min_samples` fields, not shared with Phase 2's article-level
`hdbscan_min_cluster_size`/`min_samples`).

Scripts: `notebooks/phase3_claim_clustering_fetch.py` (real corpus
collection), `notebooks/phase3_claim_clustering_eval.py` (candidate
same-fact pair generation), `notebooks/phase3_claim_clustering_sweep.py`
(ARI + pairwise sweep, this doc's numbers).
Data: `notebooks/phase3_claim_clustering_corpus.json` (6 articles / 120
claims, `claim_text` + metadata only, no article body text — no-full-text-
storage rule), `notebooks/phase3_claim_pairs_openai_gpt5_release.json`
(cosine-similarity candidate same-fact pairs), `notebooks/phase3_claim_clustering_sweep_results.json`
(full sweep + subsample sweep raw results).

## Real-data limitation, stated plainly

The task aimed for several varied real subtopics. GDELT hit a sustained
IP-level rate-limit cooldown mid-session (`GDELTError`, all 5 retries
exhausted on two of three attempted subtopics — not ordinary transient
429s, a longer block), and a subsequent RSS-fallback run also stalled on an
apparent unresponsive `trafilatura.fetch_url()` call (that function has no
explicit timeout — a real gap in `sourcing/fulltext.py`, flagged as a
follow-up, not fixed here since it's production wiring). Net result: **one
real subtopic** (`openai_gpt5_release` query, which actually surfaced a
DeepSeek-model-cost story — 6 articles, 120 claims), not the several
originally intended. This is a real limitation on this recommendation's
generalizability, not glossed over: the numbers below are grounded in real
extracted claims and a real manual same-fact ground truth, but from one
topic/story shape (a wire-syndicated cost/benchmark story with 3 near-
duplicate outlets), not a representative sample across subtopic types.
Re-running this same script against 2-3 more varied real subtopics once
GDELT's cooldown clears (or via a fixed-timeout `fetch_fulltext`) would
meaningfully strengthen this before treating the recommended values as
final rather than a strong starting point.

## Ground truth: real same-fact claim pairs, not subtopic identity

Task 3.3.1's clustering step operates on one subtopic's full claim pool, and
needs to group claims that assert **the same underlying fact** across
different articles/outlets (so Bias & Framing can compare agreement/
disagreement) — not to recover subtopic identity, which is already fixed for
this input. Ground truth was built accordingly: claim texts were pairwise
cosine-compared; pairs at similarity ≥0.90 were auto-accepted as same-fact
(spot-checked: every ≥0.90 pair inspected was a genuine same-fact paraphrase
or near-identical wire rewrite); pairs in the 0.80–0.90 band were manually
read one at a time — 10 more genuine same-fact pairs confirmed (an added
subordinate clause/extra detail, same core assertion), and several rejected
as false positives despite high cosine similarity (e.g. two *different*
named-benchmark scores for the same model, or two different products'
different price cuts — shared vocabulary, not shared assertion). Below 0.80,
no further pairs were reviewed. This produced **78 true clusters over 120
claims: 21 multi-claim same-fact groups (mostly size 2-3, three
near-duplicate wire articles produced most of the overlap) and 57 true
singletons** (facts only one article stated). This is a real, manually
confirmed ground truth from actual article overlap, not a golden dataset —
one data-scientist's same-fact judgment call per pair, stated as such per
the v1 eval-rigor rule.

## Why ARI was replaced with pairwise recall/precision

Reused Phase 2's ARI methodology first, and it produced uninterpretable
low values (0.08–0.22 across every setting tested) that don't reflect actual
quality — because Adjusted Rand Index's pairwise bookkeeping treats every
pair landing in HDBSCAN's single `-1` noise sentinel as "assigned to the
same predicted cluster," which is wrong when the ground truth is dominated
by true singletons (57 of 78): two *unrelated* true-singleton claims both
correctly marked noise (i.e., correctly *not* merged with anything) get
penalized by ARI as if they should have been split into different clusters
that don't exist. This is a real methodology finding worth surfacing, not a
detail to bury: **ARI is the wrong metric whenever ground truth has many
true singletons and the clustering method has a noise concept** (exactly
this task's actual data shape, unlike Phase 2's 4-balanced-cluster synthetic
fixture). Replaced with direct pairwise metrics that measure what actually
matters for Task 3.3.1:
- **same-fact recall**: of all true same-fact pairs, what fraction land in
  the same non-noise predicted cluster?
- **false-merge precision**: of all pairs sharing a non-noise predicted
  cluster, what fraction are true same-fact pairs (not two distinct facts
  wrongly merged)?

## Recommendation: `claim_min_cluster_size = 2`, `claim_min_samples = 1`

Full-120-claim HDBSCAN sweep (`min_cluster_size` 2-6, `min_samples` 1-3):

| min_cluster_size | min_samples | clusters found | noise | same-fact recall | false-merge precision | F1 |
|---|---|---|---|---|---|---|
| **2** | **1 or 2** | **35** | **27** | **0.829** | **0.674** | **0.744 (best)** |
| 3 | 1 or 2 | 22 | 29 | 0.986 | 0.369 | 0.539 |
| 3 | 3 | 19 | 44 | 0.971 | 0.486 | 0.649 |
| 4 | 1-2 | 11 | 32 | 0.900 | 0.192 | 0.318 |
| 5-6 | 1-3 | 9-10 | 32-48 | 0.886-0.900 | 0.173-0.238 | 0.291-0.375 |

`min_cluster_size=2` is the clear winner by F1 — the opposite direction from
Phase 2's article-level finding (`min_cluster_size=4`), confirming the task's
hypothesis that claim granularity needs different knobs, not shared ones.
The reason is structural, not a fixture quirk: real same-fact claim
duplication comes in pairs/triples (2-3 outlets restating one fact), never
the 4+-member dense cores Phase 2's `min_cluster_size=4` was tuned to find
in article-title space. Requiring `min_cluster_size≥3` forces genuine
2-member same-fact pairs to be discarded as noise (recall stays high because
they're *correctly not merged with the wrong thing*, but real signal is
lost) — and, more damningly, `min_cluster_size≥4` also *drops precision*
sharply (0.192-0.369), because with too few genuine 4+-member cores to
anchor clusters, HDBSCAN starts pulling in adjacent-but-distinct claims to
pad out required cluster size, causing false merges. `min_samples=1` vs. `2`
were tied at every `min_cluster_size` tested on this fixture (same tie
Phase 2 saw); `min_samples=1` is recommended for consistency with the
production `hdbscan_min_samples=1` value already locked in against the real
`hdbscan` package.

**Real limitation, not hidden**: even at the best setting, false-merge
precision is 0.674, not high — roughly 1 in 3 claims sharing a predicted
cluster are a wrong merge of two distinct facts. This is a genuine quality
ceiling on this one real sample, not a tuning failure; flagged as a known
limitation for the Bias & Framing agent's design (Task 3.4.1a) to account
for (e.g. treating clusters as "candidate same-fact groups" rather than
ground truth, or requiring a same-fact merge to also share `subject`/
`attributed_source` overlap as a secondary check) rather than something a
different `min_cluster_size` value fixes.

## `kmeans_fallback_threshold`: needs a dedicated, higher claim variant — flagged, not resolved here

Subsample sweep at the recommended setting (`min_cluster_size=2,
min_samples=1`), progressively fewer claims:

| n claims | true k | HDBSCAN clusters found | noise | HDBSCAN same-fact recall | HDBSCAN false-merge precision | KMeans (true k) same-fact recall | KMeans false-merge precision |
|---|---|---|---|---|---|---|---|
| 120 | 78 | 35 | 27 | 0.829 | 0.674 | 0.800 | 0.933 |
| 90 | 59 | 25 | 20 | 0.867 | 0.481 | 0.867 | 0.951 |
| 60 | 49 | 14 | 16 | 1.000 | 0.278 | 1.000 | 1.000 |
| 39 | 34 | 9 | 9 | 1.000 | 0.122 | 0.800 | 0.800 |
| 30 | 26 | 6 | 11 | 1.000 | 0.185 | 0.800 | 0.800 |
| 18 | 16 | 6 | 5 | 1.000 | 0.250 | 1.000 | 1.000 |

HDBSCAN's false-merge precision degrades sharply and monotonically as claim
count drops below the full 120 (0.674 → 0.481 → 0.278 → 0.122), the same
"HDBSCAN gets worse at low n" shape Phase 2 found for articles — but here it
starts degrading from a moderate baseline (0.674, not phase2's article-level
peak of ~0.78 ARI) and the existing `kmeans_fallback_threshold=20` (a raw
vector count, applies identically regardless of what's being clustered)
would never trigger anywhere in this range — every row above is already
≥18, and real per-subtopic claim volume is virtually always well above 20
(single articles alone produced 13-26 claims each in this corpus; a typical
multi-article subtopic pool is 60-200+). **This is exactly tech-lead's
flagged risk borne out on real data**: the shared numeric threshold is unit-
blind (article count vs. claim count), so a value tuned for "articles below
20 are too few for HDBSCAN" does not carry any meaning for claims, where the
"too few" zone is structurally much higher.

**Finding, not a final recommendation**: a dedicated
`claim_kmeans_fallback_threshold` is needed and should sit meaningfully
higher than 20 — this one sample's data doesn't show precision reaching a
clearly "good" plateau even at n=120 (0.674 is the best observed, not a
clean asymptote), so I can't responsibly hand off one precise number as
confidently as Phase 2's `kmeans_fallback_threshold=20` (which had a clear
collapse-to-zero cliff at n≤12). What this sample does support: the
fallback point should be well above 60-90 claims, not 20, and possibly the
per-subtopic path should lean on KMeans more often than the article-level
path does. **Real, unresolved tension to flag to `backend-engineer`/
`tech-lead`, not silently deferred**: KMeans needs a `k` up front, and — same
open question Phase 2's recommendation doc flagged for Task 2.5.1's
per-subtopic article clustering and never got resolved — there is still no
upstream candidate-count hint for claim-level clustering either. This
sample's KMeans numbers above used the *true* k as an oracle (same
methodology Phase 2 used), which is not available in production; falling
back to `cluster.py`'s existing `_estimate_k` silhouette sweep for claims
is untested here and inherits Phase 2's own flagged low-n degenerate-k risk
(Task 2.8's finding). Recommend `backend-engineer`/`tech-lead` treat a
concrete `claim_kmeans_fallback_threshold` value as still open pending 2-3
more real varied-subtopic samples, rather than picking one number off this
single sample.

## Summary handoff

| Setting | Recommendation | Confidence |
|---|---|---|
| `Settings.clustering.claim_min_cluster_size` | **2** | Moderate — clear, large margin over 3-6 on real data, but single-subtopic sample |
| `Settings.clustering.claim_min_samples` | **1** | Moderate — tied with 2 on this data, matches production `hdbscan_min_samples` convention |
| `Settings.clustering.claim_kmeans_fallback_threshold` (new, dedicated) | **Needs to exist and be materially higher than the article-level 20** — exact value not confidently determined from one sample; recommend backend-engineer treat as a follow-up `data-scientist` task once 2-3 more real subtopics are available, not block Task 3.3.1b's wiring on it (start with a conservative placeholder, e.g. 40, and flag it as provisional in the `Settings` docstring the same way `distinctiveness_volume_weight` is flagged as an unvalidated starting point) | Low — flagged, not resolved |

No code changes made to `clustering/cluster.py` or `config.py` (design-only,
per the design/wiring split) — `backend-engineer` adds the two dedicated
`claim_*` fields and reuses `cluster()` unchanged, per tech-lead's
architecture decision.