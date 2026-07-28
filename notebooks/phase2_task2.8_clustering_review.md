# Phase 2 Task 2.8 — Clustering quality co-sign review

Owner: `data-scientist`. This is the ownership-split review flagged at Task
2.5.1's PR #30 review (backend-engineer correctly declined to self-certify
clustering quality on a single ad hoc spot-check of one subtopic). Phase 2's
overall Done-when requires this co-sign.

Script: `notebooks/phase2_task2.8_clustering_review.py`. Raw output:
`notebooks/phase2_task2.8_clustering_review.log`. Structured results (titles
+ domains only, no article bodies — consistent with the no-full-text-storage
rule, and these are public headlines anyway, not persisted full text):
`notebooks/phase2_task2.8_real_run_samples.json`.

## What this is, and isn't

Real end-to-end runs of `agents/topical_clustering_agent.py::topical_clustering_agent`
(Task 2.5.1, PR #30) against 5 real subtopics via live GDELT + trusted RSS +
Google News backfill sourcing, local `sentence-transformers` embeddings,
current production `Settings.clustering.*` (`hdbscan_min_cluster_size=4`,
`hdbscan_min_samples=1`, `kmeans_fallback_threshold=20`, run against
`master`@`e6388fd`). This is **manual qualitative inspection of 5 real
samples, not a measured benchmark** — there's no ground-truth labeling of
what the "correct" clusters should be for real GDELT/RSS pulls, so verdicts
below are a judgment call on headline coherence, same as the recommendation
doc's own "real-data spot-check" section (§7 Evaluation Strategy names this
gap; not overstating it here).

## Samples

| id | subtopic | lookback | total articles | path | clusters | noise |
|---|---|---|---|---|---|---|
| s1 | OpenAI GPT-5 release | 14d | 32 | HDBSCAN | 4 | 11 (34%) |
| s2 | Russia Ukraine ceasefire negotiations | 14d | 8 | KMeans (k=3, estimated) | 3 | 0 |
| s3 | Federal Reserve interest rate decision | 21d | 11 | KMeans (k=5, estimated) | 5 | 0 |
| s4 | Tesla quarterly earnings | 21d | 39 | HDBSCAN | 2 | 1 (3%) |
| s5 | solar panel manufacturing | 30d | 40 | HDBSCAN | 2 | 22 (55%) |

`kmeans_fallback_threshold=20` fired correctly on s2/s3 (8, 11 articles) and
correctly did not fire on s1/s4/s5 (32, 39, 40 articles) — the threshold
itself is behaving as designed at these article counts.

## Per-sample judgment

**s1 (OpenAI GPT-5 release, HDBSCAN, n=32) — mostly coherent, one weak cluster.**
Cluster 0 (security-incident angle: "unprecedented hack", GPT-based exploit
research), cluster 1 (competitor-benchmark angle: Kimi K3 vs. GPT/Claude),
and cluster 2 (general release-announcement coverage) each read as genuine,
distinct sub-angles on manual read. Cluster 3 is weaker: it merges a
bioweapon-risk-labelling story with an unrelated file-deletion-bug story —
both "OpenAI safety incident" vocabulary, but two different actual stories,
same pattern the recommendation doc flagged as a risk to keep watching for
("weak shared vocabulary rather than real topical relatedness"). Noise rate
(34%) looks appropriate — spot-checked noise items (a "which AI is cheapest"
listicle, a ChatGPT-branded basketball story) are genuinely tangential to
the GPT-5-release story rather than being real signal getting discarded.

**s2 (Russia-Ukraine ceasefire, KMeans k=3, n=8) — coherent given a hard case.**
Cluster 0 ("return to the negotiating table") and cluster 2 (ceasefire-prospect
commentary) each read as genuinely related headlines. Cluster 1 (a UK
parliamentary Russia inquiry + a "shadow envoys" diplomacy piece) is a
plausible but looser pairing — both Russia-diplomacy-adjacent, not the same
sub-story. No noise bucket exists in the KMeans path by construction, so
articles that don't cleanly belong anywhere still get force-assigned; this
is the known, already-documented KMeans/HDBSCAN trade-off, not a new finding.

**s3 (Fed rate decision, KMeans k=5, n=11) — real over-fragmentation, matches the task's named failure mode.**
This is the clearest quality problem in the 5-sample set. Cluster 3 is a
forced singleton (one Japanese-yen article, alone). Clusters 1 and 4 are
near-duplicate in substance — both are "gold price + Fed decision + Middle
East/oil tension" commentary pieces — but landed in two separate clusters
rather than one, purely because the silhouette-driven `_estimate_k` sweep
(`clustering/cluster.py::_estimate_k`, k=2..min(5, n-1)) picked k=5 for only
11 points. This is exactly the task brief's named risk: "KMeans splitting a
tight single-topic set into K groups when K doesn't reflect real structure."

**s4 (Tesla earnings, HDBSCAN, n=39) — coherent split, but one giant catch-all cluster.**
Cluster 0 (8 articles, all Chinese-language financial-press earnings
coverage) is a genuine, tight sub-angle — language/source-driven, not noise.
Cluster 1 (30 articles) is a large catch-all spanning earnings results,
robotaxi-expansion news, stock-price reaction, and "should you buy" investor
advice — all legitimately Tesla-earnings-adjacent at the coarse subtopic
level this agent operates at, so not wrong, but it's under-differentiated:
a single 30-member cluster likely papers over real sub-angle structure that
a later, finer clustering pass (claim-level, Task 2.2/summarization) would
need to recover on its own. Flagging as a limitation of *this* coarse pass,
not a bug — Task 2.5.1's docstring explicitly scopes this as coarse
clustering, finer distinctions are a later stage's job.

**s5 (solar panel manufacturing, HDBSCAN, n=40) — the real problem case: over-noising legitimate on-topic coverage.**
Cluster 0 (13 articles, India solar-manufacturing struggles/reshoring) and
cluster 1 (5 articles, one specific Canadian Solar/CS PowerTech Indiana plant
opening, near-duplicate wire coverage of one event) both read as genuinely
coherent. But noise is 55% (22/40) — and unlike the recommendation doc's
earlier real-data spot-check (climate fixture, where noise correctly
filtered off-topic GDELT keyword false-positives like a "Crime Movies"
listicle), manual inspection of s5's noise bucket shows it's **mostly
legitimately on-topic solar-manufacturing news**: a DYCM panel-manufacturing
plant story, a "PV Module Manufacturers 2026" industry report, a
scrap-material recycling story, a zoning-dispute ruling, a factory-expansion
announcement. These are real signal, not junk, being discarded because they
don't form a dense near-duplicate cluster of 4+ near-identical stories —
each is a distinct single-company/single-event story, and `min_cluster_size=4`
requires 4+ such stories before it will form a cluster core around any of
them. This is the task brief's other named failure mode ("HDBSCAN
over-noise-ing a topic into mostly unclustered singletons") actually
occurring on real data, not just a hypothetical.

## Overall verdict

**Acceptable to co-sign as Done for Phase 2's current scope**, with two real,
specific, fixable-but-not-yet-fixed limitations flagged as follow-up (not
blocking, since Task 2.5.1's own scope is coarse clustering and later stages
are expected to refine further) — not "small-n clustering is inherently
noisy" hand-waving, but two concrete, reproducible-in-principle issues:

1. **KMeans fallback over-fragmentation at low n** (s3): `_estimate_k`'s
   k=2..min(5, n-1) sweep has no floor against degenerate results (a forced
   singleton at n=11, or splitting one coherent story angle into two
   clusters). Flagged fix, not applied by me: cap `max_k` more
   conservatively at low n (e.g. `n < 12 -> max_k = 3` instead of 5), or add
   a post-hoc merge step that folds any resulting singleton cluster into its
   nearest neighbor cluster. Location: `clustering/cluster.py::_estimate_k`.
   This is a proposed direction based on one observed case (s3), not a
   re-tuned, validated threshold — needs a few more low-n real samples
   before treating any specific new cap value as final.

2. **HDBSCAN over-noising topically-diffuse (but legitimate) coverage at
   moderate n** (s5, 55% noise, mostly real on-topic content, not GDELT
   false positives): the per-subtopic scope (Task 2.5.1) is narrower than
   the broad multi-subtopic fetch the original `min_cluster_size=4`
   recommendation was tuned against (4 distinct AI-regulation angles at
   32 points, `notebooks/phase2-clustering-recommendation.md`). A single
   subtopic's own article set can be topically diffuse (many single-company/
   single-event stories) rather than clustering into a few dense groups,
   which is a different data shape than what `min_cluster_size=4` was
   validated on. Flagged direction: re-tune `hdbscan_min_cluster_size`
   specifically for the per-subtopic path (possibly a lower value, e.g. 3,
   or a separate config knob for Task 2.5.1 vs. Task 2.2.3's broad-fetch
   path) against a larger sample of real per-subtopic runs before changing
   the default. Not done here — this is a single real-run observation, not
   a tuned recommendation.

Neither issue is severe enough to block Phase 2's Done-when on its own (both
are within the "descriptive, not perfect" bar this system already accepts
per PRD §7's named eval gap), but both are real quality gaps this review
surfaced beyond the original single-subtopic ad hoc check, and both should be
picked up as explicit follow-up work rather than left implicit.

## Handoff

- No code changes made by me (design/analysis only, per the design-vs-wiring
  split).
- Follow-up candidates for `backend-engineer`/`tech-lead` to schedule:
  (a) a floor/cap fix in `clustering/cluster.py::_estimate_k` for low-n
  degenerate KMeans results, (b) a `data-scientist` re-tune of
  `hdbscan_min_cluster_size` specifically for Task 2.5.1's per-subtopic
  scope, using more real per-subtopic samples than this review's single s5
  case.
