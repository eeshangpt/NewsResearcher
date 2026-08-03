# Zero-subtopic-candidates-at-Gate-1 — threshold review

Owner: `data-scientist`. Design/analysis deliverable, not production code.
Handoff target: `backend-engineer`. Triggered by two live-run failures
("Iraq and WMDs", "Taliban Takeover") that both reached Gate 1 with zero
subtopic candidates when GDELT was down/rate-limited and only RSS
(+ conditional Google News backfill) contributed articles.

Script: `notebooks/phase2_zero_candidates_review.py`.
Raw output: `notebooks/phase2_zero_candidates_sample.json`.

**This is a real-data reproduction against two live topics run once each on
2026-08-02/03, not a golden-dataset eval — PRD §7 names that gap explicitly
for v1.** Every number below is directional evidence from two runs against
whatever GDELT/RSS/Google-News/WHOIS/Tranco/HTTPS state happened to be true
at that moment, not a certified benchmark. Re-running the script later will
produce different absolute article sets and scores (GDELT's block status,
WHOIS results, and Google News's returned articles all vary run to run) but,
based on the mechanism traced below, should show the same qualitative shape.

## Reproduction

`sourcing_agent()` itself needs a live Postgres pool for the reputation
cache, unavailable in this analysis environment, so
`phase2_zero_candidates_review.py` reproduces its call sequence directly
(RSS fetch → backfill-if-thin → dedup → real `reputation/signals.py`
collectors, uncached → real `reputation/scorer.py::score_domain` →
threshold filter → real `clustering/cluster.py` + `agents/subtopic_agent.py`
`reconcile_candidates`), skipping only the Postgres cache layer. GDELT was
confirmed rate-limited/blocked in this environment during this session (429
on every retry, `GDELTError`) — the same failure mode the two live runs hit,
not simulated. No article full text is fetched, persisted, or written to any
artifact by this script — only `url`/`title`/`domain`, per the no-full-text
rule.

Two topics:

- **`iraq_wmd`** (keywords `["Iraq", "WMD"]`, 14-day lookback): RSS returned
  0 matches from the 4-feed trusted set → Google News backfill triggered →
  11 deduped articles. All `unknown` tier (none of Britannica, Lawfare, Iran
  Watch, Japan Today, KAS, Tehran Times are in `trusted_outlets.yaml`).
- **`taliban`** (keywords `["Taliban"]`, 14-day lookback): RSS returned 0
  matches → backfill triggered → 84 deduped articles, a mix of `wire`/
  `major`/`unknown` tier. Included as a **contrast case**: larger pool,
  shows the same cliffs behaving much less destructively at higher n.

## 1. Reputation-score-filter cliff (`min_score_threshold = 0.5`)

| Topic | n articles | pass ≥0.50 | fail | shortfall distribution (failures) |
|---|---|---|---|---|
| `iraq_wmd` | 11 | 2 (18%) | 9 (82%) | 6× exactly 0.0057 below, then 0.0153, 0.036, 0.1172 |
| `taliban` | 84 | 53 (63%) | 31 (37%) | median ≈0.021 below, max 0.080 below |

Counting survivors at alternate thresholds (real scores, both topics):

| threshold | `iraq_wmd` survivors | `taliban` survivors |
|---|---|---|
| 0.60 | 0/11 | 7/84 |
| 0.55 | 0/11 | 12/84 |
| **0.50 (current)** | 2/11 | 53/84 |
| 0.47 | 9/11 | 70/84 |
| **0.45** | 10/11 | 74/84 |
| 0.42 | 10/11 | 84/84 |
| 0.40 | 10/11 | 84/84 |

**Root cause, not just a symptom:** `weight_domain_age + weight_backlink_proxy
+ weight_presence_frequency + weight_legitimacy_flags = 0.30 =
adjustment_bound` (`config.py:49-53`). If every one of the four signals lands
exactly at its own neutral value (0.5 — WHOIS/HTTPS unreachable, domain
absent from the Tranco snapshot, single-source-type presence), the
adjustment is `0.5 * 0.30 = 0.15`, so an `unknown`-tier domain with **zero
negative evidence and zero positive evidence** scores exactly
`0.3 + 0.15 = 0.45`. That is not a corner case in this sample — it is close
to the *modal* outcome: `britannica.com` scored exactly 0.4943 five times
over (near-neutral across the board), and 9 of `taliban`'s 84 domains landed
at exactly 0.4297 (`thekabultribune.com`, a real Afghanistan-focused news
site, appears 9 times, always fully-neutral-scored). `min_score_threshold =
0.5` therefore requires an unverified domain to clear the *all-neutral*
baseline, not merely avoid a negative signal — structurally punishing every
legitimate niche/regional/newly-observed outlet a degraded (GDELT-down)
fetch is disproportionately likely to surface, since RSS-only trusted-tier
coverage is thin by construction (only 4 feeds in `OUTLET_RSS_FEEDS`) and
backfill pulls from the open web.

**Recommendation: add `Settings.reputation.min_score_threshold_degraded =
0.45`**, applied instead of `min_score_threshold` specifically when the
sourcing fetch's GDELT call contributed zero articles (`len(gdelt_articles)
== 0` in `sourcing_agent.py`, already computed and available at the
threshold-filter call site — no new detection logic needed, just branch on
an existing value).

Why a distinct degraded threshold rather than lowering `min_score_threshold`
globally: `min_score_threshold` doing real filtering work in the *healthy*
(GDELT-up) case is untouched by this change — this sample's `taliban` case
alone shows 0.50 already excludes visibly weaker/less-corroborated domains
(31 of them) when there's enough volume to afford being selective;
loosening that blanket-wide risks admitting real noise in the common case to
fix a problem that's specific to the degraded path. Why `0.45` exactly, not
lower: it is the formula's own **all-neutral floor** (`base_score_unknown +
0.5 * adjustment_bound = 0.3 + 0.15`), not an arbitrary pick — it draws the
line at "no active negative signal found," which is the correct bar for a
domain WHOIS/HTTPS/Tranco simply couldn't corroborate either way (the
`min_score_threshold_degraded` name over an unconditional lower absolute
value because this is squarely a "sourcing degraded" condition — flag to
`tech-lead` only if `Settings.sourcing`/`Settings.reputation` split feels
like it needs its own cross-cutting "degraded mode" concept beyond this one
field; as a single new field on `ReputationSettings` this is in-scope for me
to decide). Going lower than 0.45 (e.g. 0.42, which would recover
`iraq_wmd`'s `lawfaremedia.org` at 0.3828... no, 0.3828 stays excluded even
at 0.42 — the next real cliff below 0.45 sits around 0.42-0.43, admitting
domains with a genuine below-neutral signal) is not supported by this
sample; 0.45 is the principled stopping point.

## 2. Reconciliation-match-threshold cliff (`reconciliation_match_threshold = 0.60`)

| Topic | n articles fed to `cluster()` | clusters found | candidates dropped @0.60 | reconciled @0.60 |
|---|---|---|---|---|
| `iraq_wmd` | 2 (after §1's cliff) | 2 | 3 of 4 | 1 |
| `taliban` | 53 (after §1's cliff) | 2 | 0 of 4 | 3 |

`taliban`'s candidate→cluster-centroid similarities (real, all 4 candidates
survive): 0.745, 0.703, 0.697, 0.664 — comfortably above 0.60, consistent
with `phase2-reconciliation-design.md`'s original fixture-derived gap. This
sample gives **no evidence 0.60 is wrong once the pool is a few dozen
articles.**

`iraq_wmd`'s candidates: 0.658 (survives), 0.477, 0.438, 0.323 (all
dropped). But `iraq_wmd` only had **2 articles** reach clustering at all —
`cluster()`'s KMeans fallback with `k_hint=4` on `n=2` degrades `k` to 2
(capped at `n`), producing two singleton "clusters." No candidate threshold
rescues 4 genuinely distinct subtopics from 2 data points — the article
starvation is entirely upstream, from §1's reputation cliff (11→2), not
from this threshold. **This sample's own evidence says the reconciliation
threshold is a secondary lever for this failure mode, not the primary
one** — say so plainly rather than overstating its role.

That said, a real constraint on how far it's safe to lower: the design
doc's `reconciliation_drop.json` fixture found one concrete false-positive
candidate — "AI chip export control negotiations" scoring 0.525 against an
unrelated cluster — that the doc explicitly requires to keep failing.
Lowering `reconciliation_match_threshold` below ~0.53 would flip that
documented case from correctly-dropped to incorrectly-admitted.

**Recommendation: `reconciliation_match_threshold` relaxes to `0.55`**
(down from 0.60) specifically when the article pool fed to `cluster()` is
below `Settings.clustering.kmeans_fallback_threshold` (20 — already the
codebase's own signal that clustering is in its reduced-reliability
small-`n` regime, reused rather than inventing a second n-based cutoff).
`0.55` sits strictly above the one documented false-positive (0.525) so it
doesn't reopen that fixture's failure, and gives real headroom for the
`0.55-0.60` band where a small, noisy cluster's genuine centroid similarity
may sit slightly under 0.60 (not observed in this sample's two data points,
but consistent with small-`n` centroids being noisier, per the same
degradation logic `cluster.py`'s own docstring already applies to
`kmeans_fallback_threshold`). Flagged plainly: in this sample it changes
zero outcomes for either topic (nothing sits in the 0.55-0.60 band) — it is
a defensible safety margin for the general small-pool case, not something
this specific evidence proves necessary. Do not go lower than 0.53 without
a new fixture showing a lower genuine-match floor; going lower risks
exactly the false-positive class the design doc already caught once.

## 3. Structural fallback: surface raw unlabeled clusters when reconciliation returns zero?

**Yes, build it.** Reasoning:

- Even with both threshold changes above, a real run can still hit zero
  reconciled subtopics — the degenerate extreme in this sample (`iraq_wmd`'s
  2-article pool) has no threshold rescue; the underlying cluster structure
  itself is too thin to produce multiple genuine subtopics no matter what a
  candidate-matching threshold is set to.
- Gate 1 is already a human-approval gate (per PRD/TRD's two-gate design) —
  presenting raw, unlabeled clusters ("Cluster 2 (4 articles): [sample
  headlines]") for an operator to manually label/approve/reject is squarely
  within the existing human-in-the-loop interaction model, not a new UX
  paradigm. An empty Gate 1 (today's failure mode) gives the operator
  nothing to act on; an unlabeled-cluster Gate 1 gives them something real
  to work with, sourced from actual article content instead of nothing.
- Scope it precisely so it doesn't erode the existing, deliberate "unclaimed
  cluster is dropped" design (`phase2-reconciliation-design.md`'s explicit
  call): only fall back to raw clusters when **every** cluster
  `cluster()` found is unclaimed (`reconcile_candidates`'s `reconciled` list
  is empty *and* `cluster_ids` is non-empty) — i.e., last-resort only, not a
  replacement for the LLM-labeled path in the normal case where at least one
  candidate matches something.
- Each surfaced fallback cluster already has everything Task 2.2.4's ranking
  needs (a centroid, an article count) — it can flow through
  `rank_and_cap_subtopics` unchanged with a generic placeholder label (e.g.
  `f"Cluster {cluster_id} ({n_members} articles, unlabeled)"`) instead of an
  LLM-authored one.
- Left to `tech-lead`/`backend-engineer`, not decided here: whether Gate 1's
  UI should visually distinguish "LLM-labeled" from "auto-clustered,
  unlabeled" subtopics — that's a display/architecture call, not a modeling
  one.

## Summary of exact recommended values

| Setting | Current | Recommended | Trigger condition |
|---|---|---|---|
| `Settings.reputation.min_score_threshold_degraded` (new field) | n/a | **0.45** | `len(gdelt_articles) == 0` for that fetch |
| `Settings.clustering.reconciliation_match_threshold` (relaxed variant) | 0.60 | **0.55** | article pool fed to `cluster()` < `kmeans_fallback_threshold` (20) |
| Zero-reconciled structural fallback | none (drops silently) | **build it** | `reconcile_candidates` returns empty `reconciled` but non-empty `cluster_ids` |

## Caveats, stated plainly

- Two topics, one run each, on one day (2026-08-02/03) in one environment.
  This is a manual spot-check against real data, not a measured/repeated
  eval and not a golden dataset (none exists for v1, per PRD §7).
- This analysis environment's own WHOIS/HTTPS network access was itself
  degraded (SSL cert-verify failures, WHOIS timeouts on several domains
  during the run — visible in the script's stderr) — some neutral-signal
  fallbacks in the sample are therefore genuinely environment-driven, not
  purely representative of a production deployment's WHOIS/HTTPS success
  rate. This doesn't undermine the recommendation: NFR-3 already documents
  WHOIS as "frequently rate-limited/blocked for bulk automated lookups" in
  production too, so a meaningful neutral-fallback rate is a real,
  recurring production condition, not an artifact unique to this sandbox —
  but the *exact* pass/fail counts above should be read as directional, not
  as a precise production failure rate.
- `min_score_threshold_degraded = 0.45` and the relaxed
  `reconciliation_match_threshold = 0.55` are both derived from either a
  formula-level anchor (the all-neutral floor) or a previously-documented
  fixture boundary (the 0.525 false positive) — not fit to these two
  samples' specific numbers — but neither has been validated against a
  larger or repeated sample. Revisit both once real Gate-1 runs accumulate
  under the degraded path, same caveat `phase2-reconciliation-design.md`
  already states for its own thresholds.
