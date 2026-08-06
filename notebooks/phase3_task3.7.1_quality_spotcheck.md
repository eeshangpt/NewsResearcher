# Task 3.7.1 — Phase 3 manual quality spot-check (real end-to-end slice)

**Date:** 2026-08-06
**Reviewer:** data-scientist
**Scope (EXECUTION_PLAN.md line 752-753):** manual spot-check of several real claim
clusters on a Gate-2-cleared subtopic for correct assert/omit membership and
summary plausibility. Any failure filed against Task 3.3.1 (clustering) or
3.6.1 (summarization), not "Phase 3" generally.

This is a manual spot-check on one real subtopic (6 clusters inspected), not a
golden-dataset eval — no automated eval exists in v1 (PRD §7's named gap).
Findings below are read directly off real LLM output, not inferred.

## What was run, and why it's a real slice (not the 3.6.1a fixture corpus)

Ran the actual production code end-to-end against genuinely fresh real-world
data, not `tests/fixtures/phase3_claim_clustering_corpus.json` (already used
by 3.6.1a):

1. `sourcing/rss.py::fetch_trusted_rss` (real network) — the 4 direct-outlet
   RSS feeds (bbc.com, theguardian.com, npr.org, aljazeera.com).
2. `sourcing/fulltext.py::fetch_fulltext_for_cluster` (real network,
   in-memory only, never written to disk).
3. `agents/claim_extraction_agent.py::extract_claims` — real `gpt-4.1-mini`
   calls via `get_chat_model("claim_extraction")`, traced.
4. `clustering/claim_clustering.py::cluster_claims` — real local
   sentence-transformers embeddings + HDBSCAN, `Settings.clustering.claim_*`
   hyperparameters unchanged from Task 3.3.1a/b.
5. `agents/sentiment.py::score_claim_sentiment` — real VADER scoring.
6. `persistence/claim_clusters.py::write_claim_clusters` — real local
   Postgres (`NEWSRESEARCH_DATABASE_URL`), subtopic_id
   `task371-leipzig-drone`.
7. `agents/summarization_agent.py::summarize_cluster` — real `gpt-4.1-mini`
   (or configured summarization-stage model) calls, traced, writing
   `claim_clusters.summary` back.

### Two real, environment-specific substitutions (documented, not hidden)

- **GDELT stubbed to its documented soft-fail path.** GDELT DOC 2.0 is
  IP-blocked from this environment (known issue #101): every real request
  returned HTTP 429, confirmed by letting the real backoff run to exhaustion
  (5 retries, exponential, ~95s) with zero success, then confirmed again on a
  minimal 1-day-window query — same result. `sourcing_agent`'s own
  `except GDELTError` soft-fail path was exercised for real by monkeypatching
  `gdelt.fetch` to raise `GDELTError` immediately, rather than waiting out a
  backoff that cannot succeed in this network. This is the exact code path
  that runs in production whenever GDELT is genuinely down or rate-limited —
  not a fake stand-in for it.
- **WHOIS (`reputation/signals.get_domain_age_years`) stubbed to `None`.**
  Port 43 is unreachable from this sandbox (silent firewall drop, not a
  timeout `python-whois` respects) — same effect as `signals.py`'s own
  documented soft-fail contract, stubbed directly since domain-age scoring is
  out of this task's scope (claim clustering/summarization quality).

### One real, out-of-scope finding surfaced along the way (filed against neither 3.3.1 nor 3.6.1)

`sourcing/google_news_backfill.py`'s article URLs are Google's own
redirect-shell pages (`news.google.com/rss/articles/...`), not the
publisher's real URL — confirmed directly: `trafilatura.fetch_url()` on 12
such URLs returned real HTTP 200 responses (580KB Google News app-shell HTML
each) but extracted **zero** body text from all 12, because there is no
article content to extract (Google resolves the real article client-side via
JS, not an HTTP redirect `httpx`/`trafilatura` can follow). This means any
subtopic sourced solely through Google News backfill (which is what happens
whenever GDELT is down, per the above) currently cannot get full text at all,
starving claim extraction. This sits at the `sourcing/google_news_backfill.py`
/ `sourcing/fulltext.py` boundary (Task 3.1.1/Story 1.8 territory) — **not
Task 3.3.1 or 3.6.1**, since it never reaches clustering or summarization.
Flagging for `tech-lead`/`backend-engineer` triage, not fixed here.

To get real, fulltext-fetchable articles, the spot-check subtopic instead
came from the 4 direct-outlet RSS feeds' live current coverage, matched
cross-outlet on real title overlap (`rapidfuzz` token-set ratio) rather than
GDELT/backfill: the Leipzig/Halle Airport drone-and-explosives incident,
independently covered by `bbc.com`, `theguardian.com` (2 articles), and
`npr.org`. Smaller than a typical Gate-2 cluster (3 domains, 4 articles vs.
`hdbscan_min_cluster_size=4`'s usual 4+ domains) but every URL and every
claim below is real.

## Pipeline output

- 4 articles, 127 claims extracted total (25/43/35/24 per article).
- Claim clustering: 31 clusters, 38 noise claims (`Settings.clustering.
  claim_min_cluster_size=2`, `claim_min_samples=1`, unchanged from 3.3.1b).
- All 31 clusters persisted to real Postgres under subtopic_id
  `task371-leipzig-drone`.
- 6 largest (most-corroborated) clusters summarized via real LLM calls.

## Per-cluster review

### Cluster `task371-leipzig-drone:29` — core discovery fact
**Verdict: clustering correct, summary has a real faithfulness bug (3.6.1).**

Claims (10, across all 4 articles):
- [bbc] "A drone carrying an explosive device has been found at Leipzig/Halle Airport in Germany."
- [bbc] "The explosive-laden drone was spotted close to parked Antonov planes..."
- [guardian-1] "A small drone carrying explosives was discovered at Leipzig airport on Wednesday morning."
- [guardian-1] "Investigators said a drone carrying an unknown explosive device was spotted by an airport employee near the south runway..."
- [guardian-1] "The southern runway ... remained closed after the incident."
- [guardian-2] "An armed drone was found at Leipzig airport on Wednesday."
- [guardian-2] "The drone was carrying an unknown explosive device when discovered in the secure cargo flight operations area..."
- [guardian-2] "Police discovered as much as 800 grams of explosive on the drone..."
- [npr] "A drone with explosives was found at Germany's Leipzig/Halle Airport and the device was defused."
- [npr] "An airport employee discovered a drone near the airport's south runway with an unknown explosive device."

**Assert/omit membership: correct.** All 4 articles genuinely report the same
core fact (drone + explosive found at the airport); this is real cross-source
agreement, not embedding noise pulling together unrelated claims.

**Summary:** "A drone carrying an explosive device was discovered near the
southern runway of Leipzig/Halle Airport in Germany, specifically in the
secure cargo flight operations area, according to **six sources**. One
source stated that an airport employee spotted the drone, and the device was
defused after discovery. One report noted that police found as much as 800
grams of explosives on the drone. The southern runway remained closed
following the incident, according to two sources."

**Bug:** "according to six sources" is wrong — there are only **4** distinct
articles/domains in this cluster (bbc, theguardian x2, npr), not six. The
model appears to have counted the 10 individual claim *lines* for the general
core-fact assertion rather than distinct source *articles* (bbc contributes 2
lines, guardian-1 contributes 3, guardian-2 contributes 3, npr contributes
2 — none of those groupings sums cleanly to "six" either, so it's not even a
consistent line-count; it's simply miscounted). This directly violates
`summarization.txt` prompt rule 5 ("state how many of the sources listed
actually assert each fact ... do not imply broader agreement than the claims
listed show"). **Filed against Task 3.6.1.**

### Cluster `task371-leipzig-drone:14` — second object/collision
**Verdict: clustering correct, same summary bug recurs.**

5 claims across all 4 articles, all independently describing a second
unidentified object colliding with a DHL cargo plane after it aborted
landing. Real agreement, correct clustering.

**Summary:** "...this is stated as a fact or apparent event by **five
sources**." Again wrong — only 4 distinct articles in this cluster (5 claim
lines, since bbc contributes 1 and guardian-1 contributes 2). Same
miscounting pattern as cluster 29. **Filed against Task 3.6.1** — this is now
a second, independent occurrence of the same bug class within one run, not
a one-off.

### Cluster `task371-leipzig-drone:17` — airport's strategic role
**Verdict: clustering correct, same summary bug recurs a third time.**

5 claims across 3 asserting articles (bbc, guardian-1, guardian-2; npr omits
— correctly, npr's article never mentions NATO/Antonov background). Real
agreement on background fact, correctly separated from npr's omission.

**Summary:** "...serves as the European base for Ukraine's Antonov Airlines
... according to **three sources**" (correct — 3 articles) "The airport is
also used by the German military and NATO allies for the transport of
military goods, as reported by **four sources**" (wrong — this cluster only
has 3 asserting articles total; there is no 4th source to cite). Within the
same summary, one count is right and the adjacent one is wrong, confirming
this isn't a fixed offset-by-N bug but a genuine per-fact miscount. **Filed
against Task 3.6.1.**

### Cluster `task371-leipzig-drone:23` — detonator removed / faulty detonator
**Verdict: clustering correct, summary faithful and well-calibrated (positive result).**

3 claims, 3 asserting articles (bbc, guardian-2, npr; guardian-1 omits —
correct, guardian-1's article doesn't mention the detonator).

**Summary:** "Police removed the detonator ... as reported by **two
sources**" — correct (bbc + npr both say this; guardian-2 doesn't). "One
source stated that the bomb did not explode because the detonator was
faulty" — correct, only guardian-2 says this. Also correctly keeps this as a
*distinct* fact rather than blending it into the "detonator removed"
statement (prompt rule 4). **No issue.**

### Cluster `task371-leipzig-drone:27` — runway reopening timeline
**Verdict: clustering correct, summary faithful (positive result).**

5 claims, 3 asserting articles (bbc, guardian-1, npr; guardian-2 omits —
correct). Summary correctly attributes "southern runway remains closed" to
one source (bbc only) and separately notes the ~2-hour northern-runway
reopening timeline without overclaiming corroboration counts it can't
support. **No issue.**

### Cluster `task371-leipzig-drone:9` — "hybrid threat" quote attribution
**Verdict: clustering correct, summary faithful and well-calibrated (positive result).**

3 claims, 3 asserting articles (bbc, guardian-1, npr; guardian-2 omits —
correct, guardian-2's op-ed doesn't quote Dobrindt this way).

**Summary:** "...according to **three sources**. **Two sources** explicitly
referred to it as a 'professional, hybrid threat scenario,' while **one
source** described it as a 'hybrid attack scenario.'" All three counts
verified exactly correct against the claims (bbc + guardian-1 use "professional
hybrid threat scenario"; npr alone uses "hybrid attack scenario"). This also
correctly preserves the wording distinction per prompt rule 6 rather than
flattening it. **No issue.**

## Overall verdict

**Claim clustering (Task 3.3.1): PASS.** All 6 inspected clusters' assert/omit
membership reflects genuine semantic agreement/disagreement on the same
underlying fact, not embedding noise pulling together unrelated claims. Every
omission checked (npr omitting the NATO/Antonov background, guardian-1
omitting the detonator claims, guardian-2 omitting the "hybrid threat" quote)
is a real, correct omission — that article genuinely doesn't contain the
claim, not a clustering miss.

**Summarization (Task 3.6.1): FAIL on source-count faithfulness, in 3 of 6
clusters (29, 14, 17).** Content coverage itself is faithful in all 6 (no
invented facts, no dropped distinct facts, correct handling of
certainty/attribution distinctions where present) — the specific, recurring
failure is prompt rule 5 ("state how many of the sources listed actually
assert each fact"): the model inflates or otherwise miscounts the number of
distinct asserting articles in just over half the reviewed clusters, which
directly matters for this system's stated purpose (a reader comparing
cross-source corroboration should never be told "six sources" when the real
number is four). Recommend `data-scientist` follow-up: tighten
`llm/prompts/summarization.txt` rule 5 to explicitly instruct counting
**distinct listed source domains**, not claim lines (e.g. enumerate the
distinct domains present in the input claims before writing the count) —
this is a prompt-design fix, to be handed to `backend-engineer` once
iterated on, not done inline in this spot-check.

**Out-of-scope finding (neither 3.3.1 nor 3.6.1):** Google News backfill
URLs are not fulltext-fetchable by `sourcing/fulltext.py` (confirmed on 12
real URLs, 0/12 extracted). Any subtopic sourced only via backfill (e.g.
whenever GDELT is down) currently gets zero claims. Flagging to
`tech-lead`/`backend-engineer`, not filed against Phase 3's clustering or
summarization scope.

## Langfuse trace check (Task 3.7.2 bonus mechanical check — pass/fail only, not this task's full report)

**Pass.** Queried the local Langfuse API (`localhost:3000`) directly and
confirmed 18 real traces tagged `run_id:task-3.7.1-spotcheck` from this run:
12 `claim_extraction`-stage and 6 `summarization`-stage traces, each with
full input/output, token usage, cost, and latency recorded (e.g. one
`summarization` trace: `gpt-4.1-mini-2025-04-14`, 618 prompt tokens, 56
completion tokens, cost `$0.0003368`, latency `2.24s`). Both claim-extraction
and summarization LLM calls are visible and inspectable in Langfuse, matching
their `get_langfuse_callback_handler`/`trace_metadata` wiring.

## Artifacts

- `pipeline_slice.py` (this branch) — the real end-to-end script run for
  this spot-check. Not production code; a one-off analysis script, left on
  `feat/datascientist` for reproducibility. Full article text was held only
  in local variables during the run, never written to disk.
- Real Postgres rows persisted under `subtopic_id = 'task371-leipzig-drone'`
  in the local `NEWSRESEARCH_DATABASE_URL` database (31 `claim_clusters`
  rows, their `claim_cluster_articles` rows, 6 with `summary` populated).
