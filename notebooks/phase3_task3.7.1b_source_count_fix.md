# Task 3.7.1b — fix summarization source-count miscounting (issue #114)

Owner: `data-scientist`. Design/analysis deliverable, no production wiring
(that's `backend-engineer`'s Task 3.6.1b lineage, when picked up).

## Bug

Found by Task 3.7.1's real spot-check
(`notebooks/phase3_task3.7.1_quality_spotcheck.md`, commit `a4c93b4`): 3 of 6
real inspected clusters had `summarization.txt` rule 5 ("state how many of
the sources listed actually assert each fact") wrong — e.g. "according to
six sources" for a cluster with only 4 distinct asserting domains. Root cause
diagnosed there: the model was counting individual claim *lines* passed into
the `{claims}` block, not distinct source *domains* — and not even
consistently (one summary had a correct count and a wrong count for two
different facts in the same output).

## Method

Real reproduction, no mocked output — same rigor as 3.6.1a/3.7.1:

- The exact 6 clusters from 3.7.1 (`task371-leipzig-drone:29/14/17/23/27/9`)
  still existed in the real local Postgres (`NEWSRESEARCH_DATABASE_URL`), so
  claims were read back via the real
  `persistence/claim_clusters.py::read_cluster_article_relations`, not
  re-typed from the markdown.
- Summaries generated via real `get_chat_model("summarization")` calls
  (`gpt-4.1-mini`), through `llm/prompts/summarization.txt`'s exact
  production template loading path.
- Old-prompt baseline reproduced with rule 5's original wording (spliced in
  for comparison only, never written back to the prompt file) to confirm the
  bug still reproduces in this environment before trusting the fix.
- Re-ran the new prompt against 3.6.1a's original 8-cluster fixture sample
  (`notebooks/phase3_summarization_review.py`'s `sample_labels`) to check for
  regressions on the prior validation set.
- Script: `notebooks/phase3_task3.7.1b_source_count_fix.py`. Raw outputs:
  `notebooks/phase3_task3.7.1b_db_results.json`,
  `notebooks/phase3_task3.7.1b_361a_regression.json`.
- No article full text touched anywhere — only `claim_text` + `domain`
  strings, same scope the fixture and prior notebooks already used.
- This is a manual read of real LLM output against real per-domain ground
  truth pulled from Postgres, not a scored/golden-dataset eval (PRD §7's
  named v1 gap).

## Iteration

Three passes, each driven by a concrete observed failure on real data —
not a single blind guess:

**Iteration 1** — told the model to count distinct domains, not claim lines,
before stating a count. Fixed the original over-counting bug on clusters 29,
14, 17 (all 3 now stated the correct total distinct-domain count). But
introduced a new problem: several outputs collapsed to *one blanket count for
the whole summary* instead of per-fact counts (e.g. cluster 27's "This
information is reported by three distinct sources" applied to a paragraph
where the initial "both runways closed" fact is actually NPR-only — a real
precision loss versus the old prompt's per-fact breakdown, even though the
old prompt's per-fact breakdown was itself sometimes wrong).

**Iteration 2** — added "give a source count separately for each distinct
fact, never one blanket count for the whole summary." This mostly restored
per-fact granularity, but on one real repeated run cluster 14 produced
`"(Five sources: bbc.com, theguardian.com, npr.org)"` — a numeral ("Five")
contradicting its own named list (3 domains), and matching exactly the raw
claim-line count for that cluster (5 lines: bbc×1, guardian×3, npr×1). This
proved the model still leaks the line-count instinct even while correctly
enumerating domains in the same sentence.

**Iteration 3 (final, shipped)** — told the model to *prefer naming domains
directly* ("according to BBC and NPR") over stating a bare number, and if it
does state a number, that number must exactly match the domains it just
named, with an explicit self-check instruction ("recount the domains you
just named ... never carry over a number from how many claim lines
appeared"). This is what's now in `newsresearch/llm/prompts/summarization.txt`
rule 5.

## Validation of the final prompt

Each of the 6 real repro clusters was run **3 independent times** (18 real
LLM calls total) against the final prompt, to check LLM-sampling stability
rather than trusting one lucky draw:

| cluster | real distinct asserting domains | iteration-3 result (3/3 runs) |
|---|---|---|
| 29 (was wrong: "six sources") | 3 (bbc, guardian, npr) | correct in all 3 runs — named domains matched, per-fact granularity preserved (e.g. 800g figure attributed to Guardian only, not all 3) |
| 14 (was wrong: "five sources") | 3 (bbc, guardian, npr) | correct in all 3 runs — no numeral/name mismatch recurred |
| 17 (was wrong: "four sources") | 2 (bbc, guardian) | correct in all 3 runs |
| 23 (already correct) | 3 (bbc, guardian, npr) | still correct in all 3 runs |
| 27 (already correct) | 3 (bbc, guardian, npr) | still correct in all 3 runs — and per-fact breakdown got *more* precise than the original (correctly separates "closed, per NPR" from "reopened, per BBC/Guardian/NPR" from "southern still closed, per BBC") |
| 9 (already correct) | 3 (bbc, guardian, npr) | still correct in all 3 runs |

Verified against the real per-cluster domain/claim rows pulled directly from
Postgres (not against the markdown's prose descriptions), e.g. cluster 27's
real rows show the initial "both runways closed" claim is NPR-only, the
"reopened ~2hrs later" claim is asserted by all 3, and "southern still
closed" is BBC-only — the final prompt's output for this cluster attributes
each of those three sub-facts to the correct, distinct domain subset.

**18/18 real reproduction-cluster runs correct** (no over/under-count, no
numeral/named-list mismatch), across both previously-failing and
previously-passing clusters.

## Regression check against Task 3.6.1a's original 8-cluster sample

Re-ran the final prompt against `notebooks/phase3_summarization_review.py`'s
original sample clusters (21, 31, 2, 9, 34, 0, 15, 1 — the DeepSeek/Qwen/Kimi
AI-cost fixture corpus). All 8 outputs still: use only stated information
(rule 1), correctly merge true paraphrases vs. keep genuinely distinct facts
separate (rules 2-4, e.g. cluster 2's two distinct price cuts for two
distinct models kept separate, not averaged), and now additionally name
domains explicitly per fact where 3.6.1a's original output only used vaguer
"according to X sources" phrasing — a readability improvement, not a
regression. No dropped facts, no fabricated facts, no reintroduced
miscounting. Full outputs: `notebooks/phase3_task3.7.1b_361a_regression.json`.

## Change shipped

`newsresearch/llm/prompts/summarization.txt` rule 5, replaced in full (see
file for exact text). Summary of the change: explicit per-fact (not
whole-summary) domain identification, a strong preference for naming domains
over stating bare numbers, and — when a number is used — an explicit
self-check instruction against the just-named domain list rather than the
original claim-line count. No other rule and no other part of the prompt
(including the `{claims}` input formatting) changed; `agents/
summarization_agent.py::_format_claims`'s `"- {domain}: {claim_text}"` format
was left as-is since the fix worked entirely from the existing format.

## Handoff

Prompt file already updated in place at
`newsresearch/llm/prompts/summarization.txt` on `feat/datascientist` — no
other change needed, `agents/summarization_agent.py` already loads this file
by path, so `backend-engineer` only needs to pull this branch's prompt file
into whatever branch merges it (or cherry-pick the file), no code change
required for this fix specifically.
