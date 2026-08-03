# Claim extraction prompt + schema review (Task 3.2.1a)

Owner: `data-scientist`. Design/analysis deliverable, not production code.
Handoff target: `backend-engineer` (Task 3.2.1b: wire via
`get_chat_model("claim_extraction").with_structured_output(Claim)`).

Prompt: `newsresearch/llm/prompts/claim_extraction.txt` (committed to this
branch, ready to lift as-is).
Draft schema: `notebooks/claim_schema_draft.py` (`Claim`/`ClaimList`, ready
to lift into `newsresearch/llm/schemas.py`).
Scripts: `notebooks/phase3_claim_extraction_review.py` (initial 10-article
pass), `notebooks/phase3_claim_extraction_recheck.py` (2-article iteration-2
recheck).
Raw output: `notebooks/phase3_claim_extraction_samples.json` (10 articles,
iteration 1 prompt), `notebooks/phase3_claim_extraction_iteration2_recheck.json`
(2 articles, iteration 2 prompt).

**Real articles, one run each on 2026-08-03, not a golden-dataset eval —
PRD Sec.7 names that gap explicitly for v1.** Judgments below are a manual
read of real model output against real article text, not a measured/scored
metric.

## Data

10 real articles pulled live via `sourcing/rss.py::fetch_trusted_rss`
against the 4 trusted-tier feeds already wired (`bbc.com`, `theguardian.com`,
`npr.org`, `aljazeera.com`) with broad keyword queries (`election`,
`health`, `AI`, `climate`, `economy`) to get varied beats, then
`sourcing/fulltext.py::fetch_fulltext` for real body text. Full text was
never written to disk or committed — only extracted `Claim` objects plus
`url`/`title`/`domain` went into the sample JSON, per the no-full-text
rule. Topics covered: Australian state election candidates, FIFA
presidency politics, Aung San Suu Kyi/Myanmar, global HIV-prevention
funding, Iran/US negotiations, and a long-form human-interest piece on
protests in Pakistan-administered Kashmir.

## Design Question 1: `Claim` schema shape

TRD Sec.4.4 / EXECUTION_PLAN's Story 3.2 acceptance line specifies
`{claim_text, subject, attributed_source}`. Against real article text, this
minimal shape was not enough — evaluated the actual model output revealed
two recurring, concrete gaps:

1. **Attribution needs a type, not just a name.** `attributed_source` alone
   answers "who said it" but not "how solid is this." Real articles mix
   the outlet's own verified reporting ("Brent Crude oil was down 4.5%"),
   named-source attribution ("the WHO recommended..."), one-party
   allegations the article doesn't itself verify ("the local government
   says the JAAC group is armed" — disputed by JAAC in the same piece), and
   outright disputes ("authorities say X; JAAC denies it"). Downstream,
   the Bias & Framing agent (TRD Sec.4.6) needs exactly this distinction to
   do claim-level comparison across sources — collapsing all four into one
   `attributed_source` string loses the signal that made the claim
   comparison-worthy in the first place. Added `attribution_type`:
   `reported` / `attributed` / `alleged` / `opinion`.
2. **Certainty/framing-in-time matters.** Several claims are explicitly
   provisional in the article's own language ("early reports," "officials
   say they expect," a live negotiation not yet confirmed) or explicitly
   contested by another party in the same piece. Treating these the same
   as a settled fact would misrepresent the source's own hedging when
   claims get compared across articles later. Added `certainty`:
   `confirmed` / `developing` / `disputed`.

Deliberately **not** added: a numeric confidence score. That would
reintroduce a scalar-labeling pattern the PRD's risk table explicitly
warns against for framing/bias (descriptive, not scalar) — `certainty`'s
three categorical values do the same descriptive job without inventing a
precision the extraction step can't actually justify.

Deliberately **not** split into "claim as stated vs. paraphrased" as two
separate fields — real output showed `claim_text` naturally stays close to
the article's wording when the prompt instructs it to (rule 2), so a
second "verbatim" field would just duplicate `claim_text` in the common
case. Handled instead by prompt wording ("stay close to the article's own
wording... not your own interpretation").

Multi-source attribution within one sentence (e.g. "Saudi Arabia and UAE
urged Trump against strikes") is handled by letting `subject` name the
compound entity rather than forcing a schema-level list — kept simple since
splitting further would just fragment one real claim into artificial
sub-claims with the same `attributed_source`.

**Final proposed fields** (`notebooks/claim_schema_draft.py`):
`claim_text`, `subject`, `attributed_source`, `attribution_type`,
`certainty` — two more than the original three-field acceptance line. This
is a schema *addition*, not a change to the agent's stated input/output
*contract* boundary (still one `Claim`/article, still consumed the same way
downstream) — flagging to `tech-lead` only if the extra fields are judged
to cross into re-scoping Story 3.3/3.6's consumption of claim clusters,
which on inspection they don't (Story 3.3 clusters on `claim_text` alone;
the new fields ride along as metadata, same shape `attribution_type`/
`sentiment` already do for Story 3.4).

## Design Question 2: prompt iteration

**Iteration 1** (`claim_extraction.txt`, 9 rules) ran clean on structured
extraction, atomicity, and attribution — no run-on-paragraph claims, no
invented attributions, `attribution_type`/`certainty` correctly
distinguished `reported` vs. `attributed` vs. `disputed` (e.g. the Kashmir
piece: "authorities say the felled trees were placed by JAAC" tagged
`disputed` because JAAC denies it in the same article; "Grinsztejn
expressed doubt targets will be met" tagged `opinion` correctly, not
`reported`).

**Failure mode found:** long human-interest/narrative articles produced
excessive claim counts driven by low-value color/sentiment quotes that add
no new fact beyond one already extracted — e.g. the Kashmir piece (65
claims) extracted both "Salman lost his brother in the protests" (a fact)
**and** "Salman feels proud his brother gave his life for the cause" (pure
sentiment restating the same fact) as two separate claims. This inflates
claim-clustering input with singleton, uncomparable claims (Task 3.3.1
clusters across articles — a one-off grief quote will never match another
source's coverage) and costs more per-article than the substantive content
requires (NFR-1 cost ceiling).

**Fix:** added rule 9 — do not extract a claim for color/sentiment that
restates a fact already captured elsewhere; extract descriptive/
scene-setting sentences only if they themselves assert a new, checkable
fact.

**Recheck, same 2 highest-count articles, iteration-2 prompt:**

| Article | Iteration 1 | Iteration 2 |
|---|---|---|
| Kashmir protests (BBC, long-form human-interest) | 65 claims | 31 claims |
| One Nation candidates (Guardian, dense candidate-roster piece) | 44 claims | 46 claims |

Kashmir dropped 65→31 (53%) — spot-checked the dropped claims are exactly
the sentiment/color duplicates rule 9 targets (`Salman feels proud...`,
`Asma said no mother could ask for a better son`, etc.), and confirmed the
underlying facts they restated survive (`Salman lost his brother in the
protests` is still extracted). One Nation's count didn't drop (44→46, noise
band) — that piece is a dense list of distinct candidate names/facts with
almost no color quotes to begin with, so no regression there, and rule 9
correctly left it alone rather than over-suppressing legitimate content.

This is a 2-article recheck, not a systematic before/after over all 10 —
directionally strong (a clear, targeted 53% reduction on exactly the
article type the failure mode was found in, zero loss on a dense
control), but not a measured average-reduction rate.

## Design Question 3: end-to-end test

Ran for real via `get_chat_model("claim_extraction")` (`gpt-4.1-mini` per
`Settings.models.claim_extraction`) `.with_structured_output(ClaimList)`
against `ChatPromptTemplate.from_template(claim_extraction.txt)` — not
reviewed as prompt text in isolation. 10/10 real articles produced valid,
schema-conformant claims with no parse failures. Did not confirm Langfuse
trace visibility (Task 3.2.1b/3.7.2's job once wired into the graph with
`config` propagation — this script calls the model directly, outside any
`graph.invoke()` `callbacks` context).

## Example outputs (iteration 2, illustrating the target quality bar)

From the Iran/US negotiations article (BBC):
```
{"claim_text": "Iran's foreign ministry denied they were negotiating with the US or that there were any plans to do so.",
 "subject": "Iran's foreign ministry", "attributed_source": "Iran's foreign ministry",
 "attribution_type": "attributed", "certainty": "confirmed"}
{"claim_text": "Trump said he had been asked by Iran and US allies in the Middle East to \"hold off\" as the \"perimeters\" of a deal had been agreed.",
 "subject": "Iran and US allies' request to Trump", "attributed_source": "Donald Trump's post on Truth Social",
 "attribution_type": "attributed", "certainty": "developing"}
```
Two directly conflicting claims from the same article, correctly kept as
separate atomic claims with distinct attribution — exactly the shape Task
3.3.1's cross-article clustering needs to surface a disputed cluster.

From the Kashmir piece (iteration 2, post-fix):
```
{"claim_text": "The local government says the JAAC group is armed and that the protests are about disrupting the ongoing polls.",
 "subject": "local government about JAAC", "attributed_source": "the article's own reporting",
 "attribution_type": "attributed", "certainty": "disputed"}
{"claim_text": "The JAAC rejects being characterized as terrorists, says they are peaceful, and claims they are funded from the Pakistan-administered Kashmir community abroad but not sponsored by India.",
 "subject": "JAAC", "attributed_source": "the article's own reporting",
 "attribution_type": "attributed", "certainty": "confirmed"}
```

## Caveats, stated plainly

- 10 real articles, one run each, one day, 4 trusted RSS outlets (no GDELT
  — confirmed rate-limited in this environment per known issue #101, RSS
  alone is sufficient for this evaluation per the task brief). Manual
  spot-check of output quality, not a scored/repeated eval — no golden
  dataset exists for v1 (PRD Sec.7).
- The iteration-2 fix (rule 9) was validated against exactly 2 articles (the
  two worst offenders from iteration 1), not the full 10 or a larger
  sample. Directionally strong given the size and specificity of the
  Kashmir reduction, but revisit if a wider sample later shows a different
  false-negative rate (rule 9 suppressing a claim that should have counted).
- All 10 articles came from 3 outlets (BBC, Guardian, Al Jazeera) — no
  wire-service or non-English-source article tested, and no article that
  was itself opinion/editorial (rule 10's "return empty claims list" branch
  was never actually exercised in this sample).
- Model used: `gpt-4.1-mini` (`Settings.models.claim_extraction`'s existing
  configured small model, per TRD's cost-containment rationale for
  mechanical extraction stages — not re-evaluated here against a larger
  model; quality on this sample looked good enough that switching up
  isn't recommended on this evidence).
