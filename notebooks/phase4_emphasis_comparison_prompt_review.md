# Task 4.2.1a — claim-level emphasis-comparison prompt

Owner: `data-scientist`. Design deliverable only — no production wiring.
`backend-engineer` owns Task 4.2.1b.

Deliverables:
- **No new prompt file.** See the verdict below — the recommendation is a
  one-hunk amendment to the already-shipped
  `newsresearch/llm/prompts/bias_framing.txt` (rule 9), not a second prompt.
- **No schema change.** `ClusterFraming` / `ArticleFraming` as shipped in
  `newsresearch/llm/schemas.py` are sufficient.
- `notebooks/phase4_emphasis_comparison_review.py` — the harness (5 modes:
  `--coverage`, `--assemble`, `--candidate`, `--compare`, `--rule9`/`--baseline`).
- `notebooks/phase4_emphasis_comparison_samples.json` — the rejected dedicated
  prompt's real output on 8 clusters, kept so the head-to-head is auditable.
- `notebooks/phase4_emphasis_rule9_baseline.json` /
  `notebooks/phase4_emphasis_rule9_samples.json` — control and treatment arms
  of the rule-9 amendment.

---

## Verdict on the scoping question, first

**Task 4.2.1a as written is substantially redundant with Task 4.1.1a's shipped
output. I recommend narrowing it, not building a second prompt.**

TRD §4.6's third output bullet decomposes into three parts, and two and a half
of them already exist:

| TRD §4.6 clause | Where it already lives | New design needed? |
|---|---|---|
| "which sources include it" | `claim_cluster_articles.relation = 'asserts'` | No — deterministic DB read |
| "which sources omit it" | `claim_cluster_articles.relation = 'omits'` | No — deterministic DB read, and **must not** be model-generated (issue #114) |
| "where language/emphasis differs on shared claims" | `ClusterFraming.divergence` (the enum) + `divergence_note` (the prose) | Only a **quality** improvement, not a new component |

`divergence_note` is not a bureaucratic justification of the enum — on real
data it already *is* the emphasis-comparison prose the task asks for. Measured
over the 69 multi-article cluster-outputs of Task 4.1.1a's committed 3-run
sample (`--coverage`, no LLM calls):

| property of `divergence_note` on multi-article clusters | result |
|---|---|
| names ≥ 2 distinct `article_id`s | **67/69 (97%)** |
| contains a quoted span | 53/69 (77%) |
| quotes wording that really occurs in that cluster's claim text | 35/69 (51%)\* |

\* naive single-quote matcher; the same measure with trailing punctuation
stripped gives 78%. The uncorrected number is left in because it is what the
committed script prints.

Real examples, verbatim from the shipped agent's output — this is what the
"new" prompt would have been asked to produce:

> `[wording_or_emphasis_differs]` leipzig-drone-0 (bbc.com) uses the phrase
> 'parcel burst into flames,' presenting a straightforward event, while
> leipzig-drone-2 (theguardian.com) uses stronger language and labels the item
> as a 'firebomb' that 'exploded,' adding the detail 'posted via DHL.'

> `[context_differs]` leipzig-drone-0 (bbc.com) mentions both current high
> alert at airports and the link between the rise in incidents and Russia's
> invasion of Ukraine, while leipzig-drone-1 (theguardian.com) covers the
> overflights at multiple types of facilities in Germany and Europe but omits
> the explicit connection to the timing of Russia's invasion.

That second one even covers within-cluster *detail* omission — the one flavour
of omission that genuinely requires reading the text, as opposed to
cluster-level omission, which the DB already knows.

### So I built the dedicated prompt anyway, to check I wasn't just being lazy

`phase4_emphasis_comparison_review.py::CANDIDATE_PROMPT` is a real, carefully
written dedicated emphasis-comparison prompt with a one-field
`EmphasisComparison` schema, run for real against the same 8 multi-article
clusters (`--candidate`). It was deliberately given **more** than the shipped
agent gets: the omission roster is passed into it as established fact, so it
could in principle say something the shipped output structurally cannot.

Head-to-head (`--compare`):

| | dedicated candidate | shipped `divergence_note` |
|---|---|---|
| mean length | 719 chars | 350 chars |
| quoted spans found | 27 | 15 |
| of those, verbatim in the claim text | 24 (89%) | 13 (87%) |
| outputs naming an omitting article | **0/8** | n/a (forbidden by design) |

Two findings:

1. **The omission roster was inert.** 0 of 8 outputs mentioned an omitting
   article even though every prompt listed them. That is *good* — no
   fabrication — but it confirms the omission half of TRD §4.6 belongs in a
   deterministic renderer, not in a prompt, exactly as issue #114 and Task
   4.1.1a's testing already established. A second LLM pass buys nothing here.
2. **The candidate's only measurable advantage was quote density** — roughly
   twice as many source spans quoted, at the same verbatim accuracy and twice
   the length. Its actual content on every one of the 8 clusters is the same
   observation `divergence_note` already made, at greater length. Compare
   cluster 15:
   - shipped: "…leipzig-drone-1 mentions the safe landing elsewhere, while
     leipzig-drone-2 emphasises the minor nature of the damage."
   - candidate: "…one on the plane's safe landing, the other on the limited
     damage."

**A second pass would cost a full extra large-model traversal of every claim in
the subtopic** (per-cluster: 31 calls vs. the bias agent's 4 batched calls;
even batched, roughly a doubling of the bias/framing stage's token spend under
NFR-1) **to restate what the first pass already said.** That is not a defensible
cost/quality trade.

---

## What I recommend instead

### 1. One prompt-rule amendment (the only prompt work in this task)

Quote density was the candidate's single real advantage, and it is a property
of rule 9, not of having a separate agent. Amended rule 9 in
`newsresearch/llm/prompts/bias_framing.txt` on this branch — the added text:

> When articles differ, quote the differing words themselves from each article
> involved rather than describing the difference in your own words -- one
> article's id, then the words it used, then the other article's id and the
> words it used, with both sets of words copied from those articles' own claim
> lines. Each quoted span must be a single unbroken run of words copied
> character for character, under the same rules as rule 7: no ellipsis, no
> stitching, no normalising. Quote a shorter phrase rather than an inexact
> longer one. When the articles do not actually differ, do not manufacture a
> contrast to have something to quote.

The last sentence is not decoration: the whole risk of pushing a model to quote
a difference is that it invents one to fill the field. That is the same
fabricated-divergence failure Task 4.1.1a spent three iterations killing.

**Measured, on the real subtopic, at production batch size 8:**

| arm | runs / calls | quoted spans per multi-article note | verbatim | 4.1.1a's mechanical checks |
|---|---|---|---|---|
| shipped prompt (4.1.1a's committed samples) | 3 / 12 | 1.35 | 78% | 0 failures |
| shipped prompt, same-day control | 1 / 4 | 1.04 | 62% | 0 failures |
| **amended rule 9** | **2 / 8** | **3.17** | **86%** | **0 failures** |

"Mechanical checks" reuses `phase4_bias_framing_review.py::check` unchanged —
`per_article` membership, the single-article/multi-article divergence rules in
both directions, the domain check, the political-lean substring scan and the
source-counting scan. **The amendment regressed none of them across 46
multi-article cluster-outputs.** That is the property that mattered: 4.1.1a's
0/93 result is the expensive one and this does not spend it.

Before/after on the same cluster, real output:

| | |
|---|---|
| before | leipzig-drone-0 (bbc.com) uses 'professional, hybrid threat scenario', leipzig-drone-1 (theguardian.com) also emphasizes 'professional hybrid threat', while leipzig-drone-3 (npr.org) shortens this to 'hybrid attack scenario' without 'professional'. The emphasis on professionalism varies. |
| after | leipzig-drone-0 calls it "a professional, hybrid threat scenario"; leipzig-drone-1 says "a professional hybrid threat scenario"; leipzig-drone-3 refers to "a hybrid attack scenario". |

| | |
|---|---|
| before | leipzig-drone-0 (bbc.com) notes the timing of the incidents as beginning with Russia's invasion of Ukraine, adding a temporal context, while leipzig-drone-1 (theguardian.com) lists the incidents broadly among European countries without this timing connection. |
| after | leipzig-drone-0 provides German-focused timeline and heightened alert: "German airports are on high alert" and "since Russia's full-scale invasion of Ukraine"; leipzig-drone-1 expands to a European context: "Germany and other European countries have experienced a series of suspicious unauthorized drone overflights". |

The "after" text is checkable against the source by a reader; the "before" text
asks the reader to trust the model's characterisation. That is the whole
quality delta of this task.

**Known defect, stated plainly.** 21 of the 146 quoted spans did not match the
claim text exactly. I inspected all 21: **8 are ellipsis elisions** (all 8
verified segment-by-segment as contiguous spans of the real claim text — the
same faithful-elision habit Task 4.1.1a measured at 1.6% on `evidence_quote`,
which `tech-lead` has already ruled acceptable in TRD §4.6), **12 are artefacts
of my quote-extraction regex** swallowing the prose between two adjacent
quotes, and one (`'near Antonov planes'`, cluster 29) is a possible short
paraphrase. **Zero fabricated wording observed.** The amendment increases the
absolute count of ellipsis elisions, since it increases quoting overall.

One behavioural side-effect: the amended notes name `article_id`s but drop the
`(domain)` annotation more often than before. Harmless — 4.2.1b's renderer has
the id→domain map from the DB and can annotate deterministically.

### 2. The deterministic renderer for 4.2.1b (no LLM call)

`--assemble` shows the whole of TRD §4.6's third bullet assembled with zero
model involvement beyond the framing output that already exists.
`phase4_emphasis_comparison_review.py::render` is the reference shape (it has a
runnable `check()` self-check asserting an omitting article never appears on
the covering line):

```
Cluster task371-leipzig-drone:11
  Covered by : leipzig-drone-0 (bbc.com), leipzig-drone-2 (theguardian.com)
  Not covered: leipzig-drone-1 (theguardian.com), leipzig-drone-3 (npr.org)
  Difference : [wording_or_emphasis_differs] leipzig-drone-0 states 'a parcel
               burst into flames on the ground', while leipzig-drone-2 reports
               'a firebomb exploded at Leipzig airport ... posted via DHL.'
```

Constraints on the renderer, all load-bearing:

- **Key on `article_id`, carry `domain` alongside.** 29 of the 31 real clusters
  have at least one omitting article, and the Guardian appears on *both* sides
  in several of them (cluster 11: `leipzig-drone-2` covers, `leipzig-drone-1`
  omits — same domain). Collapsing to per-domain would render
  "theguardian.com covered and did not cover this claim."
- **`Not covered` comes only from `relation='omits'` rows.** Never from the
  model, never inferred from a covering-set complement computed over anything
  other than those rows.
- Single-article clusters (`divergence = single_article_only`, 24 of 93
  outputs in the 4.1.1a sample) have no difference text worth rendering. The
  interesting fact about them is the omission line.

I am not specifying the output artifact's structure or where it is persisted —
that is `backend-engineer`/`tech-lead`'s call, same as the
`claim_clusters.framing_label` storage-shape question 4.1.1a flagged and left
open.

---

## Honest caveats

- **The corpus under-exercises this task more than it did 4.1.1a, and that is
  the most important limitation here.** `task371-leipzig-drone` is a
  security-incident story where four articles broadly agree. An
  *emphasis-difference* prompt is exactly the thing that needs genuine
  cross-source disagreement to be evaluated, and there is essentially none:
  across 93 cluster-outputs, `accounts_conflict` occurs 3 times, all on one
  cluster. Everything above therefore tests the prompt on *low-stakes wording
  differences* ("major hub" vs "DHL hub"), not on the contested-story case the
  briefing agent most needs this signal for. **My redundancy verdict is
  strongest exactly where the evidence is strongest (that a second pass
  restates the first), and weakest on the possibility that a dedicated pass
  would pull ahead on a genuinely contested story.** I have not tested that and
  cannot claim it either way. If a contested-topic run ever lands in the DB,
  re-run `--candidate` and `--compare` against it before treating this verdict
  as settled.
- **This is a manual spot-check, not a scored eval** (PRD §7's named v1 gap).
  There is no golden dataset. Quote density and verbatim rate are mechanical
  and real; "the candidate said the same thing at greater length" is my reading
  of 8 outputs against the claim text, nothing more.
- **The amended arm is 2 runs (8 calls) vs. the baseline's 3 runs.** Density
  more than doubled and mechanical failures stayed at 0, so I did not spend
  more; a third run before merge would be cheap insurance if
  `backend-engineer` wants it.
- **Cluster quality ceiling unchanged.** Task 3.3.1a's 0.674 false-merge
  precision still applies; the amendment does not touch rules 1–3, and
  `cluster_coherence` remains the flag downstream should read. Pushing for more
  quoting slightly raises the risk of quoting two claims that were never about
  the same fact, which is what the "do not manufacture a contrast" sentence
  guards; 0 mechanical failures is consistent with that holding, but the check
  cannot detect it directly.
- **No article full text was touched, fetched, or written** — only
  `claim_text`, `article_id`, `domain` leave the DB (FR-25). The saved JSON
  samples contain claim text only, which is already persisted.
- **No political-lean scalar is introduced.** No schema change at all, so
  4.1.1a's four structural properties are untouched, and the amendment adds
  nothing to a vocabulary — it constrains *how* `divergence_note` cites source
  wording.

## Handoff to `backend-engineer` (Task 4.2.1b)

1. **Prompt change**: pick up the single hunk in
   `newsresearch/llm/prompts/bias_framing.txt` on `feat/datascientist` (rule 9
   only; the rest of the file is unchanged from what PR #121 merged). No new
   prompt file, no `Settings` entry, no new stage.
2. **No schema change** and **no new LLM call** in the 4.2.1b path — which
   also satisfies that task's own acceptance criterion ("no new
   embedding/clustering call is made in this path") for free.
3. **Assemble** per `render()` above: `asserts`/`omits` from
   `read_cluster_article_relations`, difference text from the bias agent's
   `ClusterFraming.divergence` + `divergence_note`, keyed per `article_id`
   with `domain` carried alongside.
4. If the assembled artifact needs somewhere to live, that is a storage-shape
   question for `tech-lead`, not something I have pre-empted.
5. Task 4.2.1a's original acceptance line ("draft/iterate the
   emphasis-comparison prompt… documented sample outputs judged plausible") is
   answered by this document as a **descope recommendation with the evidence
   behind it** — the prompt was drafted, run for real, and rejected in favour of
   a cheaper change. Whether that satisfies the criterion is
   `acceptance-verifier`'s call, not mine.
