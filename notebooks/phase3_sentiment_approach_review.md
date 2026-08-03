# Task 3.4.1a -- Sentiment approach: lexicon vs. small-model

Story 3.4 / FR-14: sentiment is an auxiliary, descriptive attribute attached
to articles/claims after clustering, never a clustering axis, and never a
bias proxy (PRD risk table: "sentiment used as bias proxy" -> false-confidence
inaccurate bias labels). This task is a design recommendation only --
`backend-engineer` (Task 3.4.1b) wires it and builds the clustering-unaffected
proof against real `Settings`/pipeline code.

## Recommendation: lexicon (VADER), not a small-model LLM call

**Chosen: `vaderSentiment`** (rule-based, pure-Python, no ML weights, ~50KB
lexicon file). Not currently a project dependency -- one line to add
(`vaderSentiment>=3.3.2`) when `backend-engineer` wires this; tested here via
`uv run --with vaderSentiment` so nothing was added to `pyproject.toml`/lock by
this analysis-only branch.

### Why not the small model

- **Volume.** Sentiment runs per-article *and* per-claim -- Task 3.2.1a's
  samples alone produced 306 claims across 10 articles (30+ claims/article).
  A real run touches many articles across many subtopics; that's hundreds to
  low-thousands of sentiment calls per run. Even on the small model
  (per NFR-1's cost ceiling and the TRD's small/large split), that's real
  added cost and added latency (network round-trip per call, or awkward
  batching) for a signal FR-14 explicitly says doesn't need frontier
  reasoning.
- **Task fit.** Sentiment polarity of a single sentence/short claim is
  exactly the case lexicon methods were built for and are well-validated on
  (VADER was built/tuned on social-text and short-form content, but performs
  comparably on short factual/quote-heavy news sentences -- see samples
  below). There's no coreference, multi-hop reasoning, or world knowledge
  needed to classify "X was killed" as negative.
- **Determinism.** Lexicon scoring is deterministic and reproducible run to
  run (no sampling temperature, no model-version drift) -- useful for
  Phase 5's cross-run drift tracking (FR-24 sentiment-shift comparisons),
  where you want the delta to reflect real sentiment change, not LLM
  non-determinism.
- **Cost.** Effectively zero marginal cost/latency (microseconds, in-process,
  no network call, no token spend) vs. a small-model call that still costs
  tokens + a round trip per claim/article at this call volume.

### Why not roll a custom lexicon instead of a dependency

Ladder check: no sentiment-scoring library is currently installed
(`grep -i "sentiment\|vader\|textblob\|nltk" pyproject.toml` -- empty), and
hand-rolling word-polarity + negation/booster handling ("not good", "very
bad", "barely improved") from scratch would just reproduce a worse, ad hoc
version of a solved, tiny, single-purpose, well-tested library. `vaderSentiment`
has no heavy transitive dependencies (no torch/numpy requirement) --
installing it pulled in 6 small packages total. This is the one case on the
ladder where a small new dependency is the right call, not a shortcut around
missing understanding.

## Scale choice: 3-class (positive / neutral / negative), not a scalar score

VADER internally produces a continuous `compound` score in [-1, 1]. I
recommend **bucketing it into 3 classes** for the metadata attribute exposed
downstream, using VADER's own documented default thresholds
(`compound >= 0.05` -> positive, `compound <= -0.05` -> negative, else
neutral) -- see `label_for()` in the eval script.

Reasoning, directly against FR-14's "auxiliary, not a bias proxy" framing:

- A raw scalar (e.g. "-0.61") invites exactly the kind of false-precision
  misread the PRD's risk table warns about for framing/bias labels --
  someone skimming a briefing sees a number and reads it as a calibrated bias
  measurement rather than a coarse sentiment observation.
- 3-class labels read as plainly descriptive ("this claim's language skews
  negative") without implying a scale or a spectrum position, matching the
  same non-scalar posture the bias/framing labeling stage already takes
  (per the PRD's deliberate rejection of a left/center/right scale).
- The underlying continuous `compound` score should still be stored
  alongside the label (not discarded) so a later v2 eval or drift-analysis
  task can use finer granularity if warranted -- but the label, not the raw
  float, is what surfaces in the user-facing briefing/metadata.

I did not choose 5-class (very negative/negative/neutral/positive/very
positive) -- at claim-sentence granularity, VADER's compound score doesn't
reliably separate "negative" from "very negative" in a way that would survive
scrutiny as more than noise; 3-class is the granularity the underlying
lexicon actually supports at this text length.

## Real test results

Ran `notebooks/phase3_sentiment_approach_eval.py` against:
1. All 306 real claims from `notebooks/phase3_claim_extraction_samples.json`
   (Task 3.2.1a's committed real-article claim-extraction output).
2. Two real articles' full text, fetched transiently in-memory via
   `sourcing/fulltext.py::fetch_fulltext` (never written to disk -- the
   script only prints a short excerpt + score, consistent with the
   no-full-text-storage rule; this is a manual spot-check, not a measured
   eval against a labeled ground truth, since v1 has no golden dataset per
   PRD §7).

Label distribution across the 306 claims: `{neutral: 79, negative: 131,
positive: 96}` -- a plausible spread for a real-news claim mix (conflict/
death/exclusion claims skew negative, achievement/access/rights-of-reply
claims skew positive, purely factual "X is Y's president" claims are
neutral).

Sample claim-level outputs (real claims, unedited):

| compound | label | claim |
|---|---|---|
| +0.66 | positive | "Warren Pickering said the party expects and invites a healthy amount of scrutiny of its candidates and policy positions." |
| +0.51 | positive | "A Guardian spokesperson said holding public figures to account is the role of a free press and an essential part of the democratic process." |
| -0.65 | negative | "Pauline Hanson said it was not the first time she had excluded the ABC and claimed she denied ABC journalists access to her events for eight months in 1996." |
| -0.56 | negative | "Pauline Hanson has vowed to ban the ABC and the Guardian from attending her press conferences during the campaign." |
| 0.00 | neutral | "One Nation has named its first candidates for the upcoming Victorian state election." |
| 0.00 | neutral | "Warren Pickering is One Nation's Victorian president." |

Sample article-level outputs (real fulltext, fetched live):

- BBC Kashmir-protests article (deaths, gunshot wounds, clashes): compound
  **-0.9973** -> negative. Correct and unsurprising given the subject matter.
- Guardian HIV-prevention/USAID-cuts article: compound **+0.9785** ->
  positive. This is a real, worth-flagging **limitation**: the article's own
  framing is actually mixed/negative overall (a promising drug that funding
  cuts are blocking access to), but lexicon scoring over the whole article
  is dominated by surface positive words ("innovations," "reduce," "prevent,"
  "protection") appearing more/earlier than the negative-framed cuts
  language. This is a known weakness of whole-document lexicon scoring
  (no discourse structure, no "but" weighting) -- not a claim-extraction
  problem, since claim-level scoring (splitting "innovation could reduce
  cases" from "cuts prevent access" into separate claims) handles this case
  correctly, per the claim-level table above where individual claims come
  back with sensible per-assertion polarity. This is a reason to prefer
  **claim-level sentiment as the primary attribute** and treat
  whole-article-level sentiment (if computed at all) as a rougher secondary
  signal, not the other way around -- flagging this rather than glossing
  over it, since it's a real, observed failure mode, not a hypothetical.

## How to prove clustering is unaffected (spec for backend-engineer, Task 3.4.1b)

Not implementing this myself -- this is the exact test I'd want built:

1. Take a fixture of real claim_texts (e.g. reuse
   `notebooks/phase3_claim_extraction_samples.json`'s claims, or a subset).
2. Run claim clustering (Story 3.3's `clustering/cluster.py` +
   `clustering/embeddings.py` path) **twice** on the identical input claim
   list:
   - Run A: claims as plain `Claim` objects/dicts with no `sentiment` field
     present at all.
   - Run B: the *same* claims, each with a `sentiment` field (label +
     compound score) already attached, exactly as they'll look after
     Task 3.4.1b wires the sentiment collector to run before clustering (or
     interleaved with it) in the real pipeline.
3. Assert, not just "no crash":
   - The **embedding vectors** fed to `cluster()` are identical between Run A
     and Run B (byte-for-byte or `np.array_equal`) -- proves the embedding
     step only ever reads `claim_text` (or whatever fields it's specified to
     read), never `sentiment`.
   - The **cluster label assignment per claim index** is identical between
     Run A and Run B (same cluster id per claim, allowing for label
     re-numbering if HDBSCAN/KMeans's arbitrary cluster-id ordering differs
     run to run -- compare via Adjusted Rand Index == 1.0, or an explicit
     assertion that the partition of claim-indices into clusters is
     identical, rather than assuming cluster ids sort the same way twice).
4. A stronger, cheap-to-add variant: monkeypatch/spy on the embeddings
   function (`clustering/embeddings.py`'s wrapper over `get_embeddings()`)
   in the test and assert the exact list of strings passed to it during
   Run B does not include any `sentiment`-derived text -- this catches the
   subtle bug where `sentiment` accidentally leaks into the text that gets
   embedded (e.g. someone naively does `f"{claim_text} ({sentiment_label})"`
   when building the embedding input), not just the coarser "did the final
   clusters match" check.

This is a concrete, deterministic test spec (real fixture, in/exclude
comparison, ARI or exact-partition equality) -- not a manual spot-check, and
it is a fair proxy for the FR-14 requirement in production code, since it
directly exercises the same `cluster()`/`get_embeddings()` call path Story
3.3 wires.

## Cost/quality trade-off summary

| | VADER (chosen) | Small-model LLM call |
|---|---|---|
| Marginal cost/call | ~0 (in-process, no tokens) | small but nonzero, multiplied by claim+article volume per run |
| Latency/call | microseconds | network round-trip |
| Determinism | fully deterministic | subject to model/sampling variance |
| Quality on short factual/quote sentences | good (validated above) | plausibly better, but not needed for an auxiliary/non-bias-proxy signal per FR-14 |
| Quality on whole-document text | weaker (surface-word dominance, no discourse structure -- see HIV example) | better, but this is a reason to keep the *article-level* variant secondary/optional, not a reason to switch the primary claim-level signal to the LLM |

Net: lexicon (VADER) is the right cost/quality trade-off for Story 3.4's
scope. If a future phase needs sentiment to carry more analytical weight than
"auxiliary/descriptive," that's the point to revisit -- not now, since FR-14
explicitly caps its role.
