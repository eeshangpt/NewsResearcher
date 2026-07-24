# Phase 2 Task 2.2.1a — subtopic-propose prompt design

Owner: `data-scientist`. Design deliverable per `EXECUTION_PLAN.md`'s
"Phase 2 — Role Ownership & Parallelization Re-analysis." Not production
code. Handoff target: `backend-engineer`, Task 2.2.1b (wire via
`get_chat_model("subtopic").with_structured_output(Schema)`).

Prompt file: `newsresearch/llm/prompts/subtopic_propose.txt`.

## UNVERIFIABLE HERE: no real model call was made

This environment has no `OPENAI_API_KEY` configured (no `.env` file exists
in this worktree at all — `.env` is gitignored and only `.env.example`,
with an empty key, is tracked; no other API key was found in the shell
environment). `get_chat_model("subtopic")` cannot be exercised for real
here. The sample-topic outputs below are explicitly **illustrative /
expected**, written by me as a plausibility check on the prompt's own
instructions and format — they are not the output of any real LLM call,
and must not be read as a validated result. This is stated here plainly per
the instruction not to overstate v1's evaluation rigor; when a
`backend-engineer` wires Task 2.2.1b with real API access, the actual model
output should be checked against this same rubric and the sample-topic set
below reused as a first real smoke test.

(By contrast, `get_embeddings()` with the local `sentence-transformers`
backend required no API key and *was* exercised for real — see the
companion clustering recommendation doc, `docs/phase2-clustering-recommendation.md`.)

## Prompt design

`newsresearch/llm/prompts/subtopic_propose.txt` (full text also in the
committed file):

```
You are the Subtopic Agent in a news research pipeline. Given a broad news
topic, propose a set of distinct, newsworthy subtopics: specific angles,
events, or developments that a reader following this topic would want
broken out separately, rather than lumped into one undifferentiated feed.

Topic: {topic}

Propose exactly {n_candidates} candidate subtopics. Follow these rules:

1. Each candidate must be a specific, concrete angle on the topic (an event,
   policy action, dispute, product, region, or storyline) -- not a
   restatement of the topic itself, and not a generic sub-heading like
   "latest news" or "background."
2. Candidates must be non-overlapping: no two candidates should describe the
   same underlying story or be a strict subset of another candidate. If two
   angles are this closely related, merge them into a single, more precisely
   worded candidate instead of listing both.
3. Candidates must be topic-relevant. Do not propose a candidate about a
   different news topic that merely happens to co-occur with this one.
4. Candidates describe *what is being reported on* -- never a political
   leaning, outlet, or audience ("the conservative angle," "coverage from
   left-leaning outlets," etc. are not valid candidates). Framing and bias
   are evaluated later, by a different stage, over the actual articles; this
   step only maps out the newsworthy angles themselves.
5. Each candidate is a short label (roughly 3-10 words) plus a one-sentence
   rationale explaining why it is a distinct, reportable angle on the topic.
6. If you cannot identify {n_candidates} genuinely distinct angles, propose
   fewer rather than padding the list with near-duplicates or filler.

Return the candidates as a structured list, one entry per candidate, with a
`label` and a `rationale` field each.
```

Loadable via `ChatPromptTemplate.from_template()` per the `example.txt`
convention (plain `{variable}` placeholders, no vendor-specific formatting).
Two template variables: `{topic}` (the run's topic string) and
`{n_candidates}` (see below).

### Design choices and the specific failure modes each rule addresses

- **Rule 1 (concrete angle, not a restatement/generic heading).** Addresses
  the most likely failure mode for an under-specified prompt: an LLM asked
  for "subtopics" of e.g. "climate policy" tends to default to generic
  buckets ("Overview", "Recent developments", "Background") that carry no
  real distinguishing signal and would all map to the same broad-fetch
  cluster during Task 2.2.3's reconciliation — producing a Gate 1 candidate
  list that looks plausible on the page but reconciles down to almost
  nothing.
- **Rule 2 (non-overlapping, explicit merge instruction) + rule 6 (fewer is
  fine).** Directly targets the TRD's stated reconciliation step (merge
  candidates mapping to the same cluster). Asking the LLM to self-merge
  near-duplicates *before* generation, rather than relying entirely on
  downstream embedding-based reconciliation to catch it, should reduce how
  often Task 2.2.3's merge branch has to fire, and reduces the odds a
  padded, repetitive candidate list makes Gate 1 harder for a human to scan.
- **Rule 3 (topic-relevant only).** Guards against a known LLM tendency to
  free-associate to adjacent-but-different topics (e.g. proposing "electric
  vehicle subsidies" as a subtopic of a "federal budget deficit" topic just
  because EVs are budget-adjacent) — this would otherwise produce a
  candidate with genuinely zero supporting articles once the real broad
  fetch runs, guaranteed to hit Task 2.2.3's "drop" branch. Better to not
  propose it in the first place than rely on reconciliation to clean it up.
- **Rule 4 (never a political-leaning/outlet/audience label).** Directly
  guards the PRD risk table's flagged concern about *not* quietly
  re-introducing a left/center/right scalar framing, extended one stage
  earlier than where it's usually discussed (Bias & Framing Agent, Phase 4).
  A subtopic list that included e.g. "the conservative reaction" as a
  candidate would smuggle a scalar political axis into a stage that's
  supposed to be about newsworthy angles, not ideology — and Gate 1's human
  reviewer would see it presented as if it were a legitimate content
  category. This rule exists specifically to prevent that, not as
  boilerplate.
- **N as a parameter, not hardcoded.** `{n_candidates}` is left as a prompt
  variable rather than a fixed number in the text, matching NFR-5's
  configurability requirement; recommended default below.

### Recommended default for `n_candidates`

Not itself a `Settings.clustering.*` or `Settings.pipeline.*` field today —
`Settings.pipeline.max_subtopics` (default 5) is the post-reconciliation cap,
not the pre-reconciliation LLM proposal count. Recommend the Subtopic Agent
call this prompt with `n_candidates = max_subtopics + 3` (default: **8**
given `max_subtopics=5`) — enough raw candidates that Task 2.2.3's
merge/split/drop reconciliation has real material to work with (some
candidates will merge or get dropped for lack of supporting articles), while
still bounded so the LLM isn't asked to invent implausible filler angles
(rule 6 already tells it to propose fewer if it can't find genuinely
distinct ones). This is a starting recommendation, not derived from a real
run — worth revisiting once Task 2.2.3 has real reconciliation-rate data
showing how many of N proposed candidates typically survive to the final
capped list.

## Proposed schema (`llm/schemas.py`-ready, not added there myself)

```python
from pydantic import BaseModel, Field


class SubtopicCandidate(BaseModel):
    """One LLM-proposed candidate subtopic, pre-reconciliation."""

    label: str = Field(
        ...,
        description=(
            "Short (roughly 3-10 word) label naming a specific, concrete "
            "angle on the topic. Never a political-leaning, outlet, or "
            "audience label."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "One sentence explaining why this is a distinct, reportable "
            "angle on the topic, not a restatement of the topic or of "
            "another candidate."
        ),
    )


class SubtopicCandidateList(BaseModel):
    """Structured output of the subtopic-propose prompt."""

    candidates: list[SubtopicCandidate] = Field(
        ...,
        description="Non-overlapping candidate subtopics, at most n_candidates long.",
    )
```

Notes for `backend-engineer` (Task 2.2.1b):
- `model.with_structured_output(SubtopicCandidateList)` should be called
  with the rendered prompt (via `ChatPromptTemplate.from_template(...)`
  reading `subtopic_propose.txt`, filling `{topic}` and `{n_candidates}`).
- `SubtopicCandidateList.candidates` is intentionally allowed to be shorter
  than `n_candidates` (rule 6) — don't validate an exact length; Task
  2.2.3/2.2.4 already handles capping/excess downstream, and generation-time
  under-filling is a legitimate outcome, not an error.
- `rationale` is retained through to Gate 1 candidate display, matching
  the TRD/plan's expectation that a human reviewer at Gate 1 can see *why*
  each candidate was proposed, not just a bare label.

## Sample-topic evaluation (ILLUSTRATIVE / EXPECTED — not a real model call)

Methodology, since a real call wasn't possible here: for each sample topic,
I wrote out a plausible `n_candidates=8` output by hand, simulating what a
capable instruction-following LLM should produce under this prompt, then
graded it against the same rubric a real output should be graded against:
(a) each candidate concrete and specific (rule 1), (b) pairwise
non-overlapping — no two candidates describable as "the same story" (rule
2), (c) genuinely topic-relevant, not free-associated (rule 3), (d) no
political-leaning/outlet/audience candidate slipped in (rule 4). This
methodology substitutes for a real call; it cannot catch failure modes that
only appear in genuine model output (e.g. an LLM's actual tendency to
hedge, over-generalize, or produce fewer than N despite plausible angles
existing) — those need re-verification once Task 2.2.1b has real API
access.

### 1. Political — "2026 US midterm elections"
Illustrative candidates: (1) Senate battleground-state races and control
of the chamber, (2) House redistricting fights following recent court
rulings, (3) Governors' races in swing states, (4) Campaign finance and
outside-spending records set this cycle, (5) Ballot measures on abortion
access, (6) Voter turnout and early-voting trends, (7) Election security
and certification disputes, (8) Down-ballot state legislature control
fights.
Judgment: distinct and non-overlapping on inspection — each names a
different institutional layer (Senate/House/governors/state legislatures)
or a different cross-cutting theme (finance, ballot measures, turnout,
security), none is a subset of another. Rule 4 held: no candidate is framed
as "the [party]'s chances" or by outlet/ideology. Plausible.

### 2. Science/tech — "quantum computing breakthroughs"
Illustrative candidates: (1) Error-correction milestones reducing qubit
noise, (2) A specific vendor's new qubit-count record announcement, (3)
Quantum computing's implications for current encryption standards, (4)
Government funding and national quantum initiatives, (5) Commercial
quantum-cloud-access partnerships with enterprise customers, (6) Talent
competition and researcher poaching between labs, (7) Skepticism/critique
of overstated near-term claims, (8) Materials-science advances enabling
new qubit designs.
Judgment: reasonably distinct; (1) and (8) both touch qubit hardware but
from different angles (error correction vs. materials) — borderline, a
real model might legitimately merge these per rule 2, which would be a
correct outcome, not a failure. (7) is a notable, useful candidate: a
"skepticism/critique" angle is a real newsworthy angle (this is genuinely
reported on), distinct from a political-leaning label under rule 4 —
worth checking in a real run that the LLM doesn't confuse "critical
coverage exists as a storyline" with prohibited framing-by-leaning; the
prompt's rule 4 wording ("what is being reported on") should keep this
distinction but is worth a real spot-check.

### 3. Business — "commercial airline industry mergers"
Illustrative candidates: (1) A specific pending merger's antitrust review
status, (2) Labor union reactions and pilot-seniority integration
disputes, (3) Route overlap and hub-consolidation concerns, (4) Loyalty
program merger terms affecting frequent flyers, (5) A rival airline's
competing acquisition bid, (6) Airport gate-slot divestiture requirements,
(7) Credit-rating agency reactions to merged carriers' debt load, (8)
International regulatory approval in a second jurisdiction.
Judgment: good — concrete and topic-relevant, no free-association into
unrelated aviation stories (e.g. unrelated safety incidents), no
leaning/outlet candidates. All 8 plausibly distinct.

### 4. International/conflict — "Middle East ceasefire negotiations"
Illustrative candidates: (1) Specific mediating country's diplomatic
shuttle efforts, (2) Hostage/prisoner exchange terms under discussion, (3)
Humanitarian aid corridor access disputes, (4) Domestic political pressure
on negotiating parties' leadership, (5) A specific violation/incident
threatening to collapse talks, (6) Reconstruction funding pledges
contingent on a deal, (7) Reactions from regional allies not directly at
the table, (8) UN Security Council resolution language disputes.
Judgment: distinct, concrete, no leaning-based candidates (this is the
domain where rule 4 is most likely to be tested by a real model, since
"which side's framing" is an easy trap for an LLM to fall into on a
conflict topic — worth the closest real-call scrutiny in Task 2.2.1b).

## What this evaluation cannot tell us (be explicit about the gap)

Since none of the above are real model outputs, this exercise validates
that the prompt's *instructions* are internally consistent and produce a
plausible target shape when followed correctly — it does not validate that
a real LLM reliably follows them (e.g., real models sometimes ignore
"propose fewer if you can't find N," or drift into rule-4-violating
framing language despite the explicit prohibition, especially on
politically charged topics like sample topics 1 and 4 above). The first
real Task 2.2.1b integration should specifically re-run these four sample
topics (plus the mandatory Langfuse trace check from that task's acceptance
criterion) and confirm this rubric holds against actual output, not just
the illustrative simulation here.
