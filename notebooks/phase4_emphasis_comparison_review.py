"""Task 4.2.1a -- is a dedicated emphasis-comparison prompt needed at all?

TRD 4.6's third output bullet is "for each claim cluster, which sources
include it, which omit it, and where language/emphasis differs on shared
claims". This script tests, against the real persisted
`task371-leipzig-drone` subtopic, how much of that bullet Task 4.1.1a's
already-shipped `divergence`/`divergence_note` output plus the deterministic
`claim_cluster_articles` relations already cover, and whether a second,
dedicated LLM pass adds anything the first one does not.

Modes:

    --coverage   measure divergence_note against the emphasis bullet, using
                 the already-saved 4.1.1a run (no LLM calls, no cost)
    --assemble   render the deterministic include/omit/emphasis paragraph
                 (no LLM call) for every multi-article cluster
    --candidate  run the candidate dedicated prompt below for real, on the
                 same clusters, and save its output for head-to-head reading
    --compare    candidate vs. the shipped divergence_note, side by side
    --rule9      re-run the shipped bias/framing prompt with this branch's
                 amended rule 9 (the actual recommendation); --baseline is
                 the same run with rule 9 reverted to master's text
    --density    quote density of any existing samples file

Only `claim_text` + `article_id` + `domain` ever leave the DB -- no article
full text is touched, fetched, or written anywhere (FR-25).

Usage:
    uv run python notebooks/phase4_emphasis_comparison_review.py --coverage
    uv run python notebooks/phase4_emphasis_comparison_review.py --assemble
    uv run python notebooks/phase4_emphasis_comparison_review.py --candidate --limit 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.persistence.claim_clusters import (
    read_cluster_article_relations,
    read_subtopic_cluster_ids,
)
from newsresearch.persistence.db import init_db

HERE = Path(__file__).resolve().parent
SUBTOPIC = "task371-leipzig-drone"
SAMPLES_411A = HERE / "phase4_bias_framing_samples.json"
CANDIDATE_OUT = HERE / "phase4_emphasis_comparison_samples.json"


# --------------------------------------------------------------------------
# The candidate dedicated prompt, kept here rather than in llm/prompts/.
# It is an experiment, not a recommendation -- see the write-up's verdict.
# Deliberately given the best shot available: it is handed the omission
# roster, which `bias_framing.txt` never sees, so it can in principle say
# something the shipped output structurally cannot.
# --------------------------------------------------------------------------
CANDIDATE_PROMPT = """You are comparing how different news articles covered the same
claim. You are given one claim cluster: the articles that made a claim in it (with
their exact claim text), and -- as established fact -- the articles from this subtopic
that did not cover it at all.

{cluster}

Write one short comparison for this cluster, for a reader deciding which coverage to
trust, following these rules.

1. The "Articles that did not cover this claim" line is authoritative and already
   correct. Do not recount it, do not derive it from the claim lines, and do not
   question it. Refer to those articles only as given.
2. Describe only what requires reading the claim text: which wording, attribution, or
   surrounding context each covering article chose, and where those choices differ.
3. Name specific article ids and quote the actual differing words. Never state a count
   of articles or sources ("two outlets say...", "most sources agree..."); name them.
4. Two articles can come from the same source domain -- for instance a news report and
   an opinion column from one outlet. Treat them as separate articles. A difference
   between two articles from one domain is a difference between those two pieces, not
   evidence about the outlet.
5. The grouping you are given is imperfect and often contains claims that share only a
   topic, not the same underlying fact. Articles covering different facts are not
   disagreeing. Say so plainly when that is what you see.
6. Do not assign, imply, or hint at a political leaning, ideological position, or
   partisan alignment for any article, outlet, or domain. Do not rate, rank, or grade
   any article or outlet, or call one more biased, trustworthy, or objective than
   another. Do not judge whether any claim is true.
7. Write plainly. Never mention clusters, algorithms, embeddings, or other pipeline
   mechanics.
"""


class EmphasisComparison(BaseModel):
    """Candidate 4.2.1a output. One free-text comparison per cluster."""

    cluster_id: str = Field(description="The cluster's id, exactly as given.")
    comparison: str = Field(
        description=(
            "Two to four sentences comparing how the covering articles worded, "
            "attributed and contextualised this claim, naming article ids and "
            "quoting the differing words."
        )
    )


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
def load(pool) -> dict[str, dict]:
    """cluster_id -> {"asserts": rows, "omits": rows}. Both come from the DB."""
    out: dict[str, dict] = {}
    for cid in read_subtopic_cluster_ids(pool, SUBTOPIC):
        rows = read_cluster_article_relations(pool, cid)
        asserts = [r for r in rows if r["relation"] == "asserts"]
        if asserts:
            out[cid] = {"asserts": asserts, "omits": [r for r in rows if r["relation"] == "omits"]}
    return out


def distinct(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(r["article_id"] for r in rows))


def format_cluster(cid: str, data: dict) -> str:
    """The candidate prompt's {cluster} block. Same roster pattern as 4.1.1a."""
    a, o = data["asserts"], data["omits"]
    lines = [
        f"Cluster {cid}",
        f"Articles that made this claim ({len(distinct(a))}): " + ", ".join(distinct(a)),
        "Articles that did not cover this claim: "
        + (", ".join(f"{r['article_id']} ({r['domain']})" for r in o) or "none"),
        "Claims (article id | source domain | claim):",
    ]
    lines += [f"- {r['article_id']} | {r['domain']} | {r['claim_text']}" for r in a]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Mode 1: does divergence_note already answer the emphasis bullet?
# --------------------------------------------------------------------------
def coverage() -> None:
    payload = json.loads(SAMPLES_411A.read_text())
    inputs = payload["input_clusters"]
    notes: list[tuple[str, int, str, str]] = []
    for r in payload["results"]:
        for c in r["output"]["clusters"]:
            cid = c["cluster_id"]
            n = len({x["article_id"] for x in inputs.get(cid, [])})
            notes.append((cid, n, c["divergence"], c["divergence_note"]))

    multi = [n for n in notes if n[1] >= 2]
    print(f"{len(notes)} cluster-outputs, {len(multi)} on multi-article clusters\n")

    named = quoted = worded = 0
    for cid, _n, _div, note in multi:
        ids = set(re.findall(r"leipzig-drone-\d+", note))
        has_quote = bool(re.search(r"['‘’\"]", note))
        # a note "points at wording" if it quotes a span that really is in the
        # claim text of one of this cluster's articles -- not just any quote marks
        spans = [s for s in re.findall(r"'([^']{6,})'", note)]
        src = " ".join(x["claim_text"].lower() for x in inputs.get(cid, []))
        real = any(s.lower() in src for s in spans)
        named += len(ids) >= 2
        quoted += has_quote
        worded += real
    print(f"names >=2 distinct article ids : {named}/{len(multi)} ({named / len(multi):.0%})")
    print(f"contains a quoted span         : {quoted}/{len(multi)} ({quoted / len(multi):.0%})")
    print(f"quotes wording really in claims: {worded}/{len(multi)} ({worded / len(multi):.0%})")

    from collections import Counter

    print("\ndivergence distribution:", dict(Counter(n[2] for n in notes)))


# --------------------------------------------------------------------------
# Mode 2: the deterministic assembly -- what 4.2.1b can render with no LLM call
# --------------------------------------------------------------------------
def render(cid: str, data: dict, framing: dict | None) -> str:
    """include / omit / emphasis, assembled from DB rows + 4.1.1a's own text.

    The include and omit halves are pure DB. The emphasis half is 4.1.1a's
    already-generated `divergence_note`. No LLM call happens here.
    """
    a, o = data["asserts"], data["omits"]
    dom = {r["article_id"]: r["domain"] for r in a + o}
    covered = ", ".join(f"{i} ({dom[i]})" for i in distinct(a))
    omitted = ", ".join(f"{r['article_id']} ({r['domain']})" for r in o) or "none"
    note = (framing or {}).get("divergence_note", "(no framing output for this cluster)")
    div = (framing or {}).get("divergence", "?")
    return (
        f"Cluster {cid}\n"
        f"  Covered by : {covered}\n"
        f"  Not covered: {omitted}\n"
        f"  Difference : [{div}] {note}"
    )


def assemble() -> None:
    pool = init_db(Settings().database_url)
    data = load(pool)
    framings = {}
    payload = json.loads(SAMPLES_411A.read_text())
    for r in payload["results"]:
        if r["run"] != 0:
            continue
        for c in r["output"]["clusters"]:
            framings[c["cluster_id"]] = c

    n_omit = sum(1 for d in data.values() if d["omits"])
    print(f"{len(data)} clusters; {n_omit} have at least one omitting article\n")
    for cid, d in data.items():
        if len(distinct(d["asserts"])) >= 2:
            print(render(cid, d, framings.get(cid)), "\n")


# --------------------------------------------------------------------------
# Mode 3: run the candidate dedicated prompt for real
# --------------------------------------------------------------------------
def candidate(limit: int) -> None:
    settings = Settings()
    pool = init_db(settings.database_url)
    data = load(pool)
    targets = [c for c, d in data.items() if len(distinct(d["asserts"])) >= 2][:limit]

    chain = ChatPromptTemplate.from_template(CANDIDATE_PROMPT) | get_chat_model(
        "bias_framing"
    ).with_structured_output(EmphasisComparison)

    results = []
    for cid in targets:
        res = chain.invoke({"cluster": format_cluster(cid, data[cid])})
        print(f"=== {cid}\n{res.comparison}\n")
        results.append(res.model_dump())
    CANDIDATE_OUT.write_text(
        json.dumps(
            {
                "subtopic": SUBTOPIC,
                "model": settings.models.bias_framing,
                "prompt": CANDIDATE_PROMPT,
                "inputs": {c: format_cluster(c, data[c]) for c in targets},
                "results": results,
            },
            indent=1,
        )
    )
    print(f"wrote {CANDIDATE_OUT}")


# --------------------------------------------------------------------------
# Mode 4: head-to-head -- candidate output vs. the shipped divergence_note
# --------------------------------------------------------------------------
def compare() -> None:
    cand = json.loads(CANDIDATE_OUT.read_text())
    base = json.loads(SAMPLES_411A.read_text())
    inputs = base["input_clusters"]
    notes = {}
    for r in base["results"]:
        if r["run"] == 0:
            for c in r["output"]["clusters"]:
                notes[c["cluster_id"]] = c["divergence_note"]

    pool = init_db(Settings().database_url)
    data = load(pool)

    def verbatim_frac(cid: str, text: str) -> tuple[int, int]:
        """Quoted spans that really occur in this cluster's claim text.

        Trailing punctuation is stripped before matching: American-style
        quoting puts the sentence comma *inside* the quote marks, and
        counting that as a non-verbatim quote inflates the defect rate --
        an artefact, not a fabrication. Spans that start or end mid-word
        (apostrophe in "leipzig-drone-1's") are dropped, same reason.
        """
        src = " ".join(x["claim_text"].lower() for x in inputs.get(cid, []))
        spans = [
            s.strip().strip(".,;:").strip()
            for s in re.findall(r"[\"'‘“]([^\"'’”]{8,})[\"'’”]", text)
        ]
        spans = [s for s in spans if len(s) >= 8]
        misses = [s for s in spans if s.lower() not in src]
        for s in misses:
            print(f"    MISS {cid}: {s!r}")
        return len(spans) - len(misses), len(spans)

    rows = []
    for r in cand["results"]:
        cid = r["cluster_id"]
        c_ok, c_n = verbatim_frac(cid, r["comparison"])
        b_ok, b_n = verbatim_frac(cid, notes[cid])
        omitted = {x["article_id"] for x in data[cid]["omits"]}
        mentions_omit = any(o in r["comparison"] for o in omitted)
        rows.append((cid, len(r["comparison"]), len(notes[cid]), c_ok, c_n, b_ok, b_n, mentions_omit))

    print(f"\n{'cluster':<30}{'cand':>6}{'note':>6}  cand-quotes  note-quotes  names-omitter")
    for cid, cl, bl, co, cn, bo, bn, m in rows:
        print(f"{cid:<30}{cl:>6}{bl:>6}      {co}/{cn}          {bo}/{bn}        {m}")
    print(
        f"\ncandidate mean chars {sum(r[1] for r in rows) / len(rows):.0f} "
        f"vs divergence_note {sum(r[2] for r in rows) / len(rows):.0f}"
    )
    print(
        f"verbatim quote accuracy: candidate {sum(r[3] for r in rows)}/{sum(r[4] for r in rows)}, "
        f"divergence_note {sum(r[5] for r in rows)}/{sum(r[6] for r in rows)}"
    )
    print(f"candidate outputs naming an omitting article: {sum(r[7] for r in rows)}/{len(rows)}")


# --------------------------------------------------------------------------
# Mode 5: the actual recommendation -- amend rule 9 of the SHIPPED prompt
# instead of adding a second one. The only measurable advantage the dedicated
# candidate had was quote density; that is a prompt-rule tweak, not an agent.
# --------------------------------------------------------------------------
RULE9_SHIPPED = """9. divergence_note explains your divergence value in one or two sentences,
   naming the specific article ids or source domains involved and pointing
   at the actual differing wording."""

RULE9_AMENDED = """9. divergence_note explains your divergence value in one or two sentences,
   naming the specific article ids or source domains involved and pointing
   at the actual differing wording. When articles differ, quote the
   differing words themselves from each article involved rather than
   describing the difference in your own words -- one article's id, then
   the words it used, then the other article's id and the words it used,
   with both sets of words copied from those articles' own claim lines.
   Each quoted span must be a single unbroken run of words copied
   character for character, under the same rules as rule 7: no ellipsis,
   no stitching, no normalising. Quote a shorter phrase rather than an
   inexact longer one. When the articles do not actually differ, do not
   manufacture a contrast to have something to quote."""

BIAS_PROMPT = HERE.parent / "newsresearch" / "llm" / "prompts" / "bias_framing.txt"
RULE9_OUT = HERE / "phase4_emphasis_rule9_samples.json"


def rule9(repeat: int, amended: bool) -> None:
    """Re-run the shipped bias/framing prompt, optionally with rule 9 amended.

    Batches exactly as production does (`bias_framing_agent._format_batch`,
    `read_subtopic_cluster_ids` order) so the result is comparable to what
    4.2.1b will actually see.
    """
    import sys

    sys.path.insert(0, str(HERE))
    from phase4_bias_framing_review import check as mechanical_check

    from newsresearch.llm.schemas import BiasFramingBatch

    settings = Settings()
    pool = init_db(settings.database_url)
    data = load(pool)

    # This branch's bias_framing.txt already carries the amended rule 9, so
    # `--baseline` is the one that rewrites: amended -> the text on master.
    text = BIAS_PROMPT.read_text()
    assert RULE9_AMENDED in text, "amended rule 9 not found -- prompt changed under me"
    if not amended:
        text = text.replace(RULE9_AMENDED, RULE9_SHIPPED)

    chain = ChatPromptTemplate.from_template(text) | get_chat_model(
        "bias_framing"
    ).with_structured_output(BiasFramingBatch)

    ids = list(data)
    size = settings.models.bias_framing_batch_size
    batches = [ids[i : i + size] for i in range(0, len(ids), size)]

    results = []
    for run_i in range(repeat):
        for b_i, batch in enumerate(batches):
            block = "\n\n".join(
                "\n".join(
                    [
                        f"Cluster {c}",
                        f"Distinct articles in this cluster ({len(distinct(data[c]['asserts']))}): "
                        + ", ".join(distinct(data[c]["asserts"])),
                        "Claims (article id | source domain | claim):",
                    ]
                    + [
                        f"- {r['article_id']} | {r['domain']} | {r['claim_text']}"
                        for r in data[c]["asserts"]
                    ]
                )
                for c in batch
            )
            res = chain.invoke({"clusters": block})
            print(f"run {run_i} batch {b_i}: asked {len(batch)}, got {len(res.clusters)}")
            results.append({"run": run_i, "batch": b_i, "asked": batch, "output": res.model_dump()})

    out = RULE9_OUT if amended else RULE9_OUT.with_name("phase4_emphasis_rule9_baseline.json")
    out.write_text(
        json.dumps(
            {
                "subtopic": SUBTOPIC,
                "batch_size": size,
                "amended_rule_9": amended,
                "model": settings.models.bias_framing,
                "input_clusters": {
                    c: [
                        {"article_id": r["article_id"], "domain": r["domain"], "claim_text": r["claim_text"]}
                        for r in d["asserts"]
                    ]
                    for c, d in data.items()
                },
                "results": results,
            },
            indent=1,
        )
    )
    print(f"wrote {out}")
    mechanical_check(out)
    quote_density(out)


def quote_density(path: Path) -> None:
    """Verbatim-quoted spans per multi-article divergence_note."""
    payload = json.loads(path.read_text())
    inputs = payload["input_clusters"]
    tot = ok = n = 0
    for r in payload["results"]:
        for c in r["output"]["clusters"]:
            cid = c["cluster_id"]
            if len({x["article_id"] for x in inputs.get(cid, [])}) < 2:
                continue
            src = " ".join(x["claim_text"].lower() for x in inputs[cid])
            spans = [
                s.strip().strip(".,;:").strip()
                for s in re.findall(r"[\"'‘“]([^\"'’”]{8,})[\"'’”]", c["divergence_note"])
            ]
            spans = [s for s in spans if len(s) >= 8]
            n += 1
            tot += len(spans)
            ok += sum(s.lower() in src for s in spans)
    print(f"\n{n} multi-article notes: {tot} quoted spans ({tot / n:.2f}/note), {ok} verbatim ({ok / max(tot, 1):.0%})")


def check() -> None:
    """One runnable self-check on the only non-trivial logic here: the renderer.

    Guards the property this whole task turns on -- omission comes from the DB
    rows, never from a model.
    """
    d = {
        "asserts": [
            {"article_id": "a1", "domain": "x.com", "claim_text": "the runway closed"},
            {"article_id": "a2", "domain": "y.com", "claim_text": "the runway was shut"},
        ],
        "omits": [{"article_id": "a3", "domain": "z.com", "claim_text": ""}],
    }
    out = render("c:1", d, {"divergence": "wording_or_emphasis_differs", "divergence_note": "a1 vs a2 wording"})
    assert "a1 (x.com), a2 (y.com)" in out, out
    assert "Not covered: a3 (z.com)" in out, out
    assert "a3" not in out.split("Not covered")[0], "omitting article must not appear as covering"
    out2 = render("c:2", {"asserts": d["asserts"], "omits": []}, None)
    assert "Not covered: none" in out2, out2
    assert "(no framing output" in out2, out2
    print("renderer self-check OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coverage", action="store_true")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--candidate", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--rule9", action="store_true", help="run the shipped prompt with rule 9 amended")
    p.add_argument("--baseline", action="store_true", help="same, unamended, for a same-day control")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--density", type=Path, help="quote density of an existing samples file")
    p.add_argument("--limit", type=int, default=8)
    a = p.parse_args()
    check()
    if a.compare:
        compare()
    if a.density:
        quote_density(a.density)
    if a.rule9 or a.baseline:
        rule9(a.repeat, amended=a.rule9)
    if a.coverage:
        coverage()
    if a.assemble:
        assemble()
    if a.candidate:
        candidate(a.limit)
