"""Task 3.3.1a — real claim-text corpus for claim-clustering hyperparameter sweep.

Pulls real articles for a handful of *narrow, single-subtopic* GDELT queries
(narrower than Task 3.2.1a's broad-topic RSS pulls, since claim clustering
operates on one subtopic's article set, per TRD Task 3.3.1's actual input
contract), fetches full text in-memory only (`sourcing/fulltext.py`, never
persisted per the no-full-text-storage rule), and runs the already-designed
Task 3.2.1a claim-extraction prompt+schema
(`llm/prompts/claim_extraction.txt` + `notebooks/claim_schema_draft.py`,
reused unchanged) against each article via
`get_chat_model("claim_extraction").with_structured_output(ClaimList)`.

Only extracted `Claim` objects + short article metadata (url/domain/title)
are written to `phase3_claim_clustering_corpus.json` -- never article body
text, consistent with every prior Phase 3 data-scientist script.

Real data, one run, 2026-08-03 -- not a golden-dataset eval (PRD Sec.7 names
this gap explicitly for v1). Run again to reproduce:
`uv run python notebooks/phase3_claim_clustering_fetch.py`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim_schema_draft import ClaimList  # noqa: E402

from newsresearch.llm.models import get_chat_model  # noqa: E402
from newsresearch.sourcing.fulltext import fetch_fulltext  # noqa: E402
from newsresearch.sourcing.gdelt import GDELTError, fetch as gdelt_fetch  # noqa: E402
from newsresearch.sourcing.rss import OUTLET_RSS_FEEDS, fetch_trusted_rss  # noqa: E402

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "newsresearch" / "llm" / "prompts" / "claim_extraction.txt"
)
OUT_PATH = Path(__file__).resolve().parent / "phase3_claim_clustering_corpus.json"

# Narrow, single-story subtopic-style queries (not broad topic keywords like
# 3.2.1a's ["election", "health", ...]) -- this is what Task 3.3.1's actual
# input looks like: one subtopic's article set, where multiple outlets cover
# the *same* specific facts (giving genuine claim-level overlap to cluster),
# not four unrelated beats.
SUBTOPICS: dict[str, list[str]] = {
    "openai_gpt5_release": ["OpenAI GPT-5"],
    "fed_rate_decision": ["Federal Reserve", "interest rate"],
    "ukraine_ceasefire": ["Ukraine ceasefire", "Russia Ukraine"],
    "tesla_earnings": ["Tesla earnings", "Tesla quarterly"],
}
LOOKBACK_DAYS = 14
MAX_ARTICLES_PER_SUBTOPIC = 8


def collect_candidate_articles() -> dict[str, list[dict]]:
    """GDELT primary, trusted-RSS fallback per subtopic.

    GDELT hit a sustained IP-level rate-limit cooldown (`GDELTError`, all 5
    retries exhausted, not just transient 429s) partway through this run --
    falls back to `sourcing/rss.py::fetch_trusted_rss` (same soft-fail
    pattern `sourcing_agent.py` already uses for this exact failure mode)
    so a temporary GDELT block doesn't stall corpus collection entirely.
    """
    by_subtopic: dict[str, list[dict]] = {}
    for subtopic, keywords in SUBTOPICS.items():
        try:
            articles = gdelt_fetch(keywords, LOOKBACK_DAYS)
            if articles:
                by_subtopic[subtopic] = articles[:MAX_ARTICLES_PER_SUBTOPIC]
                continue
        except GDELTError as exc:
            print(f"GDELT failed for subtopic {subtopic!r}, falling back to RSS: {exc}")

        rss_articles = fetch_trusted_rss(keywords, LOOKBACK_DAYS, feeds=OUTLET_RSS_FEEDS)
        by_subtopic[subtopic] = rss_articles[:MAX_ARTICLES_PER_SUBTOPIC]
    return by_subtopic


def build_prompt_and_model():
    from langchain_core.prompts import ChatPromptTemplate

    template_text = PROMPT_PATH.read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("claim_extraction").with_structured_output(ClaimList)
    return prompt, model


def main() -> None:
    # Resume support: skip re-fetching/re-extracting subtopics already
    # present in a prior run's output (GDELT's sustained rate-limit cooldown
    # made a single-pass run unreliable -- avoids burning LLM calls redoing
    # subtopics that already succeeded).
    corpus: list[dict] = []
    done_subtopics: set[str] = set()
    if OUT_PATH.exists():
        corpus = json.loads(OUT_PATH.read_text())
        done_subtopics = {a["subtopic"] for a in corpus}
        print(f"Resuming: {len(corpus)} articles already done for subtopics {done_subtopics}")

    by_subtopic = collect_candidate_articles()
    for subtopic, articles in by_subtopic.items():
        print(f"{subtopic}: {len(articles)} candidate URLs")

    prompt, model = build_prompt_and_model()

    for subtopic, articles in by_subtopic.items():
        if subtopic in done_subtopics:
            print(f"skip subtopic {subtopic!r} (already in corpus)")
            continue
        for article in articles:
            url = article.get("url")
            if not url:
                continue
            fulltext = fetch_fulltext(url)
            if not fulltext or len(fulltext) < 500:
                print(f"skip (no/short fulltext): {url}")
                continue

            messages = prompt.format_messages(article_text=fulltext)
            try:
                result: ClaimList = model.invoke(messages)
            except Exception as exc:
                print(f"skip (LLM call failed): {url} ({exc})")
                continue

            corpus.append(
                {
                    "subtopic": subtopic,
                    "url": url,
                    "domain": article.get("domain"),
                    "title": article.get("title"),
                    "n_claims": len(result.claims),
                    "claims": [c.model_dump() for c in result.claims],
                }
            )
            print(f"OK [{subtopic}]: {url} -> {len(result.claims)} claims")

    OUT_PATH.write_text(json.dumps(corpus, indent=2, default=str))
    total_claims = sum(a["n_claims"] for a in corpus)
    print(f"Wrote {len(corpus)} articles / {total_claims} claims across {len(by_subtopic)} subtopics to {OUT_PATH}")


if __name__ == "__main__":
    main()
