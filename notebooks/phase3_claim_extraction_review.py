"""Real-data fetch + prompt-eval harness for Task 3.2.1a (claim extraction).

Pulls real articles from trusted-outlet RSS (`sourcing/rss.py`), fetches
real full text via `sourcing/fulltext.py::fetch_fulltext` (in-memory only,
per NFR/FR-25 -- this script never writes full article body text to disk;
only extracted `Claim` objects and short excerpts go into the sample-output
JSON committed alongside this script), then runs the claim-extraction
prompt + schema end to end via `get_chat_model("claim_extraction")
.with_structured_output(Claim)`.

Real data, several varied topics/outlets, one run each on 2026-08-03 -- not
a golden-dataset eval (PRD Sec.7 names that gap explicitly for v1). Run
again to reproduce: `uv run python notebooks/phase3_claim_extraction_review.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim_schema_draft import ClaimList  # noqa: E402  (draft schema, see write-up)

from newsresearch.llm.models import get_chat_model  # noqa: E402
from newsresearch.sourcing.fulltext import fetch_fulltext  # noqa: E402
from newsresearch.sourcing.rss import OUTLET_RSS_FEEDS, fetch_trusted_rss  # noqa: E402

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "newsresearch"
    / "llm"
    / "prompts"
    / "claim_extraction.txt"
)
OUT_PATH = Path(__file__).resolve().parent / "phase3_claim_extraction_samples.json"

# Broad keyword per topic so RSS's client-side title/summary substring match
# actually returns entries (trusted feeds are whole-outlet firehoses, not
# keyword-searchable) -- picks up whatever's currently live across varied
# beats (politics, health, tech, environment, business).
TOPICS = ["election", "health", "AI", "climate", "economy"]


def collect_candidate_articles(max_per_topic: int = 2) -> list[dict]:
    seen_urls: set[str] = set()
    picked: list[dict] = []
    for topic in TOPICS:
        articles = fetch_trusted_rss([topic], lookback_days=14, feeds=OUTLET_RSS_FEEDS)
        count = 0
        for article in articles:
            if article["url"] in seen_urls:
                continue
            seen_urls.add(article["url"])
            picked.append({**article, "topic_query": topic})
            count += 1
            if count >= max_per_topic:
                break
    return picked


def build_prompt_and_model():
    template_text = PROMPT_PATH.read_text()
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("claim_extraction").with_structured_output(ClaimList)
    return prompt, model


def main() -> None:
    candidates = collect_candidate_articles()
    print(f"Collected {len(candidates)} candidate article URLs across {len(TOPICS)} topics.")

    prompt, model = build_prompt_and_model()

    samples = []
    for article in candidates:
        url = article["url"]
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

        samples.append(
            {
                "url": url,
                "domain": article["domain"],
                "title": article["title"],
                "n_claims": len(result.claims),
                "claims": [c.model_dump() for c in result.claims],
            }
        )
        print(f"OK: {url} -> {len(result.claims)} claims")

    # Only extracted claims + short metadata (url/title/domain) are written --
    # never full article body text, per the no-full-text-storage rule.
    OUT_PATH.write_text(json.dumps(samples, indent=2, default=str))
    print(f"Wrote {len(samples)} article samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
