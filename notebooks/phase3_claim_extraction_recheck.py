"""Re-run revised claim_extraction.txt (rule 9 added) against the two
articles that showed the highest claim counts in the first pass, to check
whether the fix reduces low-value color/sentiment claims without dropping
substantive ones. See phase3_claim_extraction_prompt_review.md Iteration 2.

Output: phase3_claim_extraction_iteration2_recheck.json (extracted claims +
short url/title metadata only -- no full article text persisted).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim_schema_draft import ClaimList  # noqa: E402

from newsresearch.llm.models import get_chat_model  # noqa: E402
from newsresearch.sourcing.fulltext import fetch_fulltext  # noqa: E402

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "newsresearch"
    / "llm"
    / "prompts"
    / "claim_extraction.txt"
)
OUT_PATH = Path(__file__).resolve().parent / "phase3_claim_extraction_iteration2_recheck.json"

URLS = [
    "https://www.bbc.co.uk/news/articles/c5yvqk69enko?at_medium=RSS&at_campaign=rss",  # Kashmir, was 65
    "https://www.theguardian.com/australia-news/2026/aug/03/one-nation-candidates-victoria-state-election-2026-ntwnfb",  # One Nation, was 44
]


def main() -> None:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PROMPT_PATH.read_text())
    model = get_chat_model("claim_extraction").with_structured_output(ClaimList)

    samples = []
    for url in URLS:
        fulltext = fetch_fulltext(url)
        if not fulltext:
            print(f"skip: {url}")
            continue
        messages = prompt.format_messages(article_text=fulltext)
        result: ClaimList = model.invoke(messages)
        samples.append(
            {"url": url, "n_claims": len(result.claims), "claims": [c.model_dump() for c in result.claims]}
        )
        print(f"{url} -> {len(result.claims)} claims (was 65/44 pre-fix)")

    OUT_PATH.write_text(json.dumps(samples, indent=2, default=str))


if __name__ == "__main__":
    main()
