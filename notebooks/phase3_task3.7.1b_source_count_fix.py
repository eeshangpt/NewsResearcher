"""Task 3.7.1b -- fix + validate summarization prompt rule 5 (source-count
miscounting bug found in Task 3.7.1, notebooks/phase3_task3.7.1_quality_
spotcheck.md).

Reproduces the exact 6 real clusters from that spot-check (29, 14, 17 wrong;
23, 27, 9 correct) straight from the still-live real Postgres rows under
subtopic_id='task371-leipzig-drone' (NEWSRESEARCH_DATABASE_URL), runs the OLD
prompt (git-shown, for the record) and the NEW prompt (current
llm/prompts/summarization.txt, rule 5 rewritten) via real
get_chat_model("summarization") calls -- no mocked output, this is a
faithfulness bug and only real model output proves the fix.

Also re-runs the new prompt against Task 3.6.1a's original 8-cluster fixture
sample (notebooks/phase3_summarization_review.py) to check for regressions
on the original validation set.

No article full text is touched anywhere in this script -- only
`claim_text` + `domain` strings already scoped that way by
`persistence/claim_clusters.py::read_cluster_article_relations` and the
existing fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from psycopg_pool import ConnectionPool

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model

_HERE = Path(__file__).resolve().parent
_PROMPTS_DIR = _HERE.parent / "newsresearch" / "llm" / "prompts"

SUBTOPIC_ID = "task371-leipzig-drone"

# Exact clusters documented in phase3_task3.7.1_quality_spotcheck.md.
FAILING_CLUSTERS = ["29", "14", "17"]  # wrong count in original run
PASSING_CLUSTERS = ["23", "27", "9"]  # correct count in original run

# Old rule 5 text, verbatim from commit 1e81649 (phase3_summarization_prompt_review.md /
# the version Task 3.7.1's spot-check ran against) -- kept here only for this
# comparison script, not written back to the prompt file.
OLD_RULE_5 = (
    "5. State how many of the sources listed actually assert each fact you\n"
    "   summarize -- do not imply broader agreement than the claims listed\n"
    "   show (e.g. do not write \"multiple outlets report\" for a fact only one\n"
    "   listed source states)."
)


def _build_prompt(rule_5: str) -> ChatPromptTemplate:
    new_text = (_PROMPTS_DIR / "summarization.txt").read_text()
    # Splice out the current rule 5 block (from "5." up to but not including "6.")
    lines = new_text.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("5."))
    end = next(i for i, l in enumerate(lines) if l.startswith("6."))
    old_variant = "".join(lines[:start]) + rule_5 + "\n\n" + "".join(lines[end:])
    return ChatPromptTemplate.from_template(old_variant)


def format_claims(rows: list[dict]) -> str:
    return "\n".join(f"- {r['domain']}: {r['claim_text']}" for r in rows if r["relation"] == "asserts")


def summarize(rows: list[dict], prompt: ChatPromptTemplate) -> str:
    model = get_chat_model("summarization")
    chain = prompt | model
    return chain.invoke({"claims": format_claims(rows)}).content


def domain_count(rows: list[dict]) -> int:
    return len({r["domain"] for r in rows if r["relation"] == "asserts"})


def run_db_clusters() -> dict:
    from newsresearch.persistence.claim_clusters import read_cluster_article_relations

    settings = Settings()
    pool = ConnectionPool(settings.database_url, min_size=1, max_size=2)
    current_text = (_PROMPTS_DIR / "summarization.txt").read_text()
    new_prompt = ChatPromptTemplate.from_template(current_text)
    old_prompt = _build_prompt(OLD_RULE_5)

    out = {}
    for label in FAILING_CLUSTERS + PASSING_CLUSTERS:
        cluster_id = f"{SUBTOPIC_ID}:{label}"
        rows = read_cluster_article_relations(pool, cluster_id)
        n_domains = domain_count(rows)
        old_summary = summarize(rows, old_prompt)
        # 3 independent new-prompt calls per cluster to check LLM-sampling
        # stability, not just a single lucky draw.
        new_summaries = [summarize(rows, new_prompt) for _ in range(3)]
        out[label] = {
            "n_distinct_asserting_domains": n_domains,
            "domains": sorted({r["domain"] for r in rows if r["relation"] == "asserts"}),
            "old_prompt_summary": old_summary,
            "new_prompt_summaries": new_summaries,
        }
        print(f"\n=== cluster {label} ({n_domains} distinct asserting domains: "
              f"{out[label]['domains']}) ===")
        print(f"  OLD -> {old_summary}")
        for i, s in enumerate(new_summaries):
            print(f"  NEW[{i}] -> {s}")
    pool.close()
    return out


def run_361a_regression() -> dict:
    """Re-run the new prompt against the original 3.6.1a 8-cluster fixture sample."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase3_summarization_review", _HERE / "phase3_summarization_review.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    clusters = mod.load_clusters()
    sample_labels = [
        "21", "31", "2", "9", "34", "0", "15", "1",
    ]
    out = {}
    for label in sample_labels:
        entry = clusters[label]
        summary = mod.summarize(entry["claims"])
        out[label] = {"claims": entry["claims"], "summary": summary}
        print(f"\n=== 3.6.1a regression cluster {label} ({len(entry['claims'])} claims) ===")
        print(f"  -> {summary}")
    return out


def main() -> None:
    db_results = run_db_clusters()
    (_HERE / "phase3_task3.7.1b_db_results.json").write_text(json.dumps(db_results, indent=2))

    regression_results = run_361a_regression()
    (_HERE / "phase3_task3.7.1b_361a_regression.json").write_text(
        json.dumps(regression_results, indent=2)
    )


if __name__ == "__main__":
    main()
