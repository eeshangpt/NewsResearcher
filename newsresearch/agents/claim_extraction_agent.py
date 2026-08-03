"""Claim Extraction Agent (Phase 3, Story 3.2, Task 3.2.1b).

Wires the data-scientist-authored `llm/prompts/claim_extraction.txt` (Task
3.2.1a, see `notebooks/phase3_claim_extraction_prompt_review.md`) through
`get_chat_model("claim_extraction").with_structured_output(ClaimList)`, per
`agents/subtopic_agent.py::propose_candidates`'s established pattern.

Takes one article's full text (from Task 3.1.1's `sourcing/fulltext.py::
fetch_fulltext`) and returns a `ClaimList` of schema-conformant `Claim`
objects. Not yet wired into `graph/build.py`'s compiled topology -- Story
3.2's acceptance line only requires the agent function itself to exist and
be callable per-article, matching Task 3.1.1's PR #105 scoping precedent.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model
from newsresearch.llm.schemas import ClaimList
from newsresearch.observability.langfuse_setup import get_langfuse_callback_handler, trace_metadata

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def extract_claims(
    article_text: str,
    *,
    run_id: str = "dev",
    settings: Settings | None = None,
    config: RunnableConfig | None = None,
) -> ClaimList:
    """Extract structured claims from one article's full body text.

    Traced via Langfuse per the CLI's established `get_langfuse_callback_handler`
    + `trace_metadata` convention, tagged with `run_id` and `stage=claim_extraction`.
    Accepts an ambient `config` and merges it with the Langfuse callback/metadata
    this call attaches, via `merge_configs`, rather than replacing it outright --
    so already-attached callbacks (e.g. `cost_callback.py`'s handler) still fire
    on this nested LLM call.
    """
    settings = settings or Settings()

    template_text = (_PROMPTS_DIR / "claim_extraction.txt").read_text()
    prompt = ChatPromptTemplate.from_template(template_text)
    model = get_chat_model("claim_extraction").with_structured_output(ClaimList)
    chain = prompt | model

    call_config: RunnableConfig = {
        "callbacks": [get_langfuse_callback_handler(settings)],
        "metadata": {**trace_metadata(run_id), "stage": "claim_extraction"},
    }
    merged_config = merge_configs(config, call_config)
    return chain.invoke({"article_text": article_text}, config=merged_config)
