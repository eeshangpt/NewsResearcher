from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "newsresearch" / "llm" / "prompts"


def test_example_prompt_template_loads_via_chat_prompt_template():
    template_text = (PROMPTS_DIR / "example.txt").read_text()

    prompt = ChatPromptTemplate.from_template(template_text)

    assert "topic" in prompt.input_variables


def test_schemas_module_imports_cleanly():
    import newsresearch.llm.schemas  # noqa: F401


def test_claim_extraction_prompt_template_loads_and_renders():
    template_text = (PROMPTS_DIR / "claim_extraction.txt").read_text()

    prompt = ChatPromptTemplate.from_template(template_text)

    assert "article_text" in prompt.input_variables
    rendered = prompt.format(article_text="Sample article body about widgets.")
    assert "Sample article body about widgets." in rendered
