from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "newsresearch" / "llm" / "prompts"


def test_example_prompt_template_loads_via_chat_prompt_template():
    template_text = (PROMPTS_DIR / "example.txt").read_text()

    prompt = ChatPromptTemplate.from_template(template_text)

    assert "topic" in prompt.input_variables


def test_schemas_module_imports_cleanly():
    import newsresearch.llm.schemas  # noqa: F401
