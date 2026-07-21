import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from newsresearch.config import Settings
from newsresearch.llm.models import get_chat_model, get_embeddings

STAGE_NAMES = ("subtopic", "claim_extraction", "summarization", "bias_framing", "briefing")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Run against a scratch cwd with no ambient .env/config.yaml, plus a
    dummy API key so ChatOpenAI/OpenAIEmbeddings construct without a real
    network call or a missing-credentials error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EMBEDDINGS__BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    return tmp_path


@pytest.mark.parametrize("stage", STAGE_NAMES)
def test_get_chat_model_returns_base_chat_model_with_configured_name(stage):
    settings = Settings()
    expected_model_name = getattr(settings.models, stage)

    model = get_chat_model(stage)

    assert isinstance(model, BaseChatModel)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == expected_model_name


def test_get_chat_model_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown model stage"):
        get_chat_model("not_a_real_stage")


def test_get_embeddings_returns_huggingface_embeddings_for_local_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS__BACKEND", "local")

    embeddings = get_embeddings()

    assert isinstance(embeddings, Embeddings)
    assert isinstance(embeddings, HuggingFaceEmbeddings)


def test_get_embeddings_returns_openai_embeddings_for_openai_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS__BACKEND", "openai")

    embeddings = get_embeddings()

    assert isinstance(embeddings, Embeddings)
    assert isinstance(embeddings, OpenAIEmbeddings)
