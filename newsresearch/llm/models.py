"""LangChain chat-model and embeddings factories (NFR-6).

Every LLM call in the pipeline goes through `get_chat_model(stage)` and every
embedding call through `get_embeddings()` rather than constructing
`ChatOpenAI`/`HuggingFaceEmbeddings`/etc. directly elsewhere. Porting to a
different vendor later means changing these two factories, nothing else.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from newsresearch.config import Settings

# Local embeddings default per Cross-Cutting Concerns: a small, fast
# sentence-transformers model, chosen over the OpenAI embeddings API for cost.
_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_chat_model(stage: str) -> BaseChatModel:
    """Return a `ChatOpenAI` configured with the model name for `stage`.

    `stage` must name one of the fields on `Settings.models` (e.g.
    "subtopic", "claim_extraction", "summarization", "bias_framing",
    "briefing").
    """
    settings = Settings()
    if not hasattr(settings.models, stage):
        raise ValueError(f"Unknown model stage: {stage!r}")
    model_name = getattr(settings.models, stage)
    return ChatOpenAI(model=model_name, api_key=settings.openai_api_key)


def get_embeddings() -> Embeddings:
    """Return the configured embeddings backend.

    `Settings.embeddings.backend == "local"` (default) returns
    `HuggingFaceEmbeddings` wrapping a local sentence-transformers model;
    `"openai"` returns `OpenAIEmbeddings`. Both satisfy the
    `langchain_core.embeddings.Embeddings` interface.
    """
    settings = Settings()
    if settings.embeddings.backend == "openai":
        return OpenAIEmbeddings(api_key=settings.openai_api_key)
    return HuggingFaceEmbeddings(model_name=_LOCAL_EMBEDDING_MODEL)
