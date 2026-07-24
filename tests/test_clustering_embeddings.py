import numpy as np

from newsresearch.clustering.embeddings import embed


class _FakeEmbeddings:
    """Stand-in for a LangChain `Embeddings` implementation.

    Records the texts it was called with so the test can assert `embed()`
    delegates to `embed_documents` rather than reimplementing embedding
    logic itself.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(i)] * self.dim for i in range(len(texts))]


def test_embed_delegates_to_get_embeddings(monkeypatch):
    fake = _FakeEmbeddings(dim=4)
    monkeypatch.setattr(
        "newsresearch.clustering.embeddings.get_embeddings", lambda: fake
    )

    texts = ["first article", "second article", "third article"]
    result = embed(texts)

    assert fake.calls == [texts]
    assert isinstance(result, np.ndarray)


def test_embed_returns_correctly_shaped_array(monkeypatch):
    fake = _FakeEmbeddings(dim=8)
    monkeypatch.setattr(
        "newsresearch.clustering.embeddings.get_embeddings", lambda: fake
    )

    texts = ["a", "b", "c", "d", "e"]
    result = embed(texts)

    assert result.shape == (len(texts), fake.dim)


def test_embed_empty_list_returns_empty_array(monkeypatch):
    fake = _FakeEmbeddings(dim=4)
    monkeypatch.setattr(
        "newsresearch.clustering.embeddings.get_embeddings", lambda: fake
    )

    result = embed([])

    assert result.shape[0] == 0
