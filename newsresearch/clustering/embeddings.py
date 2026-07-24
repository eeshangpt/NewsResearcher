"""Thin `numpy` convenience wrapper over `llm/models.py::get_embeddings()`.

Clustering code (`clustering/cluster.py`) works on arrays, not LangChain
`Embeddings` objects directly -- this module is the single seam between the
two, so clustering never constructs an embeddings backend itself.
"""

import numpy as np

from newsresearch.llm.models import get_embeddings


def embed(texts: list[str]) -> np.ndarray:
    """Embed `texts` via the `Settings.embeddings.backend`-selected implementation.

    Returns an array of shape `(len(texts), embedding_dim)`.
    """
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(texts)
    return np.asarray(vectors)
