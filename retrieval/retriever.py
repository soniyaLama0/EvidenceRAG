"""Query-time retrieval for EvidenceRAG.

``Retriever`` connects the existing ``EmbeddingModel`` and ``VectorStore``:
it embeds one question with ``embed_texts()`` and searches the stored chunks
with the resulting vector. It does not generate answers and does not load a
second embedding model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ingestion.embedding_model import EmbeddingModel
from retrieval.vector_store import SearchResult, VectorStore

DEFAULT_TOP_K = 5


class Retriever:
    """Embed a query and return the most similar stored chunks.

    Args:
        embedding_model: Existing ``EmbeddingModel`` instance.
        vector_store: Existing ``VectorStore`` instance.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the ``top_k`` chunks most similar to ``query``.

        Args:
            query: Non-empty natural-language question.
            top_k: Positive number of chunks to return.
            where: Optional Chroma metadata filter, e.g.
                ``{"source": "research_paper.pdf"}``. Collections written by the
                cleaned ingestion pipeline also accept
                ``{"section": "body"}`` or ``{"section": "references"}``.

        Returns:
            ``SearchResult`` dictionaries with ``id``, ``text``, ``metadata``,
            ``distance``, and ``similarity``. Citation metadata (source, page
            number, chunk index, offsets, etc.) is passed through unchanged.

        Raises:
            TypeError: If ``query`` is not a string or ``where`` is not a
                mapping.
            ValueError: If ``query`` is empty, ``top_k`` is not a positive
                integer, or ``where`` is an empty mapping.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise ValueError("query must not be empty.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        if where is not None:
            if not isinstance(where, Mapping):
                raise TypeError("where must be a dictionary or None.")
            if not where:
                raise ValueError("where must not be empty when provided.")

        # Avoid embedding a query when there is nothing to search.
        if self.vector_store.get_collection_count() == 0:
            return []

        # embed_texts() returns a 2-D array; take the single query row and
        # convert it to plain floats so it matches VectorStore's input format.
        query_batch = self.embedding_model.embed_texts([query])
        query_embedding = [float(value) for value in query_batch[0]]

        return self.vector_store.search(
            query_embedding,
            n_results=top_k,
            where=where,
        )
