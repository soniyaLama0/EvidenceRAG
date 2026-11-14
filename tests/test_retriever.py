"""Tests for ``retrieval.retriever``.

Fake embedding/store objects keep most tests fast and offline. One
integration test uses a real in-memory Chroma collection with fabricated
vectors, so no model download is required.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_retriever.py -v
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

DIMENSION = 384


def make_vector(position: int = 0) -> list[float]:
    """One-hot 384-dimensional vector for deterministic retrieval."""
    vector = [0.0] * DIMENSION
    vector[position] = 1.0
    return vector


def make_metadata(
    *,
    source: str = "research_paper.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_path": f"C:/data/{source}",
        "page_number": page_number,
        "total_pages": 6,
        "chunk_index": chunk_index,
        "char_start": 0,
        "char_end": 1000,
        "token_count": 200,
        "is_truncated": False,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def make_result(
    record_id: str = "chunk_1",
    text: str = "A retrieved evidence chunk.",
    metadata: dict[str, Any] | None = None,
    distance: float = 0.1,
) -> dict:
    return {
        "id": record_id,
        "text": text,
        "metadata": metadata if metadata is not None else make_metadata(),
        "distance": distance,
        "similarity": 1.0 - distance,
    }


def make_chunk(
    text: str,
    *,
    page_number: int = 1,
    chunk_index: int = 0,
    embedding: list[float] | None = None,
) -> dict:
    vector = embedding if embedding is not None else make_vector(chunk_index)
    return {
        "text": text,
        "metadata": make_metadata(
            page_number=page_number, chunk_index=chunk_index
        ),
        "embedding": vector,
    }


class FakeEmbeddingModel:
    """Records embed_texts calls and returns one fixed 384-d vector."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else make_vector(0)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vector], dtype=np.float32)


class FakeVectorStore:
    """Records search/get_count calls and returns canned results."""

    def __init__(self, count: int = 2, results: list[dict] | None = None) -> None:
        self.count = count
        self.results = results if results is not None else [make_result()]
        self.search_calls: list[dict[str, Any]] = []
        self.count_calls = 0

    def get_collection_count(self) -> int:
        self.count_calls += 1
        return self.count

    def search(
        self,
        query_embedding,
        n_results: int = 5,
        where=None,
    ):
        self.search_calls.append(
            {
                "query_embedding": query_embedding,
                "n_results": n_results,
                "where": where,
            }
        )
        return self.results


# ---------------------------------------------------------------------------
# Retrieval flow
# ---------------------------------------------------------------------------


def test_retrieve_returns_structured_results() -> None:
    expected = [make_result(), make_result("chunk_2", distance=0.3)]
    store = FakeVectorStore(results=expected)
    retriever = Retriever(FakeEmbeddingModel(), store)  # type: ignore[arg-type]

    results = retriever.retrieve("What does the paper propose?")

    assert results == expected
    assert set(results[0]) == {"id", "text", "metadata", "distance", "similarity"}


def test_query_is_embedded_once_and_passed_to_search() -> None:
    model = FakeEmbeddingModel(make_vector(7))
    store = FakeVectorStore()
    retriever = Retriever(model, store)  # type: ignore[arg-type]

    retriever.retrieve("What is Vyapar AI?")

    assert model.calls == [["What is Vyapar AI?"]]
    assert len(store.search_calls) == 1
    assert store.search_calls[0]["query_embedding"] == pytest.approx(make_vector(7))


def test_top_k_and_where_are_passed_through() -> None:
    model = FakeEmbeddingModel()
    store = FakeVectorStore()
    retriever = Retriever(model, store)  # type: ignore[arg-type]

    retriever.retrieve("default top k")
    assert store.search_calls[-1]["n_results"] == 5
    assert store.search_calls[-1]["where"] is None

    retriever.retrieve(
        "filtered query",
        top_k=3,
        where={"page_number": 2, "source": "research_paper.pdf"},
    )
    assert store.search_calls[-1]["n_results"] == 3
    assert store.search_calls[-1]["where"] == {
        "page_number": 2,
        "source": "research_paper.pdf",
    }


def test_empty_collection_returns_empty_without_embedding() -> None:
    model = FakeEmbeddingModel()
    store = FakeVectorStore(count=0)
    retriever = Retriever(model, store)  # type: ignore[arg-type]

    assert retriever.retrieve("anything") == []
    assert model.calls == []
    assert store.search_calls == []


def test_citation_metadata_is_preserved() -> None:
    metadata = make_metadata(source="study.pdf", page_number=4, chunk_index=9)
    result = make_result(metadata=metadata)
    retriever = Retriever(
        FakeEmbeddingModel(),  # type: ignore[arg-type]
        FakeVectorStore(results=[result]),  # type: ignore[arg-type]
    )

    retrieved = retriever.retrieve("cite this")[0]

    assert retrieved["metadata"] == metadata
    assert retrieved["metadata"]["source"] == "study.pdf"
    assert retrieved["metadata"]["page_number"] == 4
    assert retrieved["metadata"]["chunk_index"] == 9
    assert retrieved["metadata"]["char_start"] == 0
    assert retrieved["metadata"]["char_end"] == 1000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(query: str) -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)


@pytest.mark.parametrize("query", [123, None, ["query"], b"query"])
def test_non_string_query_is_rejected(query: object) -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="query"):
        retriever.retrieve(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "5", None])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("valid query", top_k=top_k)  # type: ignore[arg-type]


def test_invalid_where_is_rejected() -> None:
    retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="where"):
        retriever.retrieve("valid query", where=["page_number", 1])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="where"):
        retriever.retrieve("valid query", where={})


# ---------------------------------------------------------------------------
# Integration with the real VectorStore (in-memory Chroma, fake embeddings)
# ---------------------------------------------------------------------------


def make_store() -> VectorStore:
    client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    return VectorStore(
        client=client,
        collection_name=f"test_retriever_{uuid.uuid4().hex}",
    )


def test_retrieve_against_real_vector_store() -> None:
    store = make_store()
    store.add_chunks(
        [
            make_chunk(
                "Vyapar AI is a voice-first business assistant.",
                page_number=1,
                chunk_index=0,
                embedding=make_vector(0),
            ),
            make_chunk(
                "The evaluation uses a small business dataset.",
                page_number=2,
                chunk_index=1,
                embedding=make_vector(1),
            ),
        ]
    )
    retriever = Retriever(FakeEmbeddingModel(make_vector(0)), store)  # type: ignore[arg-type]

    results = retriever.retrieve("What is Vyapar AI?", top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "Vyapar AI is a voice-first business assistant."
    assert results[0]["metadata"]["page_number"] == 1
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_retrieve_with_metadata_filter_against_real_store() -> None:
    store = make_store()
    store.add_chunks(
        [
            make_chunk("page one", page_number=1, chunk_index=0, embedding=make_vector(0)),
            make_chunk("page two", page_number=2, chunk_index=1, embedding=make_vector(1)),
        ]
    )
    retriever = Retriever(FakeEmbeddingModel(make_vector(0)), store)  # type: ignore[arg-type]

    results = retriever.retrieve(
        "filtered query",
        top_k=5,
        where={"page_number": 2},
    )

    assert len(results) == 1
    assert results[0]["text"] == "page two"
    assert results[0]["metadata"]["page_number"] == 2


def test_empty_real_store_returns_empty() -> None:
    store = make_store()
    model = FakeEmbeddingModel()
    retriever = Retriever(model, store)  # type: ignore[arg-type]

    assert retriever.retrieve("nothing indexed yet") == []
    assert model.calls == []


def test_where_filter_selects_body_or_references_section() -> None:
    """Stage 6D: ``where={"section": ...}`` is enough to separate sections."""
    store = make_store()
    body = make_chunk("Body evidence.", page_number=1, chunk_index=0, embedding=make_vector(0))
    body["metadata"]["section"] = "body"
    reference = make_chunk(
        "A reference entry.", page_number=2, chunk_index=1, embedding=make_vector(1)
    )
    reference["metadata"]["section"] = "references"
    store.add_chunks([body, reference])
    retriever = Retriever(FakeEmbeddingModel(make_vector(1)), store)  # type: ignore[arg-type]

    body_hits = retriever.retrieve("query", top_k=5, where={"section": "body"})
    reference_hits = retriever.retrieve(
        "query", top_k=5, where={"section": "references"}
    )

    assert [result["text"] for result in body_hits] == ["Body evidence."]
    assert [result["text"] for result in reference_hits] == ["A reference entry."]


def test_two_retrievers_over_different_stores_are_independent() -> None:
    """Collection selection stays in the injected store, not in Retriever."""
    first = make_store()
    second = make_store()
    first.add_chunks(
        [make_chunk("from the first collection", chunk_index=0, embedding=make_vector(0))]
    )
    second.add_chunks(
        [make_chunk("from the second collection", chunk_index=0, embedding=make_vector(0))]
    )

    first_hits = Retriever(FakeEmbeddingModel(), first).retrieve("q", top_k=1)  # type: ignore[arg-type]
    second_hits = Retriever(FakeEmbeddingModel(), second).retrieve("q", top_k=1)  # type: ignore[arg-type]

    assert first_hits[0]["text"] == "from the first collection"
    assert second_hits[0]["text"] == "from the second collection"
