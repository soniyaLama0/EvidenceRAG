"""Tests for ``retrieval.vector_store``.

These tests use an in-memory ChromaDB client, so they do not touch disk or
the embedding model. One test exercises a real PersistentClient directory.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_vector_store.py -v
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from retrieval.vector_store import VectorStore

DIMENSION = 384


def make_vector(position: int = 0) -> list[float]:
    """One-hot 384-dimensional vector, useful for deterministic ranking."""
    vector = [0.0] * DIMENSION
    vector[position] = 1.0
    return vector


def make_chunk(
    text: str = "EvidenceRAG chunk text.",
    *,
    source: str = "paper.pdf",
    source_path: str | None = None,
    page_number: int = 1,
    total_pages: int = 3,
    chunk_index: int = 0,
    char_start: int = 0,
    char_end: int = 1000,
    token_count: int = 200,
    is_truncated: bool = False,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding: list[float] | None = None,
) -> dict:
    """Build the exact dictionary shape returned by embed_chunks()."""
    return {
        "text": text,
        "metadata": {
            "source": source,
            "source_path": source_path if source_path is not None else f"C:/data/{source}",
            "page_number": page_number,
            "total_pages": total_pages,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": token_count,
            "is_truncated": is_truncated,
            "embedding_model": embedding_model,
        },
        "embedding": embedding if embedding is not None else make_vector(0),
    }


@pytest.fixture()
def store() -> VectorStore:
    """Fresh collection for each test.

    ``EphemeralClient`` instances share one in-process Chroma system, so a
    unique collection name is required to keep tests isolated.
    """
    client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    return VectorStore(
        client=client,
        collection_name=f"test_chunks_{uuid.uuid4().hex}",
    )


# ---------------------------------------------------------------------------
# Basic storage
# ---------------------------------------------------------------------------


def test_empty_collection_count(store: VectorStore) -> None:
    assert store.get_collection_count() == 0


def test_empty_input_is_safe(store: VectorStore) -> None:
    assert store.add_chunks([]) == 0
    assert store.get_collection_count() == 0
    assert store.search(make_vector()) == []


def test_add_chunks_increments_count(store: VectorStore) -> None:
    chunks = [
        make_chunk("first", chunk_index=0, embedding=make_vector(0)),
        make_chunk("second", chunk_index=1, embedding=make_vector(1)),
        make_chunk("third", chunk_index=2, embedding=make_vector(2)),
    ]

    assert store.add_chunks(chunks) == 3
    assert store.get_collection_count() == 3


def test_text_and_metadata_round_trip(store: VectorStore) -> None:
    chunk = make_chunk(
        "Round-trip text with exact metadata.",
        source="study.pdf",
        source_path="C:/research/study.pdf",
        page_number=4,
        total_pages=12,
        chunk_index=7,
        char_start=2400,
        char_end=3400,
        token_count=241,
        is_truncated=True,
    )
    store.add_chunks([chunk])

    result = store.search(chunk["embedding"], n_results=1)[0]

    assert result["text"] == chunk["text"]
    assert result["metadata"] == chunk["metadata"]
    assert result["distance"] == pytest.approx(0.0, abs=1e-5)
    assert result["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_upsert_updates_existing_record(store: VectorStore) -> None:
    original = make_chunk("old text", chunk_index=0)
    updated = make_chunk("new text", chunk_index=0, token_count=222)
    updated["metadata"]["is_truncated"] = True

    store.add_chunks([original])
    store.add_chunks([updated])

    assert store.get_collection_count() == 1
    result = store.search(updated["embedding"], n_results=1)[0]
    assert result["text"] == "new text"
    assert result["metadata"]["token_count"] == 222
    assert result["metadata"]["is_truncated"] is True


def test_collection_uses_cosine_and_no_embedding_function(store: VectorStore) -> None:
    configuration = store.collection.configuration or {}
    assert configuration["hnsw"]["space"] == "cosine"
    assert configuration["embedding_function"] is None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_similarity_search_orders_by_distance(store: VectorStore) -> None:
    store.add_chunks(
        [
            make_chunk("closest", chunk_index=0, embedding=make_vector(0)),
            make_chunk("orthogonal", chunk_index=1, embedding=make_vector(1)),
            make_chunk("also far", chunk_index=2, embedding=make_vector(2)),
        ]
    )

    results = store.search(make_vector(0), n_results=2)

    assert len(results) == 2
    assert results[0]["text"] == "closest"
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)
    assert results[1]["distance"] > results[0]["distance"]
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-5)


def test_metadata_filtering(store: VectorStore) -> None:
    store.add_chunks(
        [
            make_chunk("paper one", source="one.pdf", page_number=1, chunk_index=0),
            make_chunk("paper two", source="two.pdf", page_number=2, chunk_index=1),
            make_chunk(
                "paper two page three",
                source="two.pdf",
                page_number=3,
                chunk_index=2,
            ),
        ]
    )

    results = store.search(make_vector(), n_results=10, where={"source": "two.pdf"})
    assert {result["metadata"]["source"] for result in results} == {"two.pdf"}
    assert len(results) == 2

    page_results = store.search(
        make_vector(), n_results=10, where={"page_number": 3}
    )
    assert len(page_results) == 1
    assert page_results[0]["text"] == "paper two page three"


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_delete_by_ids(store: VectorStore) -> None:
    store.add_chunks(
        [
            make_chunk("keep", chunk_index=0, embedding=make_vector(0)),
            make_chunk("remove", chunk_index=1, embedding=make_vector(1)),
        ]
    )
    record_id = store.search(make_vector(1), n_results=1)[0]["id"]

    assert store.delete(ids=[record_id]) == 1
    assert store.get_collection_count() == 1
    remaining = store.search(make_vector(0), n_results=5)
    assert [result["text"] for result in remaining] == ["keep"]


def test_delete_by_metadata(store: VectorStore) -> None:
    store.add_chunks(
        [
            make_chunk("one-a", source="one.pdf", chunk_index=0),
            make_chunk("one-b", source="one.pdf", chunk_index=1),
            make_chunk("two-a", source="two.pdf", chunk_index=2),
        ]
    )

    assert store.delete(where={"source": "one.pdf"}) == 2
    assert store.get_collection_count() == 1
    assert store.search(make_vector(), n_results=5)[0]["text"] == "two-a"


def test_delete_requires_exactly_one_selector(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.delete()
    with pytest.raises(ValueError):
        store.delete(ids=["x"], where={"source": "a.pdf"})
    assert store.delete(ids=[]) == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [383, 385, 0])
def test_invalid_embedding_dimension_for_add(store: VectorStore, size: int) -> None:
    chunk = make_chunk(embedding=[0.0] * size)
    with pytest.raises(ValueError, match="384"):
        store.add_chunks([chunk])


def test_non_finite_embeddings_are_rejected(store: VectorStore) -> None:
    nan_chunk = make_chunk(embedding=[0.0] * (DIMENSION - 1) + [math.nan])
    inf_chunk = make_chunk("inf", chunk_index=1, embedding=[math.inf] + [0.0] * (DIMENSION - 1))

    with pytest.raises(ValueError, match="finite"):
        store.add_chunks([nan_chunk])
    with pytest.raises(ValueError, match="finite"):
        store.add_chunks([inf_chunk])


def test_invalid_search_embedding(store: VectorStore) -> None:
    store.add_chunks([make_chunk()])

    with pytest.raises(ValueError, match="384"):
        store.search([0.0] * 10)
    with pytest.raises(ValueError, match="finite"):
        store.search([math.nan] + [0.0] * (DIMENSION - 1))


@pytest.mark.parametrize("n_results", [0, -1, 1.5, True])
def test_invalid_n_results(store: VectorStore, n_results: object) -> None:
    store.add_chunks([make_chunk()])
    with pytest.raises(ValueError, match="n_results"):
        store.search(make_vector(), n_results=n_results)  # type: ignore[arg-type]


def test_missing_metadata_fields_are_rejected(store: VectorStore) -> None:
    chunk = make_chunk()
    del chunk["metadata"]["source"]

    with pytest.raises(ValueError, match="source"):
        store.add_chunks([chunk])


def test_invalid_metadata_types_are_rejected(store: VectorStore) -> None:
    chunk = make_chunk()
    chunk["metadata"]["page_number"] = "1"

    with pytest.raises(ValueError, match="page_number"):
        store.add_chunks([chunk])

    bad_bool = make_chunk(chunk_index=1)
    bad_bool["metadata"]["is_truncated"] = "no"
    with pytest.raises(ValueError, match="is_truncated"):
        store.add_chunks([bad_bool])


def test_invalid_text_is_rejected(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="empty"):
        store.add_chunks([make_chunk("   ")])


def test_duplicate_ids_in_one_batch_are_rejected(store: VectorStore) -> None:
    first = make_chunk("first text", chunk_index=0)
    second = make_chunk("second text", chunk_index=0)  # same source/page/index

    with pytest.raises(ValueError, match="Duplicate record ID"):
        store.add_chunks([first, second])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persistent_storage_survives_reopening(tmp_path: Path) -> None:
    database_path = tmp_path / "chroma_db"

    first_store = VectorStore(persist_directory=database_path)
    chunk = make_chunk(
        "Persistent chunk text.",
        source="persisted.pdf",
        source_path=str(tmp_path / "persisted.pdf"),
        chunk_index=0,
    )
    assert first_store.add_chunks([chunk]) == 1
    assert first_store.get_collection_count() == 1

    # Open a brand-new client/store on the same directory.
    second_store = VectorStore(persist_directory=database_path)
    assert second_store.get_collection_count() == 1

    result = second_store.search(chunk["embedding"], n_results=1)[0]
    assert result["text"] == "Persistent chunk text."
    assert result["metadata"] == chunk["metadata"]
