"""Local ChromaDB vector store for EvidenceRAG embedded chunks.

``VectorStore`` persists the exact output of
``EmbeddingModel.embed_chunks()`` and searches it with pre-computed query
embeddings. It never imports or runs the embedding model itself.

Storage layout:
    * Client: ``chromadb.PersistentClient`` at ``data/chroma`` (default)
    * Collection: ``evidence_chunks``
    * Distance: cosine (``similarity = 1 - distance``)
    * Embedding function: ``None`` because vectors are supplied by the caller
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from ingestion.embedding_model import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EmbeddedChunk,
)

DEFAULT_PERSIST_DIRECTORY = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "evidence_chunks"
#: Stage 6C collection built from cleaned pages with ``section`` metadata.
CLEANED_COLLECTION_NAME = "evidence_chunks_v2"
DEFAULT_N_RESULTS = 5
WRITE_BATCH_SIZE = 500

# Every chunk produced by EmbeddingModel must carry these fields.
REQUIRED_METADATA_FIELDS = (
    "source",
    "source_path",
    "page_number",
    "total_pages",
    "chunk_index",
    "char_start",
    "char_end",
    "token_count",
    "is_truncated",
    "embedding_model",
)


class SearchResult(TypedDict):
    """One nearest-neighbour result returned by ``search()``."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float
    similarity: float


def _is_int(value: object) -> bool:
    """True for integers, but not for booleans (bool subclasses int)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_text(text: object, label: str) -> str:
    """Return non-empty text or raise a clear error."""
    if not isinstance(text, str):
        raise TypeError(f"{label} must be a string, got {type(text).__name__}.")
    if not text.strip():
        raise ValueError(f"{label} is empty.")
    return text


def _validate_embedding(embedding: object, label: str) -> list[float]:
    """Return a 384-dimensional list of finite floats or raise."""
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise TypeError(f"{label} must be a sequence of numbers.")

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{label} must contain exactly {EMBEDDING_DIMENSION} values, "
            f"got {len(embedding)}."
        )

    values: list[float] = []
    for index, value in enumerate(embedding):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{label}[{index}] must be a number, got {type(value).__name__}."
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] is not finite ({number!r}).")
        values.append(number)

    return values


def _validate_metadata(metadata: object, label: str) -> dict[str, Any]:
    """Validate and copy the 10 required chunk metadata fields."""
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{label} must be a dictionary.")

    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}.")

    text_fields = ("source", "source_path", "embedding_model")
    for field in text_fields:
        value = metadata[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}[{field!r}] must be a non-empty string.")

    int_fields = (
        "page_number",
        "total_pages",
        "chunk_index",
        "char_start",
        "char_end",
        "token_count",
    )
    for field in int_fields:
        value = metadata[field]
        if not _is_int(value):
            raise ValueError(f"{label}[{field!r}] must be an integer.")

    if metadata["page_number"] < 1:
        raise ValueError(f"{label}['page_number'] must be >= 1.")
    if metadata["total_pages"] < 1:
        raise ValueError(f"{label}['total_pages'] must be >= 1.")
    if metadata["chunk_index"] < 0:
        raise ValueError(f"{label}['chunk_index'] must be >= 0.")
    if metadata["char_start"] < 0:
        raise ValueError(f"{label}['char_start'] must be >= 0.")
    if metadata["char_end"] < metadata["char_start"]:
        raise ValueError(f"{label}['char_end'] must be >= 'char_start'.")
    if metadata["token_count"] < 0:
        raise ValueError(f"{label}['token_count'] must be >= 0.")

    if not isinstance(metadata["is_truncated"], bool):
        raise ValueError(f"{label}['is_truncated'] must be a boolean.")

    # Chroma metadata accepts only scalar values. Extra keys are preserved,
    # but unsupported values (None, dicts, lists, objects) are rejected.
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} contains an invalid metadata key: {key!r}.")
        if isinstance(value, bool):
            continue
        if isinstance(value, (str, int)):
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{label}[{key!r}] is not finite.")
            continue
        raise ValueError(
            f"{label}[{key!r}] has unsupported type {type(value).__name__}; "
            "Chroma metadata must be a string, integer, float, or boolean."
        )

    return dict(metadata)


def _record_id(metadata: Mapping[str, Any]) -> str:
    """Build a deterministic ID from resolved path, page, and chunk index."""
    resolved_path = Path(str(metadata["source_path"])).expanduser().resolve()
    # normcase keeps IDs stable for case-insensitive Windows paths.
    key = "|".join(
        (
            os.path.normcase(str(resolved_path)),
            str(metadata["page_number"]),
            str(metadata["chunk_index"]),
        )
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"chunk_{digest}"


class VectorStore:
    """Store and search embedded chunks in a local ChromaDB collection.

    Args:
        persist_directory: Directory for Chroma's persistent files.
            Ignored when ``client`` is supplied.
        collection_name: Chroma collection name.
        client: Optional Chroma client, e.g. ``chromadb.EphemeralClient()``
            in tests. When omitted, a ``PersistentClient`` is created with
            telemetry disabled.

    Raises:
        ValueError: If the collection exists with incompatible settings.
    """

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: ClientAPI | None = None,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client: ClientAPI = (
            client
            if client is not None
            else chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={
                "description": "EvidenceRAG research-paper chunks",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dimension": EMBEDDING_DIMENSION,
            },
        )

        self._validate_collection()

    def _validate_collection(self) -> None:
        """Reject an existing collection with incompatible configuration."""
        configuration = self.collection.configuration or {}
        hnsw = configuration.get("hnsw") or {}
        space = hnsw.get("space")
        if space is not None and space != "cosine":
            raise ValueError(
                f"Collection {self.collection_name!r} uses {space!r} distance; "
                "VectorStore requires 'cosine'."
            )

        metadata = self.collection.metadata or {}
        dimension = metadata.get("embedding_dimension")
        if dimension is not None and dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Collection {self.collection_name!r} stores "
                f"{dimension}-dimensional embeddings; expected {EMBEDDING_DIMENSION}."
            )
        model = metadata.get("embedding_model")
        if model is not None and model != EMBEDDING_MODEL_NAME:
            raise ValueError(
                f"Collection {self.collection_name!r} uses {model!r}; "
                f"expected {EMBEDDING_MODEL_NAME!r}."
            )

    @classmethod
    def for_cleaned_ingestion(
        cls,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        client: ClientAPI | None = None,
    ) -> "VectorStore":
        """Open ``evidence_chunks_v2`` for the cleaned ingestion pipeline.

        This is a thin convenience wrapper: it uses the same constructor,
        validation and API as the baseline store, only with a different
        collection name. The original ``evidence_chunks`` collection is never
        touched.
        """
        return cls(
            persist_directory=persist_directory,
            collection_name=CLEANED_COLLECTION_NAME,
            client=client,
        )

    def add_chunks(self, chunks: Sequence[EmbeddedChunk]) -> int:
        """Upsert embedded chunks and return how many records were submitted.

        Deterministic IDs make this safe to call again for the same source:
        existing records are updated instead of duplicated. Empty input
        returns ``0`` without touching ChromaDB.
        """
        if isinstance(chunks, (str, bytes, Mapping)):
            raise TypeError("chunks must be a sequence of embedded chunk dictionaries.")

        chunk_list = list(chunks)
        if not chunk_list:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []
        seen_ids: set[str] = set()

        for index, chunk in enumerate(chunk_list):
            if not isinstance(chunk, Mapping):
                raise TypeError(
                    f"chunks[{index}] must be a dictionary, "
                    f"got {type(chunk).__name__}."
                )
            if "text" not in chunk:
                raise ValueError(f"chunks[{index}] is missing the 'text' key.")
            if "metadata" not in chunk:
                raise ValueError(f"chunks[{index}] is missing the 'metadata' key.")
            if "embedding" not in chunk:
                raise ValueError(f"chunks[{index}] is missing the 'embedding' key.")

            text = _validate_text(chunk["text"], f"chunks[{index}]['text']")
            metadata = _validate_metadata(
                chunk["metadata"], f"chunks[{index}]['metadata']"
            )
            embedding = _validate_embedding(
                chunk["embedding"], f"chunks[{index}]['embedding']"
            )
            record_id = _record_id(metadata)

            if record_id in seen_ids:
                raise ValueError(
                    f"Duplicate record ID {record_id!r} in this batch "
                    f"(chunks[{index}] repeats source/page/chunk_index)."
                )

            seen_ids.add(record_id)
            ids.append(record_id)
            documents.append(text)
            metadatas.append(metadata)
            embeddings.append(embedding)

        for start in range(0, len(ids), WRITE_BATCH_SIZE):
            end = start + WRITE_BATCH_SIZE
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int = DEFAULT_N_RESULTS,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return nearest chunks for a pre-computed query embedding.

        Args:
            query_embedding: 384-dimensional query vector.
            n_results: Maximum number of results, must be positive.
            where: Optional Chroma metadata filter, e.g.
                ``{"source": "paper.pdf"}``.

        Returns:
            Results ordered by increasing cosine distance, each containing
            ``id``, ``text``, ``metadata``, ``distance``, and
            ``similarity = 1 - distance``.
        """
        embedding = _validate_embedding(query_embedding, "query_embedding")

        if not _is_int(n_results) or n_results <= 0:
            raise ValueError(f"n_results must be a positive integer, got {n_results!r}.")

        if where is not None:
            if not isinstance(where, Mapping):
                raise TypeError("where must be a dictionary or None.")
            if not where:
                raise ValueError("where must not be empty when provided.")

        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
            where=dict(where) if where is not None else None,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns column-major results grouped per query; we sent one.
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        search_results: list[SearchResult] = []
        for record_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            distance_value = float(distance)
            search_results.append(
                {
                    "id": record_id,
                    "text": text or "",
                    "metadata": dict(metadata or {}),
                    "distance": distance_value,
                    "similarity": 1.0 - distance_value,
                }
            )

        return search_results

    def delete(
        self,
        ids: Sequence[str] | None = None,
        where: Mapping[str, Any] | None = None,
    ) -> int:
        """Delete records by IDs or by metadata filter; returns the count.

        Exactly one of ``ids`` or ``where`` must be provided.
        """
        if (ids is None) == (where is None):
            raise ValueError("delete() requires exactly one of 'ids' or 'where'.")

        before = self.collection.count()
        if ids is not None:
            if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
                raise TypeError("ids must be a sequence of strings.")
            id_list = list(ids)
            if not id_list:
                return 0
            for index, record_id in enumerate(id_list):
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError(f"ids[{index}] must be a non-empty string.")
            result = self.collection.delete(ids=id_list)
        else:
            if not isinstance(where, Mapping):
                raise TypeError("where must be a dictionary.")
            if not where:
                raise ValueError("where must not be empty when provided.")
            result = self.collection.delete(where=dict(where))

        # Chroma 1.5.x returns {"deleted": n}; fall back to a count diff.
        if isinstance(result, Mapping) and _is_int(result.get("deleted")):
            return int(result["deleted"])
        return int(before - self.collection.count())

    def get_collection_count(self) -> int:
        """Return the number of records stored in the collection."""
        return int(self.collection.count())
