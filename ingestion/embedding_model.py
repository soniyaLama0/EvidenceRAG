"""Embed TextChunker chunks with sentence-transformers/all-MiniLM-L6-v2.

The model is loaded once per process (see ``_load_model``) and runs on CPU.
Input is the list of chunks produced by ``TextChunker.chunk_documents()``::

    {"text": "...", "metadata": {...}}

Output is the same chunk plus a 384-dimensional embedding::

    {
        "text": "...",
        "metadata": {
            ...original chunk metadata...,
            "token_count": 212,
            "is_truncated": False,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "embedding": [0.0123, -0.0456, ...],  # 384 floats
    }

Truncation behaviour
--------------------
all-MiniLM-L6-v2 accepts at most **256 word-piece tokens** per input
(``model.max_seq_length``), including the special ``[CLS]`` and ``[SEP]``
tokens. Its tokenizer truncates on the right, so only the first 256 tokens
of a longer text are seen by the model. This module never hides that:
it counts tokens with truncation disabled, stores ``token_count`` and
``is_truncated`` on every chunk, and emits a warning when any chunk is too
long. The original ``text`` is always kept in full; only the embedding is
affected.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from .text_chunker import Chunk, ChunkMetadata

if TYPE_CHECKING:  # imported only for type hints; keeps unit tests fast
    from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32


class EmbeddingMetadata(ChunkMetadata):
    """Chunk metadata plus information about how the embedding was made."""

    token_count: int  # tokens before truncation, including [CLS] and [SEP]
    is_truncated: bool  # True if the model saw only the first max_seq_length tokens
    embedding_model: str  # which model produced the vector


class EmbeddedChunk(TypedDict):
    """One chunk with its embedding; text and page metadata are unchanged."""

    text: str
    metadata: EmbeddingMetadata
    embedding: list[float]


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> "SentenceTransformer":
    """Load and cache the sentence-transformers model on CPU.

    ``lru_cache`` means the model is created once and reused by every later
    ``EmbeddingModel()`` with the same name. Importing torch is deferred to
    this function so tests can inject a fake model.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _model_dimension(model: Any) -> int:
    """Read the embedding size, supporting old and new sentence-transformers."""
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:  # sentence-transformers < 5 used a longer name
        getter = model.get_sentence_embedding_dimension
    return int(getter())


class EmbeddingModel:
    """Turn chunks into embeddings with all-MiniLM-L6-v2.

    Args:
        model_name: Hugging Face model ID. Defaults to all-MiniLM-L6-v2.
        batch_size: Number of texts handed to the model per batch.
        model: Optional already-loaded model. Mainly useful for tests;
            leave as ``None`` to load the real CPU model once.

    Raises:
        ValueError: For an invalid ``batch_size`` or a model whose embedding
            size is not 384.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: "SentenceTransformer | None" = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model if model is not None else _load_model(model_name)

        # The token limit is what the model will actually truncate at.
        self.max_seq_length = int(self.model.max_seq_length)
        self.embedding_dimension = _model_dimension(self.model)

        if self.embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"{model_name!r} produces {self.embedding_dimension}-dimensional "
                f"embeddings, but this project expects {EMBEDDING_DIMENSION}."
            )

    def _count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Count tokens per text *without* truncation."""
        encoded = self.model.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        return [len(input_ids) for input_ids in encoded["input_ids"]]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed raw strings and return a ``(len(texts), 384)`` float array.

        The underlying ``SentenceTransformer.encode`` splits the list into
        batches of ``self.batch_size`` internally.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string.")

        text_list = list(texts)
        for index, text in enumerate(text_list):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string, got {type(text).__name__}.")
            if not text.strip():
                raise ValueError(f"texts[{index}] is empty.")

        # Return an empty array with the correct shape rather than crashing.
        if not text_list:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        raw_embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        expected_shape = (len(text_list), self.embedding_dimension)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"Model returned embeddings with shape {embeddings.shape}, "
                f"expected {expected_shape}."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Model returned embeddings containing NaN or infinity.")

        return embeddings

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks while preserving their text and metadata exactly."""
        if isinstance(chunks, (str, bytes, Mapping)):
            raise TypeError("chunks must be a sequence of chunk dictionaries.")

        chunk_list = list(chunks)
        if not chunk_list:
            return []  # empty input is valid and produces no output

        texts: list[str] = []
        for index, chunk in enumerate(chunk_list):
            if not isinstance(chunk, Mapping):
                raise TypeError(
                    f"chunks[{index}] must be a dictionary, got {type(chunk).__name__}."
                )
            if "text" not in chunk:
                raise ValueError(f"chunks[{index}] is missing the 'text' key.")
            if not isinstance(chunk["text"], str):
                raise TypeError(f"chunks[{index}]['text'] must be a string.")
            if not chunk["text"].strip():
                raise ValueError(f"chunks[{index}]['text'] is empty.")
            if not isinstance(chunk.get("metadata"), Mapping):
                raise TypeError(f"chunks[{index}]['metadata'] must be a dictionary.")

            texts.append(chunk["text"])

        # Measure real token counts before encoding, so truncation is visible.
        token_counts = self._count_tokens(texts)
        truncation_flags = [count > self.max_seq_length for count in token_counts]
        truncated_count = sum(truncation_flags)

        if truncated_count:
            warnings.warn(
                f"{truncated_count} of {len(texts)} chunks exceed the model's "
                f"{self.max_seq_length}-token limit. They are embedded from "
                f"their first {self.max_seq_length} tokens only; their full "
                "text is preserved in 'text'. See metadata['is_truncated'].",
                UserWarning,
                stacklevel=2,
            )

        embeddings = self.embed_texts(texts)

        embedded_chunks: list[EmbeddedChunk] = []
        for index, chunk in enumerate(chunk_list):
            metadata: dict[str, Any] = dict(chunk["metadata"])
            metadata.update(
                {
                    "token_count": token_counts[index],
                    "is_truncated": truncation_flags[index],
                    "embedding_model": self.model_name,
                }
            )
            embedded_chunks.append(
                {
                    "text": chunk["text"],  # full original text, never shortened
                    "metadata": metadata,  # type: ignore[typeddict-item]
                    "embedding": embeddings[index].tolist(),
                }
            )

        return embedded_chunks


def embed_chunks(
    chunks: Sequence[Chunk],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Shortcut for ``EmbeddingModel(batch_size=batch_size).embed_chunks(chunks)``."""
    return EmbeddingModel(batch_size=batch_size).embed_chunks(chunks)
