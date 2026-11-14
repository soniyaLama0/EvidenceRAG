"""Tests for ``ingestion.embedding_model``.

Most tests inject a small fake model, so they run instantly and need no
download. Two final tests use the real all-MiniLM-L6-v2 model on CPU.

Run from the project root with the project virtual environment::

    .venv\\Scripts\\python.exe -m pytest tests/test_embedding_model.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingestion.embedding_model import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EmbeddingModel,
    embed_chunks,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Pretend tokenizer: one token per whitespace-separated word plus 2 specials."""

    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": [[0] * (len(text.split()) + 2) for text in texts],
        }


class FakeModel:
    """Small stand-in for SentenceTransformer with a controllable shape."""

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        encode_dimension: int | None = None,
        max_seq_length: int = 256,
        value: float = 0.5,
    ) -> None:
        self.dimension = dimension
        self.encode_dimension = encode_dimension or dimension
        self.max_seq_length = max_seq_length
        self.value = value
        self.tokenizer = FakeTokenizer()
        self.encode_calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return np.full(
            (len(texts), self.encode_dimension),
            self.value,
            dtype=np.float32,
        )


def make_chunk(
    text: str,
    *,
    chunk_index: int = 0,
    page_number: int = 1,
    source: str = "paper.pdf",
) -> dict:
    """Build a chunk in the same shape TextChunker returns."""
    return {
        "text": text,
        "metadata": {
            "source": source,
            "source_path": f"C:/data/{source}",
            "page_number": page_number,
            "total_pages": 2,
            "chunk_index": chunk_index,
            "char_start": 0,
            "char_end": len(text),
        },
    }


# ---------------------------------------------------------------------------
# Empty input and metadata preservation
# ---------------------------------------------------------------------------


def test_empty_input_is_safe() -> None:
    """No chunks means no output, and embed_texts returns a 0x384 array."""
    model = EmbeddingModel(model=FakeModel())

    assert model.embed_chunks([]) == []
    assert model.embed_texts([]).shape == (0, EMBEDDING_DIMENSION)


def test_preserves_text_and_metadata() -> None:
    """Original text and every metadata field survive embedding."""
    chips = [make_chunk("EvidenceRAG chunk text.", chunk_index=4, page_number=3)]
    model = EmbeddingModel(model=FakeModel())

    embedded = model.embed_chunks(chips)

    assert len(embedded) == 1
    assert embedded[0]["text"] == "EvidenceRAG chunk text."
    for key, value in chips[0]["metadata"].items():
        assert embedded[0]["metadata"][key] == value
    assert embedded[0]["metadata"]["embedding_model"] == EMBEDDING_MODEL_NAME
    assert len(embedded[0]["embedding"]) == EMBEDDING_DIMENSION


def test_batch_size_is_passed_to_the_model() -> None:
    """The configured batch size reaches encode, and order is preserved."""
    fake = FakeModel()
    chunks = [make_chunk(f"chunk number {i}") for i in range(5)]

    embedded = EmbeddingModel(model=fake, batch_size=2).embed_chunks(chunks)

    assert len(embedded) == 5
    assert len(fake.encode_calls) == 1
    assert fake.encode_calls[0]["batch_size"] == 2
    assert fake.encode_calls[0]["texts"] == [chunk["text"] for chunk in chunks]
    assert [chunk["text"] for chunk in embedded] == [
        chunk["text"] for chunk in chunks
    ]


# ---------------------------------------------------------------------------
# Truncation handling
# ---------------------------------------------------------------------------


def test_long_chunks_are_flagged_and_warned() -> None:
    """Truncation is reported in metadata and through a warning."""
    fake = FakeModel(max_seq_length=8)
    long_text = " ".join(f"word{i}" for i in range(20))  # 20 + 2 specials = 22
    short_text = "two words"  # 2 + 2 specials = 4

    chunks = [make_chunk(long_text, chunk_index=0), make_chunk(short_text, chunk_index=1)]

    with pytest.warns(UserWarning, match="token limit"):
        embedded = EmbeddingModel(model=fake).embed_chunks(chunks)

    assert embedded[0]["metadata"]["token_count"] == 22
    assert embedded[0]["metadata"]["is_truncated"] is True
    assert embedded[1]["metadata"]["token_count"] == 4
    assert embedded[1]["metadata"]["is_truncated"] is False

    # The text itself is stored in full; only the embedding is affected.
    assert embedded[0]["text"] == long_text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, "16"])
def test_invalid_batch_size_raises(batch_size: object) -> None:
    with pytest.raises(ValueError):
        EmbeddingModel(model=FakeModel(), batch_size=batch_size)  # type: ignore[arg-type]


def test_wrong_model_dimension_raises() -> None:
    """A model that is not 384-dimensional is rejected up front."""
    with pytest.raises(ValueError, match="384"):
        EmbeddingModel(model=FakeModel(dimension=768))


def test_wrong_output_shape_raises() -> None:
    """Output must be exactly (number_of_texts, 384)."""
    fake = FakeModel(encode_dimension=10)
    with pytest.raises(ValueError, match="shape"):
        EmbeddingModel(model=fake).embed_texts(["hello"])


def test_non_finite_output_raises() -> None:
    fake = FakeModel(value=math.nan)
    with pytest.raises(ValueError, match="NaN"):
        EmbeddingModel(model=fake).embed_texts(["hello"])


def test_plain_strings_are_rejected() -> None:
    model = EmbeddingModel(model=FakeModel())
    with pytest.raises(TypeError):
        model.embed_texts("one string")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model.embed_chunks("not a chunk list")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("chunk", "error"),
    [
        ("not a dict", TypeError),
        ({"metadata": {}}, ValueError),  # missing "text"
        ({"text": 123, "metadata": {}}, TypeError),
        ({"text": "   ", "metadata": {}}, ValueError),  # empty text
        ({"text": "hello"}, TypeError),  # missing metadata
        ({"text": "hello", "metadata": "nope"}, TypeError),
    ],
)
def test_invalid_chunks_raise(chunk: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        EmbeddingModel(model=FakeModel()).embed_chunks([chunk])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Real model (CPU, cached locally)
# ---------------------------------------------------------------------------


def test_real_model_returns_384_dimensions() -> None:
    """The real model loads on CPU and returns 384 finite floats per chunk."""
    model = EmbeddingModel()

    assert model.embedding_dimension == EMBEDDING_DIMENSION
    assert model.max_seq_length == 256

    chunks = [
        make_chunk("EvidenceRAG retrieves evidence from research papers."),
        make_chunk("Chunking text improves retrieval quality.", chunk_index=1),
    ]
    embedded = model.embed_chunks(chunks)

    for item, original in zip(embedded, chunks):
        assert item["text"] == original["text"]
        assert len(item["embedding"]) == 384
        assert all(math.isfinite(value) for value in item["embedding"])

    # Different texts should not produce identical vectors.
    assert embedded[0]["embedding"] != embedded[1]["embedding"]

    # A second wrapper reuses the cached model instead of loading it again.
    assert EmbeddingModel().model is model.model


def test_real_model_truncates_at_256_tokens() -> None:
    """A 300-token input is embedded from its first 256 tokens and flagged."""
    model = EmbeddingModel()
    long_text = " ".join(["evidence"] * 300)

    with pytest.warns(UserWarning, match="256-token limit"):
        embedded = embed_chunks([make_chunk(long_text)])

    assert embedded[0]["metadata"]["token_count"] > 256
    assert embedded[0]["metadata"]["is_truncated"] is True
    assert embedded[0]["text"] == long_text
    assert len(embedded[0]["embedding"]) == 384
