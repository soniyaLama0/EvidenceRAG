"""Split page-level documents into overlapping chunks of text.

``PDFLoader`` returns one dictionary per PDF page::

    {"text": "...", "metadata": {"source": ..., "page_number": ...}}

``TextChunker`` turns that list into smaller, overlapping chunks that are
easier to embed and search later::

    {
        "text": "first 1000 characters ...",
        "metadata": {
            "source": "report.pdf",
            "source_path": "C:/data/report.pdf",
            "page_number": 1,
            "total_pages": 10,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 1000,
        },
    }

Chunking never crosses page boundaries, so every chunk can still be cited
back to one page.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from .pdf_loader import Document, DocumentMetadata


class ChunkMetadata(DocumentMetadata):
    """Page metadata plus chunk-specific fields.

    ``char_start`` and ``char_end`` are offsets into the *page* text that was
    used as input, and they are half-open (``char_end`` is excluded). This
    means ``page_text[char_start:char_end] == chunk["text"]``.
    """

    chunk_index: int  # 0-based and unique across one chunk_documents() call
    char_start: int
    char_end: int


class Chunk(TypedDict):
    """A piece of a page, ready for embedding later."""

    text: str
    metadata: ChunkMetadata


class TextChunker:
    """Split documents into fixed-size character windows with overlap.

    Args:
        chunk_size: Maximum number of characters per chunk. Defaults to 1000.
        overlap: Number of characters shared by neighbouring chunks.
            Defaults to 200.

    Raises:
        ValueError: If ``chunk_size`` is not positive, if ``overlap`` is
            negative, or if ``overlap`` is not smaller than ``chunk_size``.
    """

    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_OVERLAP: int = 200

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        self.chunk_size = self._validate_chunk_size(chunk_size)
        self.overlap = self._validate_overlap(overlap)

    def _validate_chunk_size(self, chunk_size: int) -> int:
        """Return a valid chunk size or raise ValueError."""
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}."
            )
        return chunk_size

    def _validate_overlap(self, overlap: int) -> int:
        """Return a valid overlap or raise ValueError."""
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError(
                f"overlap must be a non-negative integer, got {overlap!r}."
            )

        # If overlap >= chunk_size the next window would start at or before
        # the current one and the loop would never finish.
        if overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise the window "
                "would not move forward."
            )
        return overlap

    def _split_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split one page's text into (chunk_text, start, end) tuples.

        Positions refer to ``text`` itself. Whitespace-only windows are
        dropped, and surrounding whitespace is removed from each chunk
        (with the offsets adjusted to match).
        """
        # Nothing to chunk on an empty or whitespace-only page.
        if not text.strip():
            return []

        step = self.chunk_size - self.overlap
        text_length = len(text)
        spans: list[tuple[str, int, int]] = []

        start = 0
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            window = text[start:end]

            chunk_text = window.strip()
            if chunk_text:
                # How many characters were trimmed from each side?
                leading = len(window) - len(window.lstrip())
                trailing = len(window) - len(window.rstrip())
                spans.append((chunk_text, start + leading, end - trailing))

            if end == text_length:
                break
            start += step

        return spans

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """Chunk every page produced by ``PDFLoader.load()``.

        Args:
            documents: Page dictionaries with ``text`` and ``metadata`` keys.

        Returns:
            A list of ``Chunk`` dictionaries. ``chunk_index`` is 0-based and
            unique across the whole list, while ``page_number`` still points
            to the original PDF page.
        """
        chunks: list[Chunk] = []
        chunk_index = 0  # keeps counting across pages

        for document in documents:
            page = document["metadata"]
            page_text = document["text"]

            for chunk_text, char_start, char_end in self._split_text(page_text):
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            # Copy the page metadata so citations survive.
                            "source": page["source"],
                            "source_path": page["source_path"],
                            "page_number": page["page_number"],
                            "total_pages": page["total_pages"],
                            # Chunk-specific metadata.
                            "chunk_index": chunk_index,
                            "char_start": char_start,
                            "char_end": char_end,
                        },
                    }
                )
                chunk_index += 1

        return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
    overlap: int = TextChunker.DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Shortcut for ``TextChunker(chunk_size, overlap).chunk_documents(...)``."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_documents(
        documents
    )
