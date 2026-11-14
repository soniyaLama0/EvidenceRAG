"""Tests for ``ingestion.text_chunker``.

Run from the project root with::

    python -m pytest tests/test_text_chunker.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from ingestion.pdf_loader import PDFLoader
from ingestion.text_chunker import Chunk, TextChunker, chunk_documents


def make_document(
    text: str,
    *,
    source: str = "paper.pdf",
    page_number: int = 1,
    total_pages: int = 1,
) -> dict:
    """Build one page dictionary in the same shape PDFLoader returns."""
    return {
        "text": text,
        "metadata": {
            "source": source,
            "source_path": f"C:/data/{source}",
            "page_number": page_number,
            "total_pages": total_pages,
        },
    }


def test_default_configuration() -> None:
    """The requested defaults are 1000 characters with 200 overlap."""
    chunker = TextChunker()
    assert chunker.chunk_size == 1000
    assert chunker.overlap == 200


def test_sliding_window_positions() -> None:
    """Windows advance by chunk_size - overlap and share characters."""
    text = "abcdefghij" * 250  # 2500 characters of known content
    chunks = TextChunker().chunk_documents([make_document(text)])

    # Starts: 0, 800, 1600 (step = 1000 - 200). The third window already
    # reaches the end of the page, so there is no fourth chunk.
    assert len(chunks) == 3
    expected_spans = [(0, 1000), (800, 1800), (1600, 2500)]

    for chunk, (start, end) in zip(chunks, expected_spans):
        assert (chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]) == (
            start,
            end,
        )
        assert chunk["text"] == text[start:end]

    # The last chunk is shorter because the page ended.
    assert len(chunks[-1]["text"]) == 900
    # Neighbouring chunks really do overlap by exactly 200 characters.
    assert chunks[1]["metadata"]["char_start"] == 800


def test_preserves_source_and_page_metadata() -> None:
    """Page metadata is copied onto each chunk; chunk_index is unique."""
    pages = [
        make_document("A" * 1500, source="study.pdf", page_number=1, total_pages=3),
        make_document("B" * 300, source="study.pdf", page_number=3, total_pages=3),
    ]
    chunks = chunk_documents(pages)

    # Page 1 -> 2 chunks (0-1000, 800-1500), page 3 -> 1 chunk.
    assert len(chunks) == 3
    assert [c["metadata"]["page_number"] for c in chunks] == [1, 1, 3]
    assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1, 2]
    assert all(c["metadata"]["source"] == "study.pdf" for c in chunks)
    assert all(c["metadata"]["source_path"] == "C:/data/study.pdf" for c in chunks)
    assert all(c["metadata"]["total_pages"] == 3 for c in chunks)


def test_offsets_match_page_text_after_whitespace_trimming() -> None:
    """trimmed chunk text still maps exactly to the page via its offsets."""
    page_text = "   \nHello world. This is EvidenceRAG.  \n"
    chunks = TextChunker(chunk_size=100, overlap=10).chunk_documents(
        [make_document(page_text)]
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["text"] == "Hello world. This is EvidenceRAG."
    assert page_text[
        chunk["metadata"]["char_start"] : chunk["metadata"]["char_end"]
    ] == chunk["text"]


def test_empty_and_whitespace_pages_are_skipped() -> None:
    """No chunks are produced for empty pages or an empty input list."""
    pages = [
        make_document(""),
        make_document("   \n\t  ", page_number=2),
        make_document("Real content", page_number=3),
    ]
    chunks = chunk_documents(pages)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Real content"
    assert chunks[0]["metadata"]["page_number"] == 3
    assert chunk_documents([]) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),  # chunk_size must be positive
        (-10, 0),  # chunk_size must be positive
        (1000, -1),  # overlap cannot be negative
        (1000, 1000),  # overlap must be smaller than chunk_size
        (1000, 1200),  # overlap must be smaller than chunk_size
        ("1000", 200),  # chunk_size must be an integer
    ],
)
def test_invalid_configurations_raise(chunk_size: object, overlap: object) -> None:
    """Bad chunk_size/overlap values fail fast with a clear ValueError."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=chunk_size, overlap=overlap)  # type: ignore[arg-type]


def test_pdf_loader_to_chunker_integration(tmp_path: Path) -> None:
    """A real PDF flows through PDFLoader and TextChunker with correct cites."""
    pdf_path = tmp_path / "paper.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(612, 792))

    # Page 1: enough lines to need several chunks.
    pdf.setFont("Helvetica", 9)
    for line_number in range(1, 41):
        pdf.drawString(
            40,
            780 - (line_number % 75) * 10,
            f"Line {line_number}: evidence text used for chunking checks.",
        )
    pdf.showPage()

    # Page 2: a short page that becomes a single chunk.
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, 760, "Short conclusion on page two.")
    pdf.showPage()
    pdf.save()

    pages = PDFLoader(pdf_path).load()
    chunks = TextChunker().chunk_documents(pages)

    assert len(pages) == 2
    assert len(chunks) > 2  # page 1 needed at least two chunks

    page_text_by_number = {
        page["metadata"]["page_number"]: page["text"] for page in pages
    }

    for chunk in chunks:
        metadata = chunk["metadata"]
        page_text = page_text_by_number[metadata["page_number"]]
        # Every chunk can be located exactly on its original page.
        assert page_text[metadata["char_start"] : metadata["char_end"]] == chunk["text"]
        assert metadata["source"] == "paper.pdf"
        assert metadata["total_pages"] == 2

    # chunk_index values are unique and sequential across both pages.
    indices = [chunk["metadata"]["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))
    # Page 2's chunk keeps the original page number, not a flat index.
    assert chunks[-1]["metadata"]["page_number"] == 2
