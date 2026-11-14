"""Tests for ``ingestion.pdf_loader``.

Run from the project root with::

    python -m pytest tests/test_pdf_loader.py -v

The tests that need a PDF generate a small one on the fly with reportlab.
To also test your own research PDF, set the ``EVIDENCERAG_SAMPLE_PDF``
environment variable to its path before running pytest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from ingestion.pdf_loader import PDFLoader, load_pdf

# Set this to the path of a real research PDF to enable the extra test below.
SAMPLE_ENV_VAR = "EVIDENCERAG_SAMPLE_PDF"


def make_sample_pdf(path: Path) -> None:
    """Write a 3-page PDF: page 1 text, page 2 blank, page 3 text."""
    pdf = canvas.Canvas(str(path), pagesize=(612, 792))
    pdf.setFont("Helvetica", 12)

    # Page 1
    pdf.drawString(72, 720, "EvidenceRAG sample research paper")
    pdf.drawString(72, 700, "Page one: introduction and motivation.")
    pdf.showPage()

    # Page 2: intentionally left blank to test empty-page handling.
    pdf.showPage()

    # Page 3
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 720, "Page three: results and discussion.")
    pdf.showPage()

    pdf.save()


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a temporary 3-page PDF and return its path."""
    path = tmp_path / "evidence_sample.pdf"
    make_sample_pdf(path)
    return path


def test_extracts_text_and_metadata(sample_pdf: Path) -> None:
    """Text, source filename, and 1-based page numbers survive extraction."""
    documents = PDFLoader(sample_pdf).load()

    # Page 2 is blank, so it is skipped by default.
    assert len(documents) == 2

    first, second = documents
    assert "EvidenceRAG sample research paper" in first["text"]
    assert "introduction and motivation" in first["text"]
    assert "results and discussion" in second["text"]

    # Source filename (not the full path) is stored in metadata.
    assert first["metadata"]["source"] == "evidence_sample.pdf"
    assert second["metadata"]["source"] == "evidence_sample.pdf"
    assert first["metadata"]["source_path"] == str(sample_pdf)
    assert first["metadata"]["total_pages"] == 3

    # Page numbers are 1-based and refer to the original PDF, so the
    # blank page 2 still counts: the kept pages are 1 and 3.
    assert first["metadata"]["page_number"] == 1
    assert second["metadata"]["page_number"] == 3


def test_blank_pages_kept_when_requested(sample_pdf: Path) -> None:
    """With skip_empty_pages=False the blank page is returned with empty text."""
    documents = load_pdf(sample_pdf, skip_empty_pages=False)

    assert len(documents) == 3
    assert [doc["metadata"]["page_number"] for doc in documents] == [1, 2, 3]
    assert documents[1]["text"] == ""
    assert documents[1]["metadata"]["source"] == "evidence_sample.pdf"


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing path fails fast with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        PDFLoader(tmp_path / "does_not_exist.pdf")


def test_rejects_non_pdf_extension(tmp_path: Path) -> None:
    """Only .pdf files are accepted."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        PDFLoader(text_file)


@pytest.mark.skipif(
    not os.environ.get(SAMPLE_ENV_VAR),
    reason=f"set {SAMPLE_ENV_VAR} to a real research PDF to run this test",
)
def test_user_provided_research_pdf() -> None:
    """Sanity-check the loader against a real research PDF you supply."""
    pdf_path = Path(os.environ[SAMPLE_ENV_VAR])

    documents = PDFLoader(pdf_path).load()

    assert documents, "Expected at least one page with extractable text"
    assert all(doc["text"].strip() for doc in documents)
    assert all(doc["metadata"]["source"] == pdf_path.name for doc in documents)
    assert all(doc["metadata"]["total_pages"] >= len(documents) for doc in documents)

    # Page numbers must be 1-based and strictly increasing, even if some
    # pages were skipped because they had no text.
    page_numbers = [doc["metadata"]["page_number"] for doc in documents]
    assert page_numbers[0] >= 1
    assert page_numbers == sorted(set(page_numbers))
