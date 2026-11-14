"""Tests for ``ingestion.section_segmenter``.

Synthetic pages cover the detection rules; two integration tests run against
``data/raw/research_paper.pdf`` to pin the approved page-5 boundary at 2402.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_section_segmenter.py -v
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ingestion.page_cleaner import CLEANING_VERSION, PageCleaner
from ingestion.pdf_loader import PDFLoader
from ingestion.section_segmenter import (
    BODY_SECTION,
    REFERENCES_SECTION,
    SectionSegmenter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PDF = PROJECT_ROOT / "data" / "raw" / "research_paper.pdf"

PAGE_FIVE_HEADING = "                               XIII. REFERENCES"


def make_document(
    text: str,
    page_number: int,
    *,
    total_pages: int = 6,
    source: str = "research_paper.pdf",
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


def body_page(page_number: int, text: str | None = None) -> dict:
    return make_document(
        text or f"Body text for page {page_number}. " * 8,
        page_number,
    )


def references_block(
    *,
    heading: str = PAGE_FIVE_HEADING,
    entries: list[str] | None = None,
) -> str:
    entries = entries or [
        "[1]  IRJET. Voice Based Billing System. Available:",
        "https://www.irjet.net/archives/V6/i3/IRJET-V6I31136.pdf",
        "[2]  IJARSCT. Voice Based Billing System. Available:",
        "https://ijarsct.co.in/Paper25159.pdf",
        "[3]  AI Accountant. Voice-Based Accounting Entry.",
    ]
    return heading + "\n" + "\n".join(entries)


def reference_documents() -> list[dict]:
    """Six pages with body on 1-4, references starting on page 5."""
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            "Body text before the references. " * 5 + "\n" + references_block(),
            5,
        )
    )
    documents.append(
        make_document(
            "[4]  Academia.edu / IJRASET. Design and Development of a "
            "Privacy-Preserving Offline AI Assistant for MSMEs.\n"
            "[5]  IJRASET. Building an Offline Virtual Voice Assistant using "
            "AI and NLP.",
            6,
        )
    )
    return documents


def make_chunk(
    text: str,
    *,
    page_number: int,
    char_start: int,
    char_end: int,
    chunk_index: int = 0,
) -> dict:
    """Chunk in the exact shape produced by TextChunker and EmbeddingModel."""
    return {
        "text": text[char_start:char_end],
        "metadata": {
            "source": "research_paper.pdf",
            "source_path": "C:/data/research_paper.pdf",
            "page_number": page_number,
            "total_pages": 6,
            "chunk_index": chunk_index,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": 200,
            "is_truncated": False,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_references_heading() -> None:
    documents = reference_documents()
    segmenter = SectionSegmenter()

    heading = segmenter.find_references_heading(documents)

    assert heading is not None
    page_index, offset = heading
    assert page_index == 4
    assert offset == documents[4]["text"].index("XIII. REFERENCES")


def test_segment_spans_split_heading_page() -> None:
    documents = reference_documents()
    spans = SectionSegmenter().segment(documents)

    page_five = [span for span in spans if span["page_number"] == 5]
    assert [(span["section"], span["start"]) for span in page_five] == [
        (BODY_SECTION, 0),
        (REFERENCES_SECTION, documents[4]["text"].index("XIII. REFERENCES")),
    ]
    assert page_five[0]["end"] == page_five[1]["start"]
    assert page_five[1]["end"] == len(documents[4]["text"])


def test_page_six_is_entirely_references() -> None:
    spans = SectionSegmenter().segment(reference_documents())
    page_six = [span for span in spans if span["page_number"] == 6]

    assert len(page_six) == 1
    assert page_six[0]["section"] == REFERENCES_SECTION
    assert page_six[0]["start"] == 0
    assert page_six[0]["end"] == len(reference_documents()[5]["text"])


def test_real_paper_page_five_boundary_is_2402() -> None:
    if not RESEARCH_PDF.exists():
        pytest.skip("research_paper.pdf is not available")

    raw_documents = PDFLoader(RESEARCH_PDF).load()
    spans = SectionSegmenter().segment(raw_documents)
    page_five = [span for span in spans if span["page_number"] == 5]

    heading_spans = [s for s in page_five if s["section"] == REFERENCES_SECTION]
    assert len(heading_spans) == 1
    assert heading_spans[0]["start"] == 2402
    assert heading_spans[0]["end"] == len(raw_documents[4]["text"])

    body_span = [s for s in page_five if s["section"] == BODY_SECTION][0]
    assert body_span["end"] == 2402
    assert raw_documents[4]["text"][2402:2418] == "XIII. REFERENCES"


def test_real_paper_boundary_survives_cleaning() -> None:
    if not RESEARCH_PDF.exists():
        pytest.skip("research_paper.pdf is not available")

    raw_documents = PDFLoader(RESEARCH_PDF).load()
    cleaned_documents = PageCleaner().clean_documents(raw_documents)
    spans = SectionSegmenter().segment(cleaned_documents)  # type: ignore[arg-type]

    page_five = [span for span in spans if span["page_number"] == 5]
    assert [span["section"] for span in page_five] == [
        BODY_SECTION,
        REFERENCES_SECTION,
    ]
    assert page_five[0]["end"] == 2402
    assert page_five[1]["start"] == 2402
    assert cleaned_documents[4]["text"][2402:2418] == "XIII. REFERENCES"


def test_real_paper_page_six_is_entirely_references() -> None:
    if not RESEARCH_PDF.exists():
        pytest.skip("research_paper.pdf is not available")

    documents = PDFLoader(RESEARCH_PDF).load()
    spans = SectionSegmenter().segment(documents)
    page_six = [span for span in spans if span["page_number"] == 6]

    assert len(page_six) == 1
    assert page_six[0]["section"] == REFERENCES_SECTION
    assert page_six[0]["start"] == 0
    assert page_six[0]["end"] == len(documents[5]["text"])


# ---------------------------------------------------------------------------
# Conservative rejection
# ---------------------------------------------------------------------------


def test_rejects_table_header_system_reference() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            "TABLE I\nSystem / Reference\n[1]  one\n[2]  two\n[3]  three",
            5,
        )
    )
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_ordinary_prose_mentioning_references() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            "This section describes the references used in this work and how "
            "they were selected.\n[1] one\n[2] two\n[3] three",
            5,
        )
    )
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_lowercase_heading() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(make_document(references_block(heading="References"), 5))
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_heading_too_early_in_the_document() -> None:
    documents = [make_document(references_block(), 1)]
    documents.extend(body_page(number) for number in range(2, 7))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_fewer_than_three_entries() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            references_block(entries=["[1]  one", "[2]  two"]),
            5,
        )
    )
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_entries_that_do_not_start_at_one() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            references_block(entries=["[2]  two", "[3]  three", "[4]  four"]),
            5,
        )
    )
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_rejects_entries_that_are_not_ascending() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            references_block(entries=["[3]  three", "[1]  one", "[2]  two"]),
            5,
        )
    )
    documents.append(body_page(6))

    assert SectionSegmenter().find_references_heading(documents) is None


def test_fails_open_without_a_heading() -> None:
    documents = [body_page(number) for number in range(1, 7)]
    segmenter = SectionSegmenter()

    assert segmenter.find_references_heading(documents) is None
    spans = segmenter.segment(documents)

    assert len(spans) == 6
    assert all(span["section"] == BODY_SECTION for span in spans)
    assert [(span["start"], span["end"]) for span in spans] == [
        (0, len(document["text"])) for document in documents
    ]


def test_empty_document_returns_no_spans() -> None:
    assert SectionSegmenter().segment([]) == []


# ---------------------------------------------------------------------------
# Text preservation and citation lookup
# ---------------------------------------------------------------------------


def test_segmentation_never_changes_page_text() -> None:
    documents = reference_documents()
    before = copy.deepcopy(documents)

    spans = SectionSegmenter().segment(documents)
    SectionSegmenter().section_text(documents, spans, REFERENCES_SECTION)

    assert documents == before
    assert "[4][5][6]" not in documents[4]["text"]  # sanity: citations are on p2


def test_body_citations_are_preserved_in_body_spans() -> None:
    documents = [body_page(1), body_page(2, "text [4][5][6] and [13][15]")]
    documents.extend(body_page(number) for number in range(3, 7))

    spans = SectionSegmenter().segment(documents)
    body_text = "\n".join(
        item["text"]
        for item in SectionSegmenter().section_text(
            documents, spans, BODY_SECTION
        )
    )

    assert "[4][5][6]" in body_text
    assert "[13][15]" in body_text


def test_handles_wrapped_reference_entries() -> None:
    documents = [body_page(number) for number in range(1, 5)]
    documents.append(
        make_document(
            references_block(
                entries=[
                    "[1]  First reference. Available:",
                    "https://example.org/one",
                    "[2]  Second reference. Journal of Testing,",
                    "Vol. 4, Issue 4, pp. 1-10.",
                    "[3]  Third reference.",
                ]
            ),
            5,
        )
    )
    documents.append(body_page(6))

    segmenter = SectionSegmenter()
    spans = segmenter.segment(documents)
    entries = segmenter.reference_entries(documents, spans)

    assert [entry["number"] for entry in entries] == [1, 2, 3]
    assert entries[0]["text"] == "First reference. Available: https://example.org/one"
    assert "Vol. 4, Issue 4, pp. 1-10." in entries[1]["text"]
    assert entries[1]["page_number"] == 5


def test_section_text_exposes_reference_section() -> None:
    documents = reference_documents()
    spans = SectionSegmenter().segment(documents)

    reference_text = SectionSegmenter().section_text(
        documents, spans, REFERENCES_SECTION
    )

    assert {item["page_number"] for item in reference_text} == {5, 6}
    assert "XIII. REFERENCES" in reference_text[0]["text"]
    assert "[4]  Academia.edu" in reference_text[1]["text"]


def test_invalid_section_name_is_rejected() -> None:
    documents = reference_documents()
    spans = SectionSegmenter().segment(documents)

    with pytest.raises(ValueError, match="section must be"):
        SectionSegmenter().section_text(documents, spans, "appendix")


# ---------------------------------------------------------------------------
# Chunk assignment
# ---------------------------------------------------------------------------


def test_assigns_sections_to_chunks_by_overlap() -> None:
    documents = reference_documents()
    page_five = documents[4]["text"]
    page_six = documents[5]["text"]
    boundary = page_five.index("XIII. REFERENCES")

    chunks = [
        make_chunk(page_five, page_number=5, char_start=0, char_end=boundary,
                   chunk_index=10),
        make_chunk(page_five, page_number=5, char_start=boundary - 20,
                   char_end=boundary + 180, chunk_index=11),
        make_chunk(page_five, page_number=5, char_start=boundary,
                   char_end=len(page_five), chunk_index=12),
        make_chunk(page_six, page_number=6, char_start=0,
                   char_end=len(page_six), chunk_index=13),
    ]
    before = copy.deepcopy(chunks)

    spans = SectionSegmenter().segment(documents)
    labelled = SectionSegmenter().assign_to_chunks(chunks, spans)

    assert [chunk["metadata"]["section"] for chunk in labelled] == [
        BODY_SECTION,
        REFERENCES_SECTION,
        REFERENCES_SECTION,
        REFERENCES_SECTION,
    ]
    assert all(
        chunk["metadata"]["cleaning_version"] == CLEANING_VERSION
        for chunk in labelled
    )
    assert [chunk["text"] for chunk in labelled] == [chunk["text"] for chunk in chunks]
    # Input chunks are not mutated.
    assert chunks == before
    assert all("section" not in chunk["metadata"] for chunk in before)


def test_assign_uses_page_specific_boundaries() -> None:
    documents = reference_documents()
    boundary = documents[4]["text"].index("XIII. REFERENCES")
    spans = SectionSegmenter().segment(documents)

    chunk = make_chunk(
        documents[4]["text"],
        page_number=5,
        char_start=boundary - 100,
        char_end=boundary - 10,
    )
    labelled = SectionSegmenter().assign_to_chunks([chunk], spans)[0]

    assert labelled["metadata"]["section"] == BODY_SECTION


def test_assign_rejects_chunks_for_unknown_pages() -> None:
    documents = reference_documents()
    spans = SectionSegmenter().segment(documents)
    chunk = make_chunk("text", page_number=99, char_start=0, char_end=4)

    with pytest.raises(ValueError, match="no section spans"):
        SectionSegmenter().assign_to_chunks([chunk], spans)


def test_assign_validation() -> None:
    spans = SectionSegmenter().segment(reference_documents())

    with pytest.raises(ValueError, match="character offsets"):
        SectionSegmenter().assign_to_chunks(
            [
                {
                    "text": "x",
                    "metadata": {
                        "page_number": 5,
                        "char_start": 10,
                        "char_end": 5,
                    },
                }
            ],
            spans,
        )

    with pytest.raises(ValueError, match="integer"):
        SectionSegmenter().assign_to_chunks(
            [{"text": "x", "metadata": {"page_number": 5}}],
            spans,
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_documents_must_be_in_page_order() -> None:
    documents = [body_page(2), body_page(1)]

    with pytest.raises(ValueError, match="strictly increasing"):
        SectionSegmenter().segment(documents)


def test_document_text_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="text"):
        SectionSegmenter().segment(
            [{"text": 5, "metadata": {"page_number": 1}}]  # type: ignore[dict-item]
        )


def test_document_page_number_must_be_positive() -> None:
    with pytest.raises(ValueError, match="page_number"):
        SectionSegmenter().segment(
            [{"text": "body", "metadata": {"page_number": 0}}]
        )
