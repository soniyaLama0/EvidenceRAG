"""Tests for the experimental ``ingestion.structure_chunker`` module.

Run from the project root with::

    python -m pytest tests/test_structure_chunker.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.page_cleaner import PageCleaner
from ingestion.pdf_loader import PDFLoader
from ingestion.section_segmenter import (
    REFERENCES_SECTION,
    SectionSegmenter,
)
from ingestion.structure_chunker import (
    CHUNKING_VERSION,
    StructureAwareChunker,
    detect_heading,
    split_sentences,
)

SAMPLE_PDF = Path("data/raw/research_paper.pdf")


def make_documents(
    texts: list[str], *, total_pages: int | None = None
) -> list[dict]:
    """Build page dictionaries in the same shape PDFLoader returns."""
    total = total_pages if total_pages is not None else len(texts)
    return [
        {
            "text": text,
            "metadata": {
                "source": "paper.pdf",
                "source_path": "C:/data/paper.pdf",
                "page_number": index + 1,
                "total_pages": total,
            },
        }
        for index, text in enumerate(texts)
    ]


def word_counter(text: str) -> int:
    """Deterministic counter that includes two special tokens."""
    return len(text.split()) + 2


def char_counter(text: str) -> int:
    return max(1, len(text) // 4) + 2


def chunk_texts(chunks: list[dict]) -> str:
    return "\n".join(chunk["text"] for chunk in chunks)


# ---------------------------------------------------------------------------
# 1. Exact source-offset invariants
# ---------------------------------------------------------------------------


def test_offsets_match_page_text_exactly() -> None:
    documents = make_documents(
        [
            "I. INTRODUCTION\nFirst sentence here. Second sentence follows.\n"
            "TABLE I\nComparison\nAlpha\nBeta\nGamma\n",
            "A. Background\nHard-wrapped prose continues on this page. "
            "It keeps going for a while.\n",
        ]
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        documents
    )
    assert chunks
    texts = {doc["metadata"]["page_number"]: doc["text"] for doc in documents}
    for chunk in chunks:
        metadata = chunk["metadata"]
        page_text = texts[metadata["page_number"]]
        assert page_text[metadata["char_start"] : metadata["char_end"]] == chunk[
            "text"
        ]
        assert chunk["text"].strip()
        assert metadata["chunking_version"] == CHUNKING_VERSION


# ---------------------------------------------------------------------------
# 2. Hard-wrapped lines and offset mapping
# ---------------------------------------------------------------------------


def test_hard_wrapped_lines_map_back_to_original_offsets() -> None:
    page = (
        "This is a hard-wrapped paragraph that spans\n"
        "several physical lines without sentence breaks,\n"
        "and ends here. A second sentence is also wrapped\n"
        "across lines on the page."
    )
    chunks = StructureAwareChunker(
        token_counter=word_counter,
        target_tokens=12,
        hard_max_tokens=30,
        overlap_sentences=0,
    ).chunk_documents(make_documents([page]))

    assert len(chunks) >= 2
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert page[metadata["char_start"] : metadata["char_end"]] == chunk["text"]
        assert "\n" in chunk["text"] or metadata["char_end"] == len(page)
    joined = " ".join(chunk["text"] for chunk in chunks)
    assert "without sentence breaks" in joined
    assert "second sentence" in joined


# ---------------------------------------------------------------------------
# 3. Roman, alphabetic and numeric headings
# ---------------------------------------------------------------------------


def test_roman_alphabetic_and_numeric_headings_are_detected() -> None:
    page = (
        "I. INTRODUCTION\n"
        "The introduction has one sentence. It has another sentence.\n"
        "A. Background\n"
        "Background text follows here. It continues a little.\n"
        "2.1 Method\n"
        "The method is described here. It is deterministic.\n"
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    paths = [chunk["metadata"]["heading_path"] for chunk in chunks]
    assert any("I. INTRODUCTION" in path for path in paths)
    assert any("I. INTRODUCTION > A. Background" in path for path in paths)
    assert any("2.1 Method" in path for path in paths)
    levels = {chunk["metadata"]["heading_level"] for chunk in chunks}
    assert 1 in levels and 2 in levels
    # The heading must travel with the first content chunk.
    first = next(c for c in chunks if "I. INTRODUCTION" in c["text"])
    assert first["metadata"]["heading_text"] == "I. INTRODUCTION"


# ---------------------------------------------------------------------------
# 4. False heading candidates
# ---------------------------------------------------------------------------


def test_false_heading_candidates_stay_prose() -> None:
    page = (
        "K.J. Somaiya Institute of Technology\n"
        "Keywords: offline speech, Hinglish, MSME\n"
        "System / Reference\n"
        "The system is fast. It is reliable and private.\n"
    )
    assert detect_heading("K.J. Somaiya Institute of Technology") is None
    assert detect_heading("Keywords: offline speech") is None
    assert detect_heading("System / Reference") is None

    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    assert all(chunk["metadata"]["heading_path"] == "" for chunk in chunks)
    assert any("K.J. Somaiya" in chunk["text"] for chunk in chunks)


# ---------------------------------------------------------------------------
# 5-6. Sentence boundaries, abbreviations, decimals, citations, URLs
# ---------------------------------------------------------------------------


def test_abbreviations_do_not_end_sentences() -> None:
    text = (
        "Dr. Rao et al. reported results in Fig. 3 at 3.14 GHz. "
        "The work of Smith et al. was cited. It remains useful."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert text[sentences[0][0] : sentences[0][1]].startswith("Dr. Rao")
    assert "3.14 GHz" in text[sentences[0][0] : sentences[0][1]]
    assert text[sentences[1][0] : sentences[1][1]].startswith("The work")


def test_decimal_citation_and_url_tokens_are_not_split() -> None:
    text = (
        "Version 2.1 is stable [1][2]. "
        "See https://example.com/a.b/c?x=1.2 for details. "
        "It is the official source."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    for start, end in sentences:
        sentence = text[start:end]
        if "https://" in sentence:
            assert "https://example.com/a.b/c?x=1.2" in sentence
    assert "[1][2]" in text[sentences[0][0] : sentences[0][1]]


# ---------------------------------------------------------------------------
# 7. Numbered and bulleted lists
# ---------------------------------------------------------------------------


def test_numbered_and_bulleted_lists_are_grouped() -> None:
    page = (
        "The contributions are:\n"
        "(1) First contribution with supporting detail.\n"
        "(2) Second contribution with supporting detail.\n"
        "- A bullet item that is complete.\n"
        "* Another bullet item that is complete.\n"
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    list_chunks = [
        chunk for chunk in chunks if chunk["metadata"]["structure_type"] == "list"
    ]
    assert list_chunks
    combined = chunk_texts(list_chunks)
    for marker in ("(1)", "(2)", "- A bullet", "* Another bullet"):
        assert marker in combined
    # List units are grouped, not emitted one tiny chunk per item.
    assert len(list_chunks) < 4


# ---------------------------------------------------------------------------
# 8. Table captions and rows
# ---------------------------------------------------------------------------


def test_table_caption_and_rows_are_grouped() -> None:
    page = (
        "Introductory prose before the table. It ends here.\n"
        "TABLE II\n"
        "Intent Categories\n"
        "Intent Command\n"
        "Billing create invoice\n"
        "Inventory check stock\n"
        "Payments record payment\n"
        "Reports sales summary\n"
        "Prose after the table. It ends here.\n"
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    tables = [c for c in chunks if c["metadata"]["structure_type"] == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table["metadata"]["table_id"]
    assert table["metadata"]["table_caption"].startswith("TABLE II")
    assert "Billing create invoice" in table["text"]
    assert table["metadata"]["table_part"] == 1
    page_text = page
    metadata = table["metadata"]
    assert page_text[metadata["char_start"] : metadata["char_end"]] == table["text"]


# ---------------------------------------------------------------------------
# 9. False-positive table detection
# ---------------------------------------------------------------------------


def test_short_lines_without_caption_are_not_tables() -> None:
    page = (
        "alpha beta\n"
        "gamma delta\n"
        "epsilon zeta\n"
        "These lines are not a table.\n"
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    assert all(chunk["metadata"]["table_id"] == "" for chunk in chunks)


def test_caption_followed_by_sentences_is_not_a_row_run() -> None:
    page = (
        "TABLE I\n"
        "This is a short sentence.\n"
        "Another short sentence.\n"
        "Third short sentence.\n"
        "Prose after the caption. It ends here.\n"
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents([page])
    )
    row_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"]["table_id"] and "Third short sentence" in chunk["text"]
    ]
    assert not row_chunks
    assert any("Third short sentence" in chunk["text"] for chunk in chunks)


# ---------------------------------------------------------------------------
# 10. Disconnected captions and rows
# ---------------------------------------------------------------------------


def test_disconnected_caption_and_rows_share_table_id() -> None:
    prose = (
        "This paragraph explains the comparison in detail. "
        "It deliberately separates the caption from the rows. "
    ) * 5
    page = (
        "TABLE I\n"
        "System Comparison\n"
        "\n"
        + prose
        + "\n"
        "System A\n"
        "System B\n"
        "System C\n"
        "System D\n"
    )
    chunks = StructureAwareChunker(token_counter=char_counter).chunk_documents(
        make_documents([page])
    )
    table_chunks = [c for c in chunks if c["metadata"]["table_id"]]
    assert len(table_chunks) >= 2
    table_ids = {c["metadata"]["table_id"] for c in table_chunks}
    assert len(table_ids) == 1
    caption_chunk = next(
        c for c in table_chunks if c["metadata"]["table_part"] == 1
    )
    row_chunks = [c for c in table_chunks if c["metadata"]["table_part"] > 1]
    assert row_chunks
    assert "System Comparison" in caption_chunk["text"]
    assert "System A" not in caption_chunk["text"]
    assert all(c["metadata"]["table_caption"].startswith("TABLE I") for c in table_chunks)


# ---------------------------------------------------------------------------
# 11. Body / reference boundary preservation
# ---------------------------------------------------------------------------


def test_body_and_references_never_share_a_chunk() -> None:
    page2 = (
        "A final body sentence. It ends the body.\n"
        "XIII. REFERENCES\n"
        "[1] First reference entry. Available: https://example.com/1\n"
        "[2] Second reference entry. Available: https://example.com/2\n"
        "[3] Third reference entry. Available: https://example.com/3\n"
    )
    documents = make_documents(
        ["First body page sentence. It ends here.", page2]
    )
    chunks = StructureAwareChunker(
        token_counter=word_counter,
        target_tokens=20,
        hard_max_tokens=40,
    ).chunk_documents(documents)

    spans = SectionSegmenter().segment(documents)
    heading_offset = next(
        span["start"]
        for span in spans
        if span["page_number"] == 2 and span["section"] == REFERENCES_SECTION
    )
    assert heading_offset > 0
    for chunk in chunks:
        metadata = chunk["metadata"]
        if metadata["page_number"] != 2:
            continue
        if metadata["section"] == "body":
            assert metadata["char_end"] <= heading_offset
        elif metadata["section"] == REFERENCES_SECTION:
            assert metadata["char_start"] >= heading_offset
        else:
            pytest.fail(f"unexpected section {metadata['section']!r}")
    assert any(
        chunk["metadata"]["section"] == REFERENCES_SECTION
        and chunk["metadata"]["structure_type"] == "reference_entry"
        for chunk in chunks
    )


# ---------------------------------------------------------------------------
# 12. Long sentences and forced splitting
# ---------------------------------------------------------------------------


def test_long_sentence_is_force_split_and_flagged() -> None:
    long_sentence = (
        "This sentence is deliberately long and keeps going "
        + "with many extra words " * 40
        + "until it finally ends."
    )
    chunks = StructureAwareChunker(
        token_counter=word_counter,
        target_tokens=20,
        hard_max_tokens=30,
        overlap_sentences=0,
    ).chunk_documents(make_documents([long_sentence]))

    assert len(chunks) > 1
    assert any(chunk["metadata"]["is_forced_split"] for chunk in chunks)
    assert all(chunk["metadata"]["token_count"] <= 30 for chunk in chunks)
    joined = " ".join(chunk["text"] for chunk in chunks)
    assert joined.split() == long_sentence.split()


# ---------------------------------------------------------------------------
# 13-14. Token counts including special tokens and the hard limit
# ---------------------------------------------------------------------------


def test_token_counts_include_special_tokens_and_respect_hard_limit() -> None:
    paragraph = (
        "This paragraph contains enough words to create several chunks. " * 20
    )
    chunks = StructureAwareChunker(
        token_counter=word_counter,
        target_tokens=25,
        hard_max_tokens=40,
        overlap_sentences=0,
    ).chunk_documents(make_documents([paragraph]))

    assert len(chunks) > 1
    for chunk in chunks:
        expected = word_counter(chunk["text"])
        assert chunk["metadata"]["token_count"] == expected
        assert expected <= 40  # includes the two special tokens


def test_every_chunk_is_within_the_254_token_default() -> None:
    paragraph = (
        "A long body paragraph with ordinary words and punctuation. " * 120
    )
    chunks = StructureAwareChunker(token_counter=char_counter).chunk_documents(
        make_documents([paragraph])
    )
    assert chunks
    assert all(chunk["metadata"]["token_count"] <= 254 for chunk in chunks)
    assert max(chunk["metadata"]["token_count"] for chunk in chunks) > 100


# ---------------------------------------------------------------------------
# 15. Deterministic output
# ---------------------------------------------------------------------------


def test_output_is_deterministic() -> None:
    documents = make_documents(
        [
            "I. INTRODUCTION\nOne sentence. Two sentences.\n"
            "TABLE I\nItems\nAlpha\nBeta\nGamma\n",
            "A. Background\nMore prose here. It ends here.\n",
        ]
    )
    chunker = StructureAwareChunker(token_counter=char_counter)
    first = chunker.chunk_documents(documents)
    second = chunker.chunk_documents(documents)
    assert first == second


# ---------------------------------------------------------------------------
# 16. Empty pages and unusual input
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_pages_are_skipped() -> None:
    documents = make_documents(["", "   \n\t\n", "Hello."])
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        documents
    )
    assert chunks
    assert {chunk["metadata"]["page_number"] for chunk in chunks} == {3}


def test_single_word_page_is_preserved() -> None:
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        make_documents(["Hello"])
    )
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello"


def test_invalid_page_order_is_rejected() -> None:
    documents = make_documents(["one", "two"])
    documents[0]["metadata"]["page_number"] = 2
    documents[1]["metadata"]["page_number"] = 1
    with pytest.raises(ValueError):
        StructureAwareChunker().chunk_documents(documents)


def test_non_dictionary_document_is_rejected() -> None:
    with pytest.raises(TypeError):
        StructureAwareChunker().chunk_documents(["not a page"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 17. No content loss or duplication beyond configured overlap
# ---------------------------------------------------------------------------


def test_no_content_loss_and_no_duplication_without_overlap() -> None:
    page = (
        "I. INTRODUCTION\n"
        "First paragraph sentence. Second paragraph sentence.\n"
        "(1) A list item.\n"
        "(2) Another list item.\n"
        "TABLE I\nCaption title\nAlpha\nBeta\nGamma\n"
        "A closing paragraph. It ends the page.\n"
    )
    documents = make_documents([page])
    chunks = StructureAwareChunker(
        token_counter=char_counter, overlap_sentences=0
    ).chunk_documents(documents)

    covered: set[int] = set()
    for chunk in chunks:
        metadata = chunk["metadata"]
        span = set(range(metadata["char_start"], metadata["char_end"]))
        assert not (covered & span), "unexpected duplicate characters"
        covered |= span

    non_whitespace = {index for index, char in enumerate(page) if not char.isspace()}
    assert non_whitespace <= covered


def test_overlap_is_at_most_one_sentence_and_capped() -> None:
    page = (
        "This is the first sentence of the page. "
        "This is the second sentence with more words. "
        "This is the third sentence with even more words. "
        "This is the fourth sentence and it concludes."
    )
    chunks = StructureAwareChunker(
        token_counter=word_counter,
        target_tokens=15,
        hard_max_tokens=25,
        overlap_sentences=1,
        max_overlap_chars=200,
        max_overlap_ratio=1.0,
    ).chunk_documents(make_documents([page]))

    assert len(chunks) >= 2
    overlaps = []
    for previous, current in zip(chunks, chunks[1:]):
        overlap = previous["metadata"]["char_end"] - current["metadata"]["char_start"]
        assert overlap <= 200
        overlaps.append(overlap)
        assert current["metadata"]["overlap_sentences"] <= 1
    assert any(overlap > 0 for overlap in overlaps)


# ---------------------------------------------------------------------------
# 18. Pipeline / chunk_documents interface compatibility
# ---------------------------------------------------------------------------


def test_chunk_documents_interface_and_scalar_metadata() -> None:
    documents = make_documents(
        [
            "I. INTRODUCTION\nOne sentence. Two sentences.\n"
            "TABLE I\nItems\nAlpha\nBeta\nGamma\n",
            "A. Background\nMore prose. It ends here.\n",
        ]
    )
    chunks = StructureAwareChunker(token_counter=word_counter).chunk_documents(
        documents
    )
    assert isinstance(chunks, list)
    required = {
        "source",
        "source_path",
        "page_number",
        "total_pages",
        "chunk_index",
        "char_start",
        "char_end",
        "section",
        "chunking_version",
        "structure_type",
        "structure_confidence",
        "heading_path",
        "heading_level",
        "heading_text",
        "table_id",
        "table_caption",
        "table_part",
        "block_index",
        "starts_mid_sentence",
        "ends_mid_sentence",
        "is_forced_split",
        "overlap_sentences",
    }
    for index, chunk in enumerate(chunks):
        assert set(required) <= set(chunk["metadata"])
        assert chunk["metadata"]["chunk_index"] == index
        for value in chunk["metadata"].values():
            assert isinstance(value, (str, int, float, bool))
            assert value is not None

    spans = SectionSegmenter().segment(documents)
    labelled = SectionSegmenter().assign_to_chunks(chunks, spans)
    assert all("section" in chunk["metadata"] for chunk in labelled)
    assert all("cleaning_version" in chunk["metadata"] for chunk in labelled)


# ---------------------------------------------------------------------------
# Real paper (optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF is not available")
def test_real_paper_structures_and_invariants() -> None:
    pages = PageCleaner().clean_documents(
        PDFLoader(str(SAMPLE_PDF)).load(skip_empty_pages=False)
    )
    chunks = StructureAwareChunker().chunk_documents(pages)
    assert chunks

    texts = {page["metadata"]["page_number"]: page["text"] for page in pages}
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert texts[metadata["page_number"]][
            metadata["char_start"] : metadata["char_end"]
        ] == chunk["text"]
        assert metadata["token_count"] <= 254

    spans = SectionSegmenter().segment(pages)
    heading_offsets = {
        span["page_number"]: span["start"]
        for span in spans
        if span["section"] == REFERENCES_SECTION
    }
    for chunk in chunks:
        metadata = chunk["metadata"]
        if metadata["page_number"] not in heading_offsets:
            continue
        heading = heading_offsets[metadata["page_number"]]
        if metadata["section"] == "body":
            assert metadata["char_end"] <= heading
        else:
            assert metadata["char_start"] >= heading

    assert any(chunk["metadata"]["heading_path"] for chunk in chunks)
    assert any(chunk["metadata"]["table_id"] for chunk in chunks)
