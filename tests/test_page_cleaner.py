"""Tests for ``ingestion.page_cleaner``.

The synthetic pages mirror the real paper's footer exactly. One integration
test also runs the cleaner over ``data/raw/research_paper.pdf`` when present.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_page_cleaner.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.page_cleaner import (
    CLEANING_VERSION,
    FOOTER_LINES,
    MIN_FOOTER_LINES,
    PAGE_NUMBER_PATTERN,
    PageCleaner,
    clean_page,
    is_footer_line,
)
from ingestion.pdf_loader import PDFLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_PDF = PROJECT_ROOT / "data" / "raw" / "research_paper.pdf"

PAGE_FOUR_BRIDGE = "Integration with WhatsApp through deep links allows for"
CITATION_LINE = (
    "[4][5][6]. The systematic study of offline speech recognition\n"
    "limitations in areas with poor connectivity was used as a\n"
    "source when deciding on the on-device speech engine [7]."
)


def footer_lines(page_number: int = 1) -> list[str]:
    """The exact five-line IJERT banner, with the real page number."""
    return [
        "Published by : International Journal of Engineering Research & Technology (IJERT)",
        "https://www.ijert.org/ ISSN: 2278-0181",
        "An International Peer-Reviewed Journal Vol. 15 Issue 06 , June - 2026",
        f"IJERTV15IS061078 Page {page_number}",
        "(This work is licensed under a Creative Commons Attribution 4.0 International License.)",
    ]


def make_page(body: str, page_number: int = 1, *, footer: bool = True) -> str:
    """Build one page exactly as ``PDFLoader`` would return it."""
    if not footer:
        return body
    return body + "\n" + "\n".join(footer_lines(page_number))


# ---------------------------------------------------------------------------
# Footer recognition and removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_number", [1, 2, 3, 4, 5, 6])
def test_removes_all_six_known_page_footers(page_number: int) -> None:
    body = "Body text about Vyapar AI and the MSME market."
    page = make_page(body, page_number)

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == body
    assert report["footer_detected"] is True
    assert report["removed_lines"] == footer_lines(page_number)
    assert report["removed_char_count"] == len(page) - len(body)
    assert report["removal_ratio"] == pytest.approx(
        (len(page) - len(body)) / len(page)
    )
    assert report["original_char_count"] == len(page)
    assert report["cleaned_char_count"] == len(body)


def test_real_pdf_removes_every_footer() -> None:
    if not RESEARCH_PDF.exists():
        pytest.skip("research_paper.pdf is not available")

    documents = PDFLoader(RESEARCH_PDF).load()
    cleaner = PageCleaner()

    for document in documents:
        raw = document["text"]
        cleaned, report = cleaner.clean(raw)
        page_number = document["metadata"]["page_number"]

        assert report["footer_detected"] is True, page_number
        assert report["removed_lines"] == footer_lines(page_number)
        assert "Published by : International Journal" not in cleaned
        assert "IJERTV15IS061078 Page" not in cleaned
        assert "An International Peer-Reviewed Journal" not in cleaned
        assert len(cleaned) == len(raw) - report["removed_char_count"]


def test_footer_detection_is_signature_based() -> None:
    assert is_footer_line(footer_lines(1)[0])
    assert is_footer_line("IJERTV15IS061078 Page 6")
    assert is_footer_line("  IJERTV15IS061078 Page 6  ")
    assert not is_footer_line("References")
    assert not is_footer_line("[4][5][6].")
    assert not is_footer_line("Yes")
    assert not is_footer_line("IJERTV15IS061078 Page six")


# ---------------------------------------------------------------------------
# Content that must survive
# ---------------------------------------------------------------------------


def test_preserves_page_four_sentence_bridge() -> None:
    page = make_page(PAGE_FOUR_BRIDGE, page_number=4)

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == PAGE_FOUR_BRIDGE
    assert cleaned.endswith("allows for")
    assert report["footer_detected"] is True


def test_preserves_body_citations() -> None:
    page = make_page(CITATION_LINE, page_number=2)

    cleaned, _ = PageCleaner().clean(page)

    assert cleaned == CITATION_LINE
    assert "[4][5][6]" in cleaned
    assert "The systematic study of offline speech recognition" in cleaned


def test_preserves_abbreviation_definitions() -> None:
    abbreviations = "\n".join(
        [
            "X. LIST OF ABREVATIONS",
            "1. MSME  –  Micro, Small and Medium Enterprise",
            "2. NLP     –  Natural Language Processing",
            "3. STT     –  Speech-to-Text",
            "10. UI/UX  –  User Interface / User Experience",
        ]
    )
    page = make_page(abbreviations, page_number=5)

    cleaned, _ = PageCleaner().clean(page)

    assert cleaned == abbreviations
    assert "1. MSME  –  Micro, Small and Medium Enterprise" in cleaned
    assert "10. UI/UX  –  User Interface / User Experience" in cleaned


def test_preserves_repeated_table_cells() -> None:
    table = "\n".join(
        ["TABLE IV", "Feature Status Offline Capable"]
        + ["Implemented Yes"] * 10
        + ["On-device STT (Vosk) Implemented Yes"]
    )
    page = make_page(table, page_number=4)

    cleaned, _ = PageCleaner().clean(page)

    assert cleaned == table
    assert cleaned.count("Implemented Yes") == 11


def test_does_not_remove_arbitrary_short_or_uppercase_lines() -> None:
    body = "\n".join(
        [
            "TABLE IV",
            "Yes",
            "REFERENCES AND NOTES",
            "1. MSME  –  Micro, Small and Medium Enterprise",
            "[13][14].",
            "Voice-",
            "first",
        ]
    )

    cleaned, report = PageCleaner().clean(body)

    assert cleaned == body
    assert report["footer_detected"] is False
    assert report["removed_lines"] == []
    assert report["removed_char_count"] == 0


def test_body_lines_before_the_footer_are_retained() -> None:
    body = "\n".join(
        [
            "TABLE IV",
            "1. MSME  –  Micro, Small and Medium Enterprise",
            "[4][5][6]. The systematic study of offline speech recognition",
            "Implemented Yes",
            "Yes",
        ]
    )
    page = make_page(body, page_number=4)

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == body
    assert report["removed_lines"] == footer_lines(4)


# ---------------------------------------------------------------------------
# Safety and idempotence
# ---------------------------------------------------------------------------


def test_cleaning_is_idempotent() -> None:
    page = make_page("Body text.\nSecond line.", page_number=3)
    cleaner = PageCleaner()

    first, first_report = cleaner.clean(page)
    second, second_report = cleaner.clean(first)

    assert second == first
    assert first_report["footer_detected"] is True
    assert second_report["footer_detected"] is False
    assert second_report["removed_lines"] == []


def test_fails_open_with_too_few_footer_lines() -> None:
    page = "Body text.\n" + "\n".join(footer_lines(2)[:2])

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == page
    assert report["footer_detected"] is False
    assert report["original_char_count"] == report["cleaned_char_count"]


def test_fails_open_with_only_page_number_lines() -> None:
    page = "\n".join(
        [
            "Body text.",
            "IJERTV15IS061078 Page 1",
            "IJERTV15IS061078 Page 2",
            "IJERTV15IS061078 Page 3",
        ]
    )

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == page
    assert report["footer_detected"] is False


def test_fails_open_when_footer_is_not_at_the_end() -> None:
    footer = "\n".join(footer_lines(1))
    page = "Body text.\n" + footer + "\nTrailing body line after the banner."

    cleaned, report = PageCleaner().clean(page)

    assert cleaned == page
    assert report["footer_detected"] is False


def test_empty_and_whitespace_pages_are_safe() -> None:
    cleaner = PageCleaner()

    for text in ("", "   ", "\n\n\n"):
        cleaned, report = cleaner.clean(text)
        assert cleaned == text
        assert report["footer_detected"] is False
        assert report["removal_ratio"] == 0.0


def test_non_string_text_is_rejected() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        PageCleaner().clean(None)  # type: ignore[arg-type]


def test_clean_documents_preserves_metadata_and_order() -> None:
    documents = [
        {
            "text": make_page("Page one body.", page_number=1),
            "metadata": {
                "source": "paper.pdf",
                "source_path": "C:/data/paper.pdf",
                "page_number": 1,
                "total_pages": 2,
            },
        },
        {
            "text": make_page("Page two body.", page_number=2),
            "metadata": {
                "source": "paper.pdf",
                "source_path": "C:/data/paper.pdf",
                "page_number": 2,
                "total_pages": 2,
            },
        },
    ]

    cleaned = PageCleaner().clean_documents(documents)

    assert [item["text"] for item in cleaned] == ["Page one body.", "Page two body."]
    assert [item["metadata"]["page_number"] for item in cleaned] == [1, 2]
    assert all(item["cleaning_report"]["footer_detected"] for item in cleaned)
    # The input documents are untouched.
    assert "Published by" in documents[0]["text"]


def test_module_constants_match_the_approved_rules() -> None:
    assert CLEANING_VERSION == "1"
    assert MIN_FOOTER_LINES == 3
    assert len(FOOTER_LINES) == 4
    assert PAGE_NUMBER_PATTERN.fullmatch("IJERTV15IS061078 Page 5")
    assert clean_page("Body.")[0] == "Body."
