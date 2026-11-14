"""Conservative removal of repeated journal footer lines.

Some publishers stamp the same five-line banner at the bottom of every page.
Those lines leak into chunks, dominate short windows, and are then retrieved as
if they were evidence. This module removes that banner and nothing else.

Safety rules:

* Only whole lines are removed, and only from a contiguous block at the very
  end of the page.
* A line counts as furniture only when it exactly matches a known IJERT
  signature (or the page-number pattern). Short lines, uppercase headings,
  numbered lists, citations such as ``[4][5][6]`` and table cells are never
  matched.
* At least :data:`MIN_FOOTER_LINES` qualifying lines must form the trailing
  block, and the block must contain at least one exact string (not just a page
  number).
* Text is never reflowed, dehyphenated or joined; retained lines keep their
  original order and characters.
* The cleaner fails open: when the trailing block is uncertain the page is
  returned unchanged.

Run from the project root with::

    python -m pytest tests/test_page_cleaner.py -v
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TypedDict

from ingestion.pdf_loader import Document, DocumentMetadata

#: Bumped whenever the cleaning rules change. Stored as chunk metadata so a
#: future re-index can distinguish text produced by different cleaners.
CLEANING_VERSION = "1"

#: A trailing furniture block needs at least this many qualifying lines.
MIN_FOOTER_LINES = 3

#: The four constant IJERT banner lines found on every page of the paper.
FOOTER_LINES: frozenset[str] = frozenset(
    {
        "Published by : International Journal of Engineering Research & Technology (IJERT)",
        "https://www.ijert.org/ ISSN: 2278-0181",
        "An International Peer-Reviewed Journal Vol. 15 Issue 06 , June - 2026",
        "(This work is licensed under a Creative Commons Attribution 4.0 "
        "International License.)",
    }
)

#: The page-number line differs per page, so it must be matched as a pattern.
PAGE_NUMBER_PATTERN = re.compile(r"^IJERTV15IS061078 Page \d+$")


class CleaningReport(TypedDict):
    """Structured, auditable record of what one ``clean()`` call removed.

    ``removed_char_count`` is measured as ``len(original) - len(cleaned)``, so
    it includes the newlines that separated the removed lines.
    """

    footer_detected: bool
    removed_lines: list[str]
    removed_char_count: int
    removal_ratio: float
    original_char_count: int
    cleaned_char_count: int


class CleanedDocument(TypedDict):
    """A page document plus the report describing its cleaning."""

    text: str
    metadata: DocumentMetadata
    cleaning_report: CleaningReport


def is_footer_line(line: str) -> bool:
    """Return ``True`` when a whole line is a known IJERT footer signature.

    Matching is exact after stripping surrounding whitespace. This deliberately
    does **not** treat arbitrary short, uppercase or numbered lines as
    furniture.
    """
    stripped = line.strip()
    return stripped in FOOTER_LINES or bool(PAGE_NUMBER_PATTERN.fullmatch(stripped))


def _empty_report(original_length: int) -> CleaningReport:
    """Build a report describing a no-op cleaning pass."""
    return {
        "footer_detected": False,
        "removed_lines": [],
        "removed_char_count": 0,
        "removal_ratio": 0.0,
        "original_char_count": original_length,
        "cleaned_char_count": original_length,
    }


class PageCleaner:
    """Remove verified trailing IJERT footer blocks from extracted page text.

    The class is stateless: ``clean()`` can be called on any number of pages in
    any order. It works on the text produced by ``PDFLoader`` and does not read
    the PDF itself.
    """

    def __init__(self, min_footer_lines: int = MIN_FOOTER_LINES) -> None:
        if (
            isinstance(min_footer_lines, bool)
            or not isinstance(min_footer_lines, int)
            or min_footer_lines < 1
        ):
            raise ValueError(
                "min_footer_lines must be a positive integer, "
                f"got {min_footer_lines!r}."
            )
        self.min_footer_lines = min_footer_lines

    def clean(self, text: str) -> tuple[str, CleaningReport]:
        """Return ``(cleaned_text, report)`` for one page of extracted text.

        Args:
            text: Raw page text, exactly as returned by ``PDFLoader``.

        Returns:
            The page text with a verified trailing footer block removed, plus a
            :class:`CleaningReport`. If the trailing block is uncertain, the
            original text is returned unchanged and the report says so.

        Raises:
            TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")

        lines = text.split("\n")

        # Find the last non-empty line. Trailing blank lines are ignored when
        # looking for the banner, but they are not treated as content either.
        last = len(lines) - 1
        while last >= 0 and not lines[last].strip():
            last -= 1
        if last < 0:
            return text, _empty_report(len(text))

        # Walk backwards over the contiguous block of footer lines.
        first = last
        removed: list[str] = []
        while first >= 0 and lines[first].strip() and is_footer_line(lines[first]):
            removed.append(lines[first])
            first -= 1

        # Fail open: too few lines to be the IJERT banner.
        if len(removed) < self.min_footer_lines:
            return text, _empty_report(len(text))

        # A page number alone is not enough proof; require a banner line too.
        if not any(line.strip() in FOOTER_LINES for line in removed):
            return text, _empty_report(len(text))

        removed.reverse()
        cleaned = "\n".join(lines[: first + 1])
        removed_chars = len(text) - len(cleaned)

        return cleaned, {
            "footer_detected": True,
            "removed_lines": removed,
            "removed_char_count": removed_chars,
            "removal_ratio": removed_chars / len(text) if text else 0.0,
            "original_char_count": len(text),
            "cleaned_char_count": len(cleaned),
        }

    def clean_documents(
        self, documents: Sequence[Document]
    ) -> list[CleanedDocument]:
        """Clean a list of page documents in order.

        Metadata is copied unchanged; the cleaning report is attached under
        ``cleaning_report`` so callers can audit every removal without losing
        the raw text they supplied.
        """
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError("documents must be a sequence of page dictionaries.")

        cleaned_documents: list[CleanedDocument] = []
        for index, document in enumerate(documents):
            if not isinstance(document, Mapping):
                raise TypeError(
                    f"documents[{index}] must be a dictionary, "
                    f"got {type(document).__name__}."
                )
            text = document.get("text")
            metadata = document.get("metadata")
            if not isinstance(text, str):
                raise TypeError(f"documents[{index}]['text'] must be a string.")
            if not isinstance(metadata, Mapping):
                raise TypeError(f"documents[{index}]['metadata'] must be a dictionary.")

            cleaned_text, report = self.clean(text)
            cleaned_documents.append(
                {
                    "text": cleaned_text,
                    "metadata": dict(metadata),  # type: ignore[typeddict-item]
                    "cleaning_report": report,
                }
            )

        return cleaned_documents


def clean_page(text: str) -> tuple[str, CleaningReport]:
    """Shortcut for ``PageCleaner().clean(text)``."""
    return PageCleaner().clean(text)
