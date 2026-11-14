"""Document-level body / references segmentation for cleaned pages.

The segmenter never edits text. It inspects the cleaned pages produced by
:class:`ingestion.page_cleaner.PageCleaner` and returns page-aware section
spans. Those spans are then used to label chunks with a ``section`` value, so
retrieval can prefer body evidence while the reference list stays available for
citation lookup.

Conservative detection rules:

* The heading must be a standalone, fully uppercase line such as
  ``REFERENCES`` or ``XIII. REFERENCES``. Ordinary prose mentioning references,
  table headers such as ``System / Reference``, and lowercase headings are
  rejected.
* The heading must sit in the back half of the document.
* At least three numbered entries (``[1]``, ``[2]``, ``[3]`` …) must follow in
  strictly ascending order, starting at 1.
* Roman numerals are accepted as an optional prefix but are never sufficient on
  their own.
* Once a heading is confirmed, every remaining page is classified as references.
* If no heading is confirmed, the whole document is body (fail open).

Offsets refer to the page text passed to :meth:`SectionSegmenter.segment`.
When those documents come from ``PageCleaner``, offsets refer to **cleaned**
page text, not raw ``pypdf`` output. Cleaning only removes a trailing footer, so
body offsets are unchanged; section starts after the cleaned region are also
unchanged because the footer is after them.

Run from the project root with::

    python -m pytest tests/test_section_segmenter.py -v
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from ingestion.page_cleaner import CLEANING_VERSION
from ingestion.pdf_loader import Document

BODY_SECTION = "body"
REFERENCES_SECTION = "references"

#: Minimum number of ascending numbered entries needed to confirm a heading.
MIN_REFERENCE_ENTRIES = 3

#: Optional roman prefix + a standalone uppercase back-matter heading.
HEADING_PATTERN = re.compile(
    r"^(?:[IVXLC]+\.\s*)?(?:REFERENCES?|BIBLIOGRAPHY|LITERATURE CITED|WORKS CITED)$"
)

#: Numbered reference entries, e.g. ``[13]  arXiv. Dukawalla ...``.
ENTRY_PATTERN = re.compile(r"^\s*\[(\d{1,3})\]\s+")


class SectionSpan(TypedDict):
    """A contiguous section inside one page.

    ``start`` and ``end`` are half-open character offsets into the page text
    that was passed to ``segment()`` (cleaned text when using ``PageCleaner``).
    """

    page_number: int
    section: str
    start: int
    end: int


class ReferenceEntry(TypedDict):
    """One parsed reference entry, with wrapped lines joined for lookup."""

    number: int
    text: str
    page_number: int


class SectionText(TypedDict):
    """A page slice belonging to one section."""

    page_number: int
    text: str


def _is_int(value: object) -> bool:
    """True for integers, but not booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_documents(documents: Sequence[Document]) -> None:
    """Check the page order and the fields the segmenter relies on."""
    if isinstance(documents, (str, bytes, Mapping)) or not isinstance(
        documents, Sequence
    ):
        raise TypeError("documents must be a sequence of page dictionaries.")

    previous_page: int | None = None
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

        page_number = metadata.get("page_number")
        if not _is_int(page_number) or page_number < 1:
            raise ValueError(
                f"documents[{index}]['metadata']['page_number'] must be a "
                f"positive integer, got {page_number!r}."
            )
        if previous_page is not None and page_number <= previous_page:
            raise ValueError(
                "documents must be in strictly increasing page order; "
                f"page {page_number} follows page {previous_page}."
            )
        previous_page = page_number


class SectionSegmenter:
    """Split a document into body and references without editing it."""

    def find_references_heading(
        self, documents: Sequence[Document]
    ) -> tuple[int, int] | None:
        """Return ``(page_index, char_offset)`` of a confirmed heading.

        The offset points at the first character of the heading text (after any
        indentation). Returns ``None`` when no heading can be confirmed.
        """
        _validate_documents(documents)
        if not documents:
            return None

        last_page_number = documents[-1]["metadata"]["page_number"]

        for page_index, document in enumerate(documents):
            page_number = document["metadata"]["page_number"]

            # "Near the end": back half of the document, inclusive of the
            # middle page. This rejects table headers and early mentions.
            if page_number * 2 < last_page_number:
                continue

            offset = 0
            lines = document["text"].split("\n")
            for line_index, line in enumerate(lines):
                stripped = line.strip()
                if stripped and stripped == stripped.upper():
                    if HEADING_PATTERN.fullmatch(stripped):
                        heading_offset = offset + (len(line) - len(line.lstrip()))
                        if self._has_ascending_entries(
                            documents,
                            start_page_index=page_index,
                            start_line_index=line_index + 1,
                        ):
                            return page_index, heading_offset
                offset += len(line) + 1

        return None

    def _has_ascending_entries(
        self,
        documents: Sequence[Document],
        *,
        start_page_index: int,
        start_line_index: int,
    ) -> bool:
        """Confirm the heading is followed by ``[1]``, ``[2]``, ``[3]`` …"""
        numbers: list[int] = []

        for page_index in range(start_page_index, len(documents)):
            lines = documents[page_index]["text"].split("\n")
            first_line = start_line_index if page_index == start_page_index else 0
            for line in lines[first_line:]:
                match = ENTRY_PATTERN.match(line)
                if not match:
                    continue

                numbers.append(int(match.group(1)))
                if len(numbers) >= MIN_REFERENCE_ENTRIES:
                    first = numbers[0]
                    strictly_ascending = all(
                        later > earlier
                        for earlier, later in zip(numbers, numbers[1:])
                    )
                    return first == 1 and strictly_ascending

        return False

    def segment(self, documents: Sequence[Document]) -> list[SectionSpan]:
        """Return section spans for every page, in page order.

        Each non-empty page receives at least one span. The heading page is
        split at the heading offset; later pages become entirely references.
        """
        _validate_documents(documents)
        heading = self.find_references_heading(documents)

        spans: list[SectionSpan] = []
        for page_index, document in enumerate(documents):
            text = document["text"]
            page_number = document["metadata"]["page_number"]
            length = len(text)

            if heading is None or page_index < heading[0]:
                spans.append(
                    {
                        "page_number": page_number,
                        "section": BODY_SECTION,
                        "start": 0,
                        "end": length,
                    }
                )
                continue

            if page_index == heading[0]:
                heading_offset = heading[1]
                if heading_offset > 0:
                    spans.append(
                        {
                            "page_number": page_number,
                            "section": BODY_SECTION,
                            "start": 0,
                            "end": heading_offset,
                        }
                    )
                spans.append(
                    {
                        "page_number": page_number,
                        "section": REFERENCES_SECTION,
                        "start": heading_offset,
                        "end": length,
                    }
                )
                continue

            spans.append(
                {
                    "page_number": page_number,
                    "section": REFERENCES_SECTION,
                    "start": 0,
                    "end": length,
                }
            )

        return spans

    def assign_to_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        spans: Sequence[SectionSpan],
        *,
        cleaning_version: str = CLEANING_VERSION,
    ) -> list[dict[str, Any]]:
        """Return copies of ``chunks`` with ``section`` metadata added.

        A chunk is labelled by the section it overlaps most, using its
        ``page_number``, ``char_start`` and ``char_end``. This means a chunk that
        starts just before the heading but is mostly reference text is labelled
        ``references``. Input chunks are never mutated.

        The offsets must belong to the same page text the spans describe
        (cleaned text when the spans came from cleaned documents).
        """
        if isinstance(chunks, (str, bytes, Mapping)) or not isinstance(
            chunks, Sequence
        ):
            raise TypeError("chunks must be a sequence of chunk dictionaries.")

        spans_by_page: dict[int, list[SectionSpan]] = {}
        for span in spans:
            spans_by_page.setdefault(span["page_number"], []).append(span)

        labelled: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping):
                raise TypeError(
                    f"chunks[{index}] must be a dictionary, "
                    f"got {type(chunk).__name__}."
                )
            metadata = chunk.get("metadata")
            if not isinstance(metadata, Mapping):
                raise TypeError(f"chunks[{index}]['metadata'] must be a dictionary.")

            page_number = metadata.get("page_number")
            char_start = metadata.get("char_start")
            char_end = metadata.get("char_end")
            if not _is_int(page_number) or not _is_int(char_start) or not _is_int(char_end):
                raise ValueError(
                    f"chunks[{index}] must have integer page_number, "
                    "char_start and char_end metadata."
                )
            if char_start < 0 or char_end < char_start:
                raise ValueError(f"chunks[{index}] has invalid character offsets.")

            page_spans = spans_by_page.get(page_number)
            if not page_spans:
                raise ValueError(
                    f"chunks[{index}] refers to page {page_number}, "
                    "which has no section spans."
                )

            section = self._section_for(
                page_spans,
                char_start=char_start,
                char_end=char_end,
            )

            new_metadata = dict(metadata)
            new_metadata["section"] = section
            new_metadata["cleaning_version"] = cleaning_version
            labelled.append({**chunk, "metadata": new_metadata})

        return labelled

    @staticmethod
    def _section_for(
        page_spans: Sequence[SectionSpan],
        *,
        char_start: int,
        char_end: int,
    ) -> str:
        """Pick the section with the largest overlap, preferring body on ties."""
        page_length = max(span["end"] for span in page_spans)
        start = max(0, min(char_start, page_length))
        end = max(start, min(char_end, page_length))

        best_section = page_spans[0]["section"]
        best_overlap = -1
        for span in page_spans:
            overlap = max(0, min(end, span["end"]) - max(start, span["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_section = span["section"]

        if best_overlap > 0:
            return best_section

        # No measurable overlap (for example an empty page): fall back to the
        # span containing char_start, then to body when one exists.
        for span in page_spans:
            if span["start"] <= char_start <= span["end"]:
                return span["section"]
        return BODY_SECTION

    def section_text(
        self,
        documents: Sequence[Document],
        spans: Sequence[SectionSpan],
        section: str,
    ) -> list[SectionText]:
        """Return the text slices belonging to ``section``, page by page.

        Reference text is retained (never deleted), so this is the entry point
        for future citation lookup.
        """
        if section not in (BODY_SECTION, REFERENCES_SECTION):
            raise ValueError(
                f"section must be {BODY_SECTION!r} or {REFERENCES_SECTION!r}, "
                f"got {section!r}."
            )
        _validate_documents(documents)

        page_text = {
            document["metadata"]["page_number"]: document["text"]
            for document in documents
        }

        slices: list[SectionText] = []
        for span in spans:
            if span["section"] != section:
                continue
            text = page_text[span["page_number"]][span["start"] : span["end"]]
            if text:
                slices.append({"page_number": span["page_number"], "text": text})
        return slices

    def reference_entries(
        self,
        documents: Sequence[Document],
        spans: Sequence[SectionSpan],
    ) -> list[ReferenceEntry]:
        """Parse numbered reference entries from the reference spans.

        Wrapped lines (``Available:``, URLs, journal names) are appended to the
        entry that precedes them. Fragments are joined with a single space and
        are **not** dehyphenated, so the stored text still matches the paper.
        """
        _validate_documents(documents)

        page_text = {
            document["metadata"]["page_number"]: document["text"]
            for document in documents
        }

        reference_spans = sorted(
            (span for span in spans if span["section"] == REFERENCES_SECTION),
            key=lambda span: (span["page_number"], span["start"]),
        )

        entries: list[ReferenceEntry] = []
        current_number: int | None = None
        current_page: int | None = None
        parts: list[str] = []

        def flush() -> None:
            if current_number is None or current_page is None:
                return
            text = " ".join(" ".join(parts).split())
            entries.append(
                {
                    "number": current_number,
                    "text": text,
                    "page_number": current_page,
                }
            )

        for span in reference_spans:
            text = page_text[span["page_number"]][span["start"] : span["end"]]
            for line in text.split("\n"):
                match = ENTRY_PATTERN.match(line)
                if match:
                    flush()
                    current_number = int(match.group(1))
                    current_page = span["page_number"]
                    parts = [line[match.end() :]]
                elif current_number is not None and line.strip():
                    parts.append(line)

        flush()
        return entries
