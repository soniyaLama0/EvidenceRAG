"""Structure-aware chunking for cleaned research-paper pages.

``StructureAwareChunker`` is an experimental alternative to
:class:`ingestion.text_chunker.TextChunker`. It implements the same
``chunk_documents(documents) -> list[Chunk]`` protocol, so it can be injected
into :class:`ingestion.pipeline.IngestionPipeline` without changing the
pipeline or the default chunker.

Design rules
------------
* Every chunk is a **verbatim slice of one cleaned page**. The invariant
  ``page_text[char_start:char_end] == chunk["text"]`` always holds. Line
  reflowing and sentence detection use an internal analysis view only; source
  offsets are always mapped back to the original page text.
* Chunks never cross a body/references boundary. The boundary comes from the
  existing :class:`ingestion.section_segmenter.SectionSegmenter`. If that
  segmenter cannot confirm a references section, the whole page stays body
  (fail open).
* Prose is packed on sentence boundaries with a token budget. Headings are
  attached to the first content chunk and inherited through ``heading_path``
  metadata.
* Tables are detected conservatively: a caption is required, rows must form a
  short-line run, and caption/row segments that are not adjacent are linked
  with a stable ``table_id`` instead of being merged across unrelated text.
* Detection is advisory. Any block that cannot be classified is kept verbatim
  as prose, and a word/character fallback is used when sentence segmentation
  is impossible. Text is never rewritten or dropped.

Token budget
------------
``token_counter`` must return the token count **including** the model's
special tokens (``[CLS]``/``[SEP]``). Pass a single-text wrapper around the
embedding model's tokenizer, for example::

    counter = lambda text: model._count_tokens([text])[0]

When no counter is supplied, a deliberately conservative word/character
heuristic is used and two special tokens are reserved. The heuristic
over-estimates rather than under-estimates, but exact enforcement of a model
limit requires the real tokenizer.

The chunker does not add embeddings, call retrieval, or write to Chroma. It
only produces ``{"text", "metadata"}`` dictionaries.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from .pdf_loader import Document, DocumentMetadata
from .section_segmenter import (
    BODY_SECTION,
    REFERENCES_SECTION,
    ENTRY_PATTERN,
    SectionSegmenter,
    SectionSpan,
)

#: Distinguishes these records from the fixed-window chunkers.
CHUNKING_VERSION = "structure-1"

DEFAULT_TARGET_TOKENS = 190
DEFAULT_HARD_MAX_TOKENS = 254
DEFAULT_OVERLAP_SENTENCES = 1
DEFAULT_MAX_OVERLAP_CHARS = 200
DEFAULT_MAX_OVERLAP_RATIO = 0.25
DEFAULT_HEADING_MAX_CHARS = 90
DEFAULT_TABLE_ORPHAN_SEARCH_CHARS = 1200
DEFAULT_TABLE_JOIN_GAP_CHARS = 120
DEFAULT_TABLE_MIN_ROWS = 3
DEFAULT_TABLE_ROW_MAX_CHARS = 120
DEFAULT_FALLBACK_CHARS = 800

#: Used only when no tokenizer is injected; always reserves two special tokens.
_HEURISTIC_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_CORE_RE = re.compile(r"^\W+|\W+$")

_TABLE_CAPTION_RE = re.compile(
    r"^(?:TABLE|Table|FIGURE|Figure|Fig\.?|FIG\.?)\s+"
    r"(?:[IVXLC]+|\d{1,3})\b"
)
_ROMAN_RE = re.compile(r"^(?P<num>[IVXLCDM]+)\.\s*(?P<title>\S.*)$")
_NUMERIC_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)\.?\s+(?P<title>\S.*)$"
)
_ALPHA_RE = re.compile(r"^(?P<num>[A-Z])\.\s*(?P<title>\S.*)$")
_VALID_ROMAN_RE = re.compile(
    r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\(\d{1,2}\)|\d{1,2}[.)]|[-•*▪◦])\s+\S"
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
_SENTENCE_START_RE = re.compile(r"[.!?]\s+")

_NON_HEADING_PREFIXES = (
    "keywords:",
    "key words:",
    "index terms:",
    "e-mail:",
    "email:",
    "doi:",
    "issn:",
    "received:",
    "accepted:",
    "published:",
    "copyright",
)

_ABBREVIATIONS = {
    "al",
    "approx",
    "cf",
    "ch",
    "dr",
    "e.g",
    "eg",
    "eq",
    "etc",
    "fig",
    "figs",
    "i.e",
    "ie",
    "mr",
    "mrs",
    "ms",
    "no",
    "pp",
    "prof",
    "ref",
    "refs",
    "resp",
    "sec",
    "st",
    "tab",
    "vol",
    "vs",
}


class ChunkMetadata(DocumentMetadata):
    """Page metadata plus structure-aware chunk metadata."""

    chunk_index: int
    char_start: int
    char_end: int
    section: str
    chunking_version: str
    structure_type: str
    structure_confidence: str
    heading_path: str
    heading_level: int
    heading_text: str
    table_id: str
    table_caption: str
    table_part: int
    block_index: int
    starts_mid_sentence: bool
    ends_mid_sentence: bool
    is_forced_split: bool
    overlap_sentences: int


class Chunk(TypedDict):
    """A verbatim page slice plus scalar metadata."""

    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    start: int
    end: int
    raw: str

    @property
    def stripped(self) -> str:
        return self.raw.strip()


@dataclass
class _Heading:
    text: str
    level: int
    start: int
    end: int
    block_index: int
    path: str


@dataclass
class _Unit:
    start: int
    end: int
    text: str
    kind: str  # sentence | list | reference | formula | table_row | fallback
    block_index: int
    forced: bool = False
    single_sentence: bool = True


@dataclass
class _Draft:
    start: int
    end: int
    structure_type: str
    block_index: int
    section: str
    confidence: str
    section_start: int
    heading_path: str = ""
    heading_level: int = 0
    heading_text: str = ""
    table_id: str = ""
    table_caption: str = ""
    table_part: int = 0
    forced: bool = False
    overlap: int = 0
    has_heading: bool = False
    starts_forced: bool = False


@dataclass
class _TablePlan:
    table_id: str
    caption: str
    caption_start: int
    caption_end: int
    caption_block: int
    caption_lines: tuple[int, int]
    rows_start: int
    rows_end: int
    rows_block: int
    same_block: bool


# ---------------------------------------------------------------------------
# Token counting and text helpers
# ---------------------------------------------------------------------------


def heuristic_token_count(text: str) -> int:
    """Conservative token estimate that reserves two special tokens.

    Words are charged ``ceil(len(core) / 4)`` tokens (long identifiers are
    split by BPE) plus one token per punctuation character. Two special tokens
    are added so the result can be compared directly with the 256-token model
    limit.
    """
    count = 2
    for match in _HEURISTIC_TOKEN_RE.finditer(text):
        token = match.group(0)
        if token[0].isalnum() or token[0] == "_":
            count += max(1, math.ceil(len(token) / 4))
        else:
            count += 1
    return count


def _trim(page_text: str, start: int, end: int) -> tuple[int, int]:
    """Trim whitespace from a span without leaving the page bounds."""
    start = max(0, start)
    end = min(len(page_text), end)
    while start < end and page_text[start].isspace():
        start += 1
    while end > start and page_text[end - 1].isspace():
        end -= 1
    return start, end


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Reflow with an explicit normalised -> original offset map
# ---------------------------------------------------------------------------


class _Reflow:
    """Join hard-wrapped lines for analysis while retaining source offsets."""

    def __init__(self, lines: Sequence[_Line]) -> None:
        self.text = ""
        # (normalised_start, normalised_end, original_start, original_end)
        self.spans: list[tuple[int, int, int, int]] = []
        for line in lines:
            raw = line.raw
            stripped = raw.strip()
            if not stripped:
                continue
            leading = len(raw) - len(raw.lstrip())
            original_start = line.start + leading
            if self.text:
                self.text += " "
            normalised_start = len(self.text)
            self.text += stripped
            self.spans.append(
                (
                    normalised_start,
                    len(self.text),
                    original_start,
                    original_start + len(stripped),
                )
            )

    def original_start(self, position: int) -> int:
        """Map a normalised start position to an original offset."""
        if not self.spans:
            return 0
        previous_end = self.spans[0][2]
        for norm_start, norm_end, original_start, original_end in self.spans:
            if position < norm_start:
                return original_start
            if position < norm_end:
                return original_start + (position - norm_start)
            previous_end = original_end
        return previous_end

    def original_end(self, position: int) -> int:
        """Map a normalised exclusive end position to an original offset."""
        if not self.spans:
            return 0
        previous_end = self.spans[0][2]
        for norm_start, norm_end, original_start, original_end in self.spans:
            if position <= norm_start:
                return previous_end
            if position <= norm_end:
                return original_start + (position - norm_start)
            previous_end = original_end
        return self.spans[-1][3]


def _split_lines(text: str, base_start: int) -> list[_Line]:
    """Split a page/section slice into lines with absolute offsets."""
    lines: list[_Line] = []
    offset = base_start
    for raw in text.split("\n"):
        lines.append(_Line(offset, offset + len(raw), raw))
        offset += len(raw) + 1
    return lines
# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


def _token_before(text: str, position: int) -> str:
    """Return the whitespace-delimited token ending at ``position`` (exclusive)."""
    start = position
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start:position]


def _is_url_token(token: str) -> bool:
    return bool(_URL_RE.search(token) or _EMAIL_RE.search(token))


def _is_sentence_boundary(text: str, punctuation_index: int) -> bool:
    """Decide whether ``text[punctuation_index]`` really ends a sentence."""
    char = text[punctuation_index]
    if char not in ".!?":
        return False

    next_index = punctuation_index + 1
    while next_index < len(text) and text[next_index] == " ":
        next_index += 1
    if next_index >= len(text):
        return True
    following = text[next_index]
    if not (following.isupper() or following.isdigit() or following in "([\"'"):
        return False

    # Decimals and version numbers: 3.14, v2.1
    if char == ".":
        previous_char = text[punctuation_index - 1] if punctuation_index else ""
        if previous_char.isdigit() and following.isdigit():
            return False

    token = _token_before(text, punctuation_index)
    cleaned = token.strip("()[]{}\"'").lower()
    if _is_url_token(token):
        return False
    if cleaned in _ABBREVIATIONS:
        return False
    # Single initials and "K.J." chains.
    if re.fullmatch(r"[A-Za-z]", cleaned):
        return False
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", cleaned):
        return False
    return True


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` sentence spans in ``text`` coordinates.

    The splitter protects abbreviations, initials, decimals, citations,
    e-mail addresses and URLs. When no reliable boundary exists the whole text
    is returned as one span; callers then apply the word/character fallback.
    """
    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]", text):
        index = match.start()
        if not _is_sentence_boundary(text, index):
            continue
        end = index + 1
        while end < len(text) and text[end] in "\"'":
            end += 1
        spans.append((start, end))
        start = end
        while start < len(text) and text[start] == " ":
            start += 1

    if start < len(text):
        spans.append((start, len(text)))

    cleaned: list[tuple[int, int]] = []
    for sentence_start, sentence_end in spans:
        while sentence_start < sentence_end and text[sentence_start].isspace():
            sentence_start += 1
        while sentence_end > sentence_start and text[sentence_end - 1].isspace():
            sentence_end -= 1
        if sentence_start < sentence_end:
            cleaned.append((sentence_start, sentence_end))
    return cleaned


# ---------------------------------------------------------------------------
# Heading and line classification
# ---------------------------------------------------------------------------


def _heading_title_is_plausible(title: str, max_chars: int) -> bool:
    title = title.strip()
    if not title or len(title) > max_chars:
        return False
    if title.endswith((".", ",", ";")):
        return False
    if len(title.split()) > 14:
        return False
    if _URL_RE.search(title) or _EMAIL_RE.search(title):
        return False
    if title[0].islower():
        return False
    # Reject "K.J. Somaiya Institute of Technology" style abbreviation chains.
    if re.match(r"^[A-Z]\.(?:\s|$)", title):
        return False
    if re.match(r"^[A-Z]\.[A-Z]\.", title):
        return False
    return True


def detect_heading(
    line: str,
    *,
    previous_line: str = "",
    next_line: str = "",
    max_chars: int = DEFAULT_HEADING_MAX_CHARS,
) -> tuple[int, str] | None:
    """Return ``(level, heading_text)`` for a conservative heading candidate.

    Only numbered lines, all-caps standalone lines and their combinations are
    accepted. Unnumbered title-case lines are treated as prose because they are
    too easy to confuse with table headers, affiliations and captions.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > max_chars:
        return None
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _NON_HEADING_PREFIXES):
        return None
    if _is_table_caption(stripped):
        return None
    if ENTRY_PATTERN.match(stripped) or _LIST_ITEM_RE.match(stripped):
        return None
    if _URL_RE.search(stripped) or _EMAIL_RE.search(stripped):
        return None
    if stripped[-1] in ",;":
        return None

    roman = _ROMAN_RE.match(stripped)
    if roman and _VALID_ROMAN_RE.fullmatch(roman.group("num")):
        if _heading_title_is_plausible(roman.group("title"), max_chars):
            return 1, stripped

    numeric = _NUMERIC_RE.match(stripped)
    if numeric:
        depth = numeric.group("num").count(".") + 1
        if _heading_title_is_plausible(numeric.group("title"), max_chars):
            return min(depth, 3), stripped

    alpha = _ALPHA_RE.match(stripped)
    if alpha:
        if _heading_title_is_plausible(alpha.group("title"), max_chars):
            return 2, stripped

    letters = [char for char in stripped if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        if len(stripped) >= 8 or len(stripped.split()) >= 2:
            return 1, stripped

    return None


def _is_table_caption(line: str) -> bool:
    """Conservative table/figure caption test.

    The marker must be followed by a number or roman numeral, the line must be
    caption-shaped (not a prose sentence), and a lowercase continuation such as
    "Table I presents ..." is rejected.
    """
    match = _TABLE_CAPTION_RE.match(line)
    if not match:
        return False
    stripped = line.strip()
    if len(stripped) > 160:
        return False
    rest = stripped[match.end() :].lstrip()
    if rest:
        if rest[0].islower():
            return False
        if rest[0] in ".:,;":
            remainder = rest[1:].lstrip()
            if remainder and remainder[0].islower():
                return False
    if stripped.endswith(".") and len(stripped.split()) >= 8:
        return False
    return True


def _looks_like_formula(line: str) -> bool:
    """True for symbol-heavy lines that should never be sentence-split."""
    if len(line) < 3:
        return False
    if re.search(r"\[\d+\]", line):
        return False
    letters = sum(char.isalpha() for char in line)
    symbols = sum(not char.isalnum() and not char.isspace() for char in line)
    operators = sum(
        char in "=+*/^<>≤≥≠∑∫√" for char in line
    )
    if letters < 1 or symbols < 3 or symbols < letters / 2:
        return False
    words = line.split()
    has_lower_word = any(
        any(char.islower() for char in word) for word in words
    )
    if line.rstrip().endswith(".") and has_lower_word:
        return False
    return operators >= 1 or symbols >= 6


def _is_caption_continuation(line: _Line, next_gap: int) -> bool:
    stripped = line.stripped
    if not stripped or next_gap > 2:
        return False
    if len(stripped) > 120 or len(stripped.split()) > 14:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    if _is_table_caption(stripped):
        return False
    if ENTRY_PATTERN.match(stripped) or _LIST_ITEM_RE.match(stripped):
        return False
    return True


def _is_row_candidate(
    line: _Line,
    *,
    previous_line: str = "",
    next_line: str = "",
    row_max_chars: int = DEFAULT_TABLE_ROW_MAX_CHARS,
) -> bool:
    """True for one short line that may belong to a table row run."""
    stripped = line.stripped
    if not stripped or len(stripped) > row_max_chars:
        return False
    if _is_table_caption(stripped):
        return False
    if detect_heading(stripped, previous_line=previous_line, next_line=next_line):
        return False
    if ENTRY_PATTERN.match(stripped) or _LIST_ITEM_RE.match(stripped):
        return False
    if stripped.endswith(":"):
        return False
    if stripped.endswith(".") and len(stripped.split()) >= 6:
        return False
    return True


def _row_run_strength(lines: Sequence[_Line], min_rows: int) -> bool:
    if len(lines) < min_rows:
        return False
    lengths = [len(line.stripped) for line in lines]
    mean_length = sum(lengths) / len(lengths)
    if mean_length > 45:
        return False
    short_ratio = sum(1 for length in lengths if length <= 40) / len(lengths)
    if short_ratio < 0.5:
        return False
    terminal = sum(
        1 for line in lines if line.stripped.endswith((".", "!", "?"))
    )
    if terminal / len(lines) > 0.4:
        return False
    return True
# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class StructureAwareChunker:
    """Structure-aware experimental chunker.

    Args:
        token_counter: ``Callable[[str], int]`` returning the token count
            *including* special tokens. When omitted, a conservative heuristic
            is used.
        target_tokens: Soft budget; chunks close after crossing it.
        hard_max_tokens: Hard ceiling, including special tokens.
        overlap_sentences: Maximum carried sentences (0 or 1).
        max_overlap_chars: Character cap for one carried sentence.
        max_overlap_ratio: A carried sentence may not exceed this fraction of
            the previous chunk's characters.
        heading_max_chars: Longest accepted heading line.
        table_orphan_search_chars: Forward caption-to-row search distance.
        table_join_gap_chars: Maximum gap for joining caption and rows.
        table_min_rows: Minimum short-line run length for a table.
        table_row_max_chars: Longest line accepted as a table row fragment.
        fallback_chars: Approximate character window for the last-resort
            fallback when no sentence boundary exists.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int] | None = None,
        *,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        max_overlap_chars: int = DEFAULT_MAX_OVERLAP_CHARS,
        max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
        heading_max_chars: int = DEFAULT_HEADING_MAX_CHARS,
        table_orphan_search_chars: int = DEFAULT_TABLE_ORPHAN_SEARCH_CHARS,
        table_join_gap_chars: int = DEFAULT_TABLE_JOIN_GAP_CHARS,
        table_min_rows: int = DEFAULT_TABLE_MIN_ROWS,
        table_row_max_chars: int = DEFAULT_TABLE_ROW_MAX_CHARS,
        fallback_chars: int = DEFAULT_FALLBACK_CHARS,
        segmenter: SectionSegmenter | None = None,
    ) -> None:
        if not isinstance(target_tokens, int) or target_tokens <= 0:
            raise ValueError("target_tokens must be a positive integer.")
        if not isinstance(hard_max_tokens, int) or hard_max_tokens <= 0:
            raise ValueError("hard_max_tokens must be a positive integer.")
        if target_tokens > hard_max_tokens:
            raise ValueError("target_tokens must not exceed hard_max_tokens.")
        if overlap_sentences not in (0, 1):
            raise ValueError("overlap_sentences must be 0 or 1.")
        if max_overlap_chars < 0:
            raise ValueError("max_overlap_chars must be non-negative.")
        if not 0 <= max_overlap_ratio <= 1:
            raise ValueError("max_overlap_ratio must be between 0 and 1.")
        if table_min_rows < 1:
            raise ValueError("table_min_rows must be at least 1.")

        self.target_tokens = target_tokens
        self.hard_max_tokens = hard_max_tokens
        self.overlap_sentences = overlap_sentences
        self.max_overlap_chars = max_overlap_chars
        self.max_overlap_ratio = max_overlap_ratio
        self.heading_max_chars = heading_max_chars
        self.table_orphan_search_chars = table_orphan_search_chars
        self.table_join_gap_chars = table_join_gap_chars
        self.table_min_rows = table_min_rows
        self.table_row_max_chars = table_row_max_chars
        self.fallback_chars = fallback_chars
        self._custom_counter = token_counter
        self._segmenter = segmenter if segmenter is not None else SectionSegmenter()

    # -- public API --------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens with the injected counter or the safe heuristic."""
        if self._custom_counter is None:
            return heuristic_token_count(text)
        value = self._custom_counter(text)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                "token_counter must return a non-negative integer, "
                f"got {value!r}."
            )
        return value

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """Chunk cleaned page documents.

        The output shape matches ``TextChunker.chunk_documents``: page
        metadata is preserved, ``chunk_index`` is sequential across the whole
        call, and ``char_start``/``char_end`` refer to the page text passed in.
        """
        pages = self._validate_documents(documents)
        spans = self._segmenter.segment(pages)  # type: ignore[arg-type]
        spans_by_page: dict[int, list[SectionSpan]] = {}
        for span in spans:
            if span["end"] > span["start"]:
                spans_by_page.setdefault(span["page_number"], []).append(span)

        chunks: list[Chunk] = []
        chunk_index = 0
        previous_page_text = ""
        for document in pages:
            page_number = document["metadata"]["page_number"]
            page_text = document["text"]
            if not page_text.strip():
                previous_page_text = ""
                continue
            page_spans = spans_by_page.get(page_number)
            if not page_spans:
                page_spans = [
                    {
                        "page_number": page_number,
                        "section": BODY_SECTION,
                        "start": 0,
                        "end": len(page_text),
                    }
                ]
            drafts = self._chunk_page(document, page_spans)
            for draft in drafts:
                start, end = _trim(page_text, draft.start, draft.end)
                if start >= end:
                    continue
                text = page_text[start:end]
                if self.count_tokens(text) > self.hard_max_tokens:
                    # Should not happen; guard preserves the hard limit.
                    for piece in self._fallback_units(
                        page_text, start, end, draft.block_index
                    ):
                        piece_start, piece_end = _trim(
                            page_text, piece.start, piece.end
                        )
                        if piece_start < piece_end:
                            chunks.append(
                                self._make_chunk(
                                    document,
                                    page_text,
                                    piece_start,
                                    piece_end,
                                    draft,
                                    chunk_index,
                                    forced=True,
                                    previous_page_text=previous_page_text,
                                )
                            )
                            chunk_index += 1
                    continue
                chunks.append(
                    self._make_chunk(
                        document,
                        page_text,
                        start,
                        end,
                        draft,
                        chunk_index,
                        previous_page_text=previous_page_text,
                    )
                )
                chunk_index += 1

            previous_page_text = page_text

        return chunks

    # -- validation --------------------------------------------------------

    @staticmethod
    def _validate_documents(
        documents: Iterable[Document],
    ) -> list[Document]:
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError("documents must be an iterable of page dictionaries.")
        materialised = list(documents)
        ordered: list[Document] = []
        previous_page: int | None = None
        for index, document in enumerate(materialised):
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
                raise TypeError(
                    f"documents[{index}]['metadata'] must be a dictionary."
                )
            page_number = metadata.get("page_number")
            if not isinstance(page_number, int) or isinstance(page_number, bool):
                raise ValueError(
                    f"documents[{index}]['metadata']['page_number'] must be "
                    f"an integer, got {page_number!r}."
                )
            if page_number < 1:
                raise ValueError("page_number must be positive.")
            if previous_page is not None and page_number <= previous_page:
                raise ValueError(
                    "documents must be in strictly increasing page order; "
                    f"page {page_number} follows page {previous_page}."
                )
            previous_page = page_number
            ordered.append(document)  # type: ignore[arg-type]
        return ordered
    # -- page and section orchestration ------------------------------------

    def _chunk_page(
        self, document: Document, spans: Sequence[SectionSpan]
    ) -> list[_Draft]:
        page_text = document["text"]
        page_number = document["metadata"]["page_number"]
        drafts: list[_Draft] = []
        block_counter = 0
        flags = {"heading": False, "table": False, "fallback": False}
        table_state = {"count": 0}

        for span in sorted(spans, key=lambda item: item["start"]):
            section = span["section"]
            section_start = span["start"]
            section_end = span["end"]
            lines = _split_lines(
                page_text[section_start:section_end], section_start
            )
            section_drafts, block_counter = self._process_section(
                lines,
                page_text,
                page_number=page_number,
                section=section,
                section_start=section_start,
                section_end=section_end,
                block_index_start=block_counter,
                flags=flags,
                table_state=table_state,
            )
            drafts.extend(section_drafts)

        if flags["fallback"]:
            confidence = "none"
        elif flags["heading"] or flags["table"]:
            confidence = "high"
        else:
            confidence = "medium"
        for draft in drafts:
            draft.confidence = confidence
        return drafts

    def _process_section(
        self,
        lines: Sequence[_Line],
        page_text: str,
        *,
        page_number: int,
        section: str,
        section_start: int,
        section_end: int,
        block_index_start: int,
        flags: dict[str, bool],
        table_state: dict[str, int],
    ) -> tuple[list[_Draft], int]:
        drafts: list[_Draft] = []
        block_counter = block_index_start
        pending: list[_Heading] = []
        stack: list[_Heading] = []
        units: list[_Unit] = []
        mode: str | None = None
        row_plans: dict[int, _TablePlan] = {}

        def consume_pending() -> list[_Heading]:
            nonlocal pending
            taken = pending[:]
            pending.clear()
            return taken

        def flush_units() -> None:
            nonlocal units, mode
            if not units:
                return
            taken = consume_pending()
            drafts.extend(
                self._emit_units(
                    units,
                    page_text=page_text,
                    section=section,
                    section_start=section_start,
                    pending=taken,
                    heading=stack[-1] if stack else None,
                    allow_overlap=(mode == "sentence"),
                )
            )
            units = []
            mode = None

        def add_units(new_units: Sequence[_Unit], new_mode: str) -> None:
            nonlocal mode
            if not new_units:
                return
            if mode is not None and mode != new_mode:
                flush_units()
            mode = new_mode
            units.extend(new_units)

        index = 0
        total = len(lines)
        while index < total:
            line = lines[index]
            stripped = line.stripped
            if not stripped:
                index += 1
                continue

            if index in row_plans:
                flush_units()
                plan = row_plans.pop(index)
                taken = consume_pending()
                drafts.extend(
                    self._emit_table_lines(
                        lines[plan.rows_start : plan.rows_end + 1],
                        page_text=page_text,
                        section=section,
                        section_start=section_start,
                        plan=plan,
                        start_part=2 if not plan.same_block else 1,
                        pending=taken,
                        heading=stack[-1] if stack else None,
                    )
                )
                index = plan.rows_end + 1
                block_counter = max(block_counter, plan.rows_block + 1)
                continue

            if _is_table_caption(stripped):
                flush_units()
                plan = self._build_table_plan(
                    lines,
                    index,
                    page_text,
                    page_number=page_number,
                    ordinal=table_state["count"],
                    block_index=block_counter,
                )
                flags["table"] = True
                taken = consume_pending()
                if plan is None:
                    drafts.extend(
                        self._emit_table_lines(
                            [lines[index]],
                            page_text=page_text,
                            section=section,
                            section_start=section_start,
                            plan=_TablePlan(
                                table_id=f"p{page_number}-t{table_state['count'] + 1}",
                                caption=stripped,
                                caption_start=line.start,
                                caption_end=line.end,
                                caption_block=block_counter,
                                caption_lines=(index, index),
                                rows_start=index + 1,
                                rows_end=index,
                                rows_block=block_counter,
                                same_block=True,
                            ),
                            start_part=1,
                            pending=taken,
                            heading=stack[-1] if stack else None,
                        )
                    )
                    index += 1
                    block_counter += 1
                    continue

                table_state["count"] += 1
                if plan.same_block:
                    drafts.extend(
                        self._emit_table_lines(
                            lines[plan.caption_lines[0] : plan.rows_end + 1],
                            page_text=page_text,
                            section=section,
                            section_start=section_start,
                            plan=plan,
                            start_part=1,
                            pending=taken,
                            heading=stack[-1] if stack else None,
                        )
                    )
                    index = plan.rows_end + 1
                    block_counter = max(block_counter, plan.rows_block + 1)
                else:
                    drafts.extend(
                        self._emit_table_lines(
                            lines[plan.caption_lines[0] : plan.caption_lines[1] + 1],
                            page_text=page_text,
                            section=section,
                            section_start=section_start,
                            plan=plan,
                            start_part=1,
                            pending=taken,
                            heading=stack[-1] if stack else None,
                        )
                    )
                    row_plans[plan.rows_start] = plan
                    index = plan.caption_lines[1] + 1
                    block_counter = max(block_counter, plan.caption_block + 1)
                continue

            heading = detect_heading(
                stripped,
                previous_line=lines[index - 1].raw if index else "",
                next_line=(
                    lines[index + 1].raw if index + 1 < total else ""
                ),
                max_chars=self.heading_max_chars,
            )
            if heading is not None:
                flush_units()
                level, heading_text = heading
                if level == 2 and not any(item.level == 1 for item in stack):
                    level = 1
                while stack and stack[-1].level >= level:
                    stack.pop()
                path = " > ".join([item.text for item in stack] + [heading_text])
                record = _Heading(
                    text=heading_text,
                    level=level,
                    start=line.start,
                    end=line.end,
                    block_index=block_counter,
                    path=path,
                )
                stack.append(record)
                pending.append(record)
                flags["heading"] = True
                block_counter += 1
                index += 1
                continue

            if section == REFERENCES_SECTION and ENTRY_PATTERN.match(stripped):
                flush_units()
                entry_lines = [line]
                cursor = index + 1
                while cursor < total:
                    candidate = lines[cursor]
                    candidate_text = candidate.stripped
                    if not candidate_text:
                        cursor += 1
                        continue
                    if ENTRY_PATTERN.match(candidate_text):
                        break
                    if detect_heading(
                        candidate_text,
                        previous_line=lines[cursor - 1].raw if cursor else "",
                        next_line=(
                            lines[cursor + 1].raw
                            if cursor + 1 < total
                            else ""
                        ),
                        max_chars=self.heading_max_chars,
                    ):
                        break
                    entry_lines.append(candidate)
                    cursor += 1
                entry_start = entry_lines[0].start
                entry_end = entry_lines[-1].end
                unit = _Unit(
                    entry_start,
                    entry_end,
                    page_text[entry_start:entry_end],
                    "reference",
                    block_counter,
                    single_sentence=False,
                )
                block_counter += 1
                drafts.extend(
                    self._emit_units(
                        [unit],
                        page_text=page_text,
                        section=section,
                        section_start=section_start,
                        pending=consume_pending(),
                        heading=stack[-1] if stack else None,
                        allow_overlap=False,
                    )
                )
                index = cursor
                continue

            if _LIST_ITEM_RE.match(stripped):
                item_lines = [line]
                cursor = index + 1
                while cursor < total:
                    candidate = lines[cursor]
                    candidate_text = candidate.stripped
                    if not candidate_text:
                        break
                    if _LIST_ITEM_RE.match(candidate_text):
                        break
                    if _is_table_caption(candidate_text):
                        break
                    if detect_heading(
                        candidate_text,
                        previous_line=lines[cursor - 1].raw if cursor else "",
                        next_line=(
                            lines[cursor + 1].raw
                            if cursor + 1 < total
                            else ""
                        ),
                        max_chars=self.heading_max_chars,
                    ):
                        break
                    item_lines.append(candidate)
                    cursor += 1
                item_start = item_lines[0].start
                item_end = item_lines[-1].end
                add_units(
                    [
                        _Unit(
                            item_start,
                            item_end,
                            page_text[item_start:item_end],
                            "list",
                            block_counter,
                            single_sentence=False,
                        )
                    ],
                    "list",
                )
                block_counter += 1
                index = cursor
                continue

            if _looks_like_formula(stripped):
                add_units(
                    [
                        _Unit(
                            line.start,
                            line.end,
                            page_text[line.start : line.end],
                            "formula",
                            block_counter,
                            single_sentence=False,
                        )
                    ],
                    "formula",
                )
                block_counter += 1
                index += 1
                continue

            # Prose paragraph: consume consecutive unclassified lines.
            paragraph = [line]
            cursor = index + 1
            while cursor < total:
                candidate = lines[cursor]
                candidate_text = candidate.stripped
                if not candidate_text:
                    break
                if cursor in row_plans or _is_table_caption(candidate_text):
                    break
                if _LIST_ITEM_RE.match(candidate_text):
                    break
                if _looks_like_formula(candidate_text):
                    break
                if section == REFERENCES_SECTION and ENTRY_PATTERN.match(
                    candidate_text
                ):
                    break
                if detect_heading(
                    candidate_text,
                    previous_line=lines[cursor - 1].raw if cursor else "",
                    next_line=(
                        lines[cursor + 1].raw if cursor + 1 < total else ""
                    ),
                    max_chars=self.heading_max_chars,
                ):
                    break
                paragraph.append(candidate)
                cursor += 1

            reflow = _Reflow(paragraph)
            paragraph_start = paragraph[0].start
            paragraph_end = paragraph[-1].end
            has_punctuation = bool(re.search(r"[.!?]", reflow.text))
            tokens = self.count_tokens(page_text[paragraph_start:paragraph_end])
            if not has_punctuation and tokens > self.hard_max_tokens:
                fallback_units = self._fallback_units(
                    page_text, paragraph_start, paragraph_end, block_counter
                )
                add_units(fallback_units, "fallback")
                flags["fallback"] = True
                index = cursor
                block_counter += 1
                continue

            sentence_spans = split_sentences(reflow.text)
            if not sentence_spans:
                fallback_units = self._fallback_units(
                    page_text, paragraph_start, paragraph_end, block_counter
                )
                add_units(fallback_units, "fallback")
                flags["fallback"] = True
                index = cursor
                block_counter += 1
                continue

            sentence_units: list[_Unit] = []
            for sentence_start, sentence_end in sentence_spans:
                original_start = reflow.original_start(sentence_start)
                original_end = reflow.original_end(sentence_end)
                original_start, original_end = _trim(
                    page_text, original_start, original_end
                )
                if original_start >= original_end:
                    continue
                sentence_units.append(
                    _Unit(
                        original_start,
                        original_end,
                        page_text[original_start:original_end],
                        "sentence",
                        block_counter,
                        single_sentence=(len(sentence_spans) == 1),
                    )
                )
            add_units(sentence_units, "sentence")
            index = cursor
            block_counter += 1

        flush_units()
        for record in pending:
            drafts.append(
                _Draft(
                    start=record.start,
                    end=record.end,
                    structure_type="heading",
                    block_index=record.block_index,
                    section=section,
                    confidence="high",
                    section_start=section_start,
                    heading_path=record.path,
                    heading_level=record.level,
                    heading_text=record.text,
                    has_heading=True,
                )
            )
        return drafts, block_counter
    # -- table detection ---------------------------------------------------

    def _build_table_plan(
        self,
        lines: Sequence[_Line],
        caption_index: int,
        page_text: str,
        *,
        page_number: int,
        ordinal: int,
        block_index: int,
    ) -> _TablePlan | None:
        total = len(lines)
        caption_end_index = caption_index
        if caption_index + 1 < total:
            nxt = lines[caption_index + 1]
            gap = nxt.start - lines[caption_index].end
            if _is_caption_continuation(nxt, gap):
                caption_end_index += 1

        caption_start = lines[caption_index].start
        caption_end = lines[caption_end_index].end
        search_limit = caption_end + self.table_orphan_search_chars

        rows_start: int | None = None
        rows_end: int | None = None
        cursor = caption_end_index + 1
        while cursor < total and lines[cursor].start <= search_limit:
            line = lines[cursor]
            if not line.stripped:
                cursor += 1
                continue
            if not _is_row_candidate(
                line,
                previous_line=lines[cursor - 1].raw if cursor else "",
                next_line=lines[cursor + 1].raw if cursor + 1 < total else "",
                row_max_chars=self.table_row_max_chars,
            ):
                cursor += 1
                continue
            run_end = cursor
            while run_end + 1 < total:
                candidate = lines[run_end + 1]
                if not candidate.stripped:
                    break
                if not _is_row_candidate(
                    candidate,
                    previous_line=lines[run_end].raw,
                    next_line=(
                        lines[run_end + 2].raw
                        if run_end + 2 < total
                        else ""
                    ),
                    row_max_chars=self.table_row_max_chars,
                ):
                    break
                run_end += 1
            run = lines[cursor : run_end + 1]
            if _row_run_strength(run, self.table_min_rows):
                rows_start = cursor
                rows_end = run_end
                break
            cursor = run_end + 1

        if rows_start is None or rows_end is None:
            return None

        gap = lines[rows_start].start - caption_end
        between_is_blank = all(
            not lines[k].stripped
            for k in range(caption_end_index + 1, rows_start)
        )
        same_block = gap <= self.table_join_gap_chars and between_is_blank
        caption = _normalise_whitespace(page_text[caption_start:caption_end])
        return _TablePlan(
            table_id=f"p{page_number}-t{ordinal + 1}",
            caption=caption,
            caption_start=caption_start,
            caption_end=caption_end,
            caption_block=block_index,
            caption_lines=(caption_index, caption_end_index),
            rows_start=rows_start,
            rows_end=rows_end,
            rows_block=block_index + 1,
            same_block=same_block,
        )

    def _emit_table_lines(
        self,
        lines: Sequence[_Line],
        *,
        page_text: str,
        section: str,
        section_start: int,
        plan: _TablePlan,
        start_part: int,
        pending: Sequence[_Heading],
        heading: _Heading | None = None,
    ) -> list[_Draft]:
        block_index = (
            plan.caption_block if start_part <= 1 else plan.rows_block
        )
        units = [
            _Unit(
                line.start,
                line.end,
                page_text[line.start : line.end],
                "table_row",
                block_index,
                single_sentence=False,
            )
            for line in lines
            if line.stripped
        ]
        return self._emit_units(
            units,
            page_text=page_text,
            section=section,
            section_start=section_start,
            pending=pending,
            heading=heading,
            allow_overlap=False,
            table=plan,
            start_part=start_part,
            target=self.hard_max_tokens,
        )

    # -- unit packing ------------------------------------------------------

    def _emit_units(
        self,
        units: Sequence[_Unit],
        *,
        page_text: str,
        section: str,
        section_start: int,
        pending: Sequence[_Heading],
        heading: _Heading | None = None,
        allow_overlap: bool,
        table: _TablePlan | None = None,
        start_part: int = 0,
        target: int | None = None,
    ) -> list[_Draft]:
        if not units:
            return []

        hard_limit = self.hard_max_tokens
        soft_limit = target if target is not None else self.target_tokens
        heading_meta = pending[-1] if pending else heading
        first_start = pending[0].start if pending else None
        expanded: list[_Unit] = []
        for unit in units:
            if self.count_tokens(page_text[unit.start : unit.end]) > hard_limit:
                expanded.extend(self._split_oversized(unit, page_text))
            else:
                expanded.append(unit)
        if not expanded:
            return []

        drafts: list[_Draft] = []
        first_chunk = True
        part = start_part if start_part > 0 else 1
        overlap_count = 0
        index = 0
        while index < len(expanded):
            chunk_start = (
                first_start
                if first_chunk and first_start is not None
                else expanded[index].start
            )
            if (
                first_chunk
                and first_start is not None
                and self.count_tokens(
                    page_text[chunk_start : expanded[index].end]
                )
                > hard_limit
            ):
                drafts.append(
                    self._heading_only_draft(
                        pending, section=section, section_start=section_start
                    )
                )
                first_start = None
                chunk_start = expanded[index].start

            cursor = index
            last_end: int | None = None
            while cursor < len(expanded):
                candidate_end = expanded[cursor].end
                if (
                    self.count_tokens(page_text[chunk_start:candidate_end])
                    > hard_limit
                ):
                    break
                last_end = candidate_end
                cursor += 1
                if (
                    self.count_tokens(page_text[chunk_start:last_end])
                    >= soft_limit
                ):
                    break

            if last_end is None:
                pieces = self._split_oversized(expanded[index], page_text)
                if len(pieces) <= 1:
                    # Cannot split further without losing a character.
                    last_end = expanded[index].end
                    cursor = index + 1
                else:
                    expanded = (
                        expanded[:index] + pieces + expanded[index + 1 :]
                    )
                    continue

            group = expanded[index:cursor]
            draft = self._draft_from_group(
                group,
                chunk_start,
                last_end,
                page_text=page_text,
                section=section,
                section_start=section_start,
                heading_meta=heading_meta,
                includes_heading=bool(
                    pending and chunk_start <= pending[0].start
                ),
                table=table,
                table_part=part,
                overlap=overlap_count,
            )
            drafts.append(draft)
            if table is not None:
                part += 1

            next_index = cursor
            next_overlap = 0
            if (
                allow_overlap
                and self.overlap_sentences > 0
                and group
                and group[-1].kind == "sentence"
                and len(group) >= 2
                and cursor < len(expanded)
            ):
                last = group[-1]
                size = last.end - last.start
                chunk_chars = max(1, last_end - chunk_start)
                if (
                    size <= self.max_overlap_chars
                    and size <= self.max_overlap_ratio * chunk_chars
                ):
                    next_index = cursor - 1
                    next_overlap = 1
            overlap_count = next_overlap
            index = next_index
            first_chunk = False

        return drafts

    def _draft_from_group(
        self,
        group: Sequence[_Unit],
        start: int,
        end: int,
        *,
        page_text: str,
        section: str,
        section_start: int,
        heading_meta: _Heading | None,
        includes_heading: bool,
        table: _TablePlan | None,
        table_part: int,
        overlap: int,
    ) -> _Draft:
        kinds = {unit.kind for unit in group}
        if table is not None:
            structure_type = "table"
        elif kinds == {"reference"}:
            structure_type = "reference_entry"
        elif kinds == {"list"}:
            structure_type = "list"
        elif kinds == {"formula"}:
            structure_type = "formula"
        elif kinds == {"fallback"}:
            structure_type = "fallback"
        elif kinds == {"sentence"}:
            structure_type = (
                "sentence_group" if len(group) == 1 else "paragraph"
            )
        else:
            structure_type = "paragraph"

        return _Draft(
            start=start,
            end=end,
            structure_type=structure_type,
            block_index=group[0].block_index,
            section=section,
            confidence="medium",
            section_start=section_start,
            heading_path=heading_meta.path if heading_meta else "",
            heading_level=heading_meta.level if heading_meta else 0,
            heading_text=heading_meta.text if heading_meta else "",
            table_id=table.table_id if table else "",
            table_caption=table.caption if table else "",
            table_part=table_part if table else 0,
            forced=any(unit.forced for unit in group),
            overlap=overlap,
            has_heading=includes_heading,
            starts_forced=bool(group and group[0].forced),
        )

    def _heading_only_draft(
        self,
        pending: Sequence[_Heading],
        *,
        section: str,
        section_start: int,
    ) -> _Draft:
        last = pending[-1]
        return _Draft(
            start=pending[0].start,
            end=last.end,
            structure_type="heading",
            block_index=last.block_index,
            section=section,
            confidence="high",
            section_start=section_start,
            heading_path=last.path,
            heading_level=last.level,
            heading_text=last.text,
            has_heading=True,
        )
    # -- oversized units and fallback --------------------------------------

    def _max_prefix(self, text: str, start: int, limit: int) -> int:
        """Largest end offset in ``(start, limit]`` within the hard limit."""
        low = start + 1
        high = limit
        best = start + 1
        while low <= high:
            middle = (low + high) // 2
            if self.count_tokens(text[start:middle]) <= self.hard_max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _split_oversized(
        self, unit: _Unit, page_text: str
    ) -> list[_Unit]:
        """Split one over-budget unit at a clause or word boundary."""
        pieces: list[_Unit] = []
        start = unit.start
        end = unit.end
        while start < end:
            while start < end and page_text[start].isspace():
                start += 1
            if start >= end:
                break
            if self.count_tokens(page_text[start:end]) <= self.hard_max_tokens:
                pieces.append(
                    _Unit(
                        start,
                        end,
                        page_text[start:end],
                        unit.kind,
                        unit.block_index,
                        forced=True,
                        single_sentence=unit.single_sentence,
                    )
                )
                break

            cut = self._max_prefix(page_text, start, end)
            window_start = start + max(1, (cut - start) // 2)
            clause = -1
            for separator in ";:,—":
                position = page_text.rfind(separator, window_start, cut)
                if position > clause:
                    clause = position
            if clause >= window_start:
                cut = clause + 1
            else:
                space = page_text.rfind(" ", start + 1, cut)
                if space > start:
                    cut = space
            if cut <= start:
                cut = start + 1

            piece_end = cut
            while piece_end > start and page_text[piece_end - 1].isspace():
                piece_end -= 1
            if piece_end > start:
                pieces.append(
                    _Unit(
                        start,
                        piece_end,
                        page_text[start:piece_end],
                        unit.kind,
                        unit.block_index,
                        forced=True,
                        single_sentence=unit.single_sentence,
                    )
                )
            start = cut
        return pieces

    def _fallback_units(
        self, page_text: str, start: int, end: int, block_index: int
    ) -> list[_Unit]:
        """Conservative word-window fallback for text with no sentences."""
        units: list[_Unit] = []
        cursor = start
        while cursor < end:
            while cursor < end and page_text[cursor].isspace():
                cursor += 1
            if cursor >= end:
                break
            limit = min(end, cursor + self.fallback_chars)
            cut = self._max_prefix(page_text, cursor, limit)
            if cut < end:
                space = page_text.rfind(" ", cursor + 1, cut)
                if space > cursor:
                    cut = space
            piece_end = cut
            while piece_end > cursor and page_text[piece_end - 1].isspace():
                piece_end -= 1
            if piece_end <= cursor:
                piece_end = min(end, cursor + 1)
            units.append(
                _Unit(
                    cursor,
                    piece_end,
                    page_text[cursor:piece_end],
                    "fallback",
                    block_index,
                    single_sentence=False,
                )
            )
            cursor = cut
        return units

    # -- chunk emission ----------------------------------------------------

    @staticmethod
    def _starts_mid_sentence(
        page_text: str, start: int, section_start: int
    ) -> bool:
        if start <= section_start:
            return False
        cursor = start - 1
        while cursor >= 0 and page_text[cursor].isspace():
            cursor -= 1
        if cursor < 0:
            return False
        return page_text[cursor] not in ".!?;:"

    @staticmethod
    def _ends_mid_sentence(page_text: str, end: int) -> bool:
        if end >= len(page_text):
            return False
        cursor = end - 1
        while cursor >= 0 and page_text[cursor].isspace():
            cursor -= 1
        if cursor < 0:
            return False
        char = page_text[cursor]
        if char in ".!?:;":
            return False
        if char in "\"')]}":
            inner = cursor - 1
            while inner >= 0 and page_text[inner] in "\"')]}":
                inner -= 1
            if inner >= 0 and page_text[inner] in ".!?:;":
                return False
        return True

    def _make_chunk(
        self,
        document: Document,
        page_text: str,
        start: int,
        end: int,
        draft: _Draft,
        chunk_index: int,
        *,
        forced: bool = False,
        previous_page_text: str = "",
    ) -> Chunk:
        text = page_text[start:end]
        page = document["metadata"]
        prose_types = {"sentence_group", "paragraph", "fallback"}
        starts_mid = False
        ends_mid = False
        if draft.structure_type in prose_types:
            if draft.overlap == 0 and not draft.has_heading:
                if draft.starts_forced:
                    starts_mid = True
                elif start == 0 and page["page_number"] > 1:
                    tail = previous_page_text.rstrip()
                    if tail and tail[-1] not in ".!?:;":
                        starts_mid = True
                elif draft.structure_type == "fallback":
                    starts_mid = self._starts_mid_sentence(
                        page_text, start, draft.section_start
                    )
            ends_mid = self._ends_mid_sentence(page_text, end)

        metadata: dict[str, Any] = {
            "source": page["source"],
            "source_path": page["source_path"],
            "page_number": page["page_number"],
            "total_pages": page["total_pages"],
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
            "section": draft.section,
            "chunking_version": CHUNKING_VERSION,
            "structure_type": draft.structure_type,
            "structure_confidence": draft.confidence,
            "heading_path": draft.heading_path,
            "heading_level": draft.heading_level,
            "heading_text": draft.heading_text,
            "table_id": draft.table_id,
            "table_caption": draft.table_caption,
            "table_part": draft.table_part,
            "block_index": draft.block_index,
            "starts_mid_sentence": starts_mid,
            "ends_mid_sentence": ends_mid,
            "is_forced_split": bool(draft.forced or forced),
            "overlap_sentences": draft.overlap,
            "token_count": self.count_tokens(text),
        }
        return {"text": text, "metadata": metadata}  # type: ignore[return-value]


def chunk_documents(
    documents: Iterable[Document],
    *,
    token_counter: Callable[[str], int] | None = None,
    **kwargs: Any,
) -> list[Chunk]:
    """Shortcut for ``StructureAwareChunker(...).chunk_documents(...)``."""
    return StructureAwareChunker(
        token_counter=token_counter, **kwargs
    ).chunk_documents(documents)
