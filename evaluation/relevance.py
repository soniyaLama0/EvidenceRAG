"""Manually reviewed relevance labels for the Stage 6D retrieval evaluation.

Labels are expressed as **literal snippets of the cleaned paper text**, not as
chunk IDs. Two reasons:

* Chunk IDs and offsets differ between ``evidence_chunks`` (raw pages) and
  ``evidence_chunks_v2`` (cleaned pages). Gold chunk IDs would therefore not be
  comparable, and would quietly change when a chunk boundary moves.
* A literal snippet can be re-verified by anyone: ``resolve_gold()`` fails
  loudly if a snippet is missing or appears more than once, so a label can
  never silently drift away from the paper.

Snippets are resolved against the cleaned page text produced by
``PageCleaner``. Cleaning only removes a trailing publisher footer, so the same
offsets are valid for the body text stored in ``evidence_chunks`` as well.

Reviewer notes on sufficiency (see ``QUESTIONS``):

* Q1-Q3 and Q6 have direct, explicit answers in the paper.
* Q4 and Q5 are marked ``partial``: the paper evaluates *feature completion and
  latency*, not user outcomes. There is no quantitative accuracy table and no
  user study, and the authors say so explicitly (page 4, "Formal User
  Evaluation"). Retrieval can only return what exists; the report must not
  claim more.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

#: ``"sufficient"``: the paper states the answer directly.
#: ``"partial"``: relevant text exists, but the paper is explicit that the
#: evidence is incomplete (no formal user study / no accuracy metrics).
SUFFICIENT = "sufficient"
PARTIAL = "partial"


class Question(TypedDict):
    """One evaluation question and the reviewer's sufficiency judgement."""

    id: str
    text: str
    sufficiency: str
    note: str


class GoldPassage(TypedDict):
    """A reviewed evidence passage resolved to cleaned-text offsets."""

    page_number: int
    start: int
    end: int
    text: str


QUESTIONS: list[Question] = [
    {
        "id": "q1",
        "text": "What is Vyapar AI?",
        "sufficiency": SUFFICIENT,
        "note": (
            "Abstract and introduction both define the system: a voice-first, "
            "offline Android application for Hinglish business commands."
        ),
    },
    {
        "id": "q2",
        "text": "What problem does Vyapar AI solve?",
        "sufficiency": SUFFICIENT,
        "note": (
            "Page 1 states the MSME digitisation gap (63M businesses, >95% "
            "undigitised, low digital literacy, connectivity) and page 2 "
            "states the integration gap explicitly."
        ),
    },
    {
        "id": "q3",
        "text": "Which technologies are used?",
        "sufficiency": SUFFICIENT,
        "note": (
            "Table III on page 4 lists the full stack; page 1 names the same "
            "components in prose."
        ),
    },
    {
        "id": "q4",
        "text": "What evaluation does the paper perform?",
        "sufficiency": PARTIAL,
        "note": (
            "Only feature validation and latency claims, no user study and no "
            "quantitative accuracy results. The authors acknowledge the gap."
        ),
    },
    {
        "id": "q5",
        "text": "What are the main findings?",
        "sufficiency": PARTIAL,
        "note": (
            "Findings are limited to 'all twelve features implemented', "
            "sub-500 ms STT and sub-100 ms NLP; no measured business outcome."
        ),
    },
    {
        "id": "q6",
        "text": "What are the limitations?",
        "sufficiency": SUFFICIENT,
        "note": (
            "Section VII lists five named limitations: NLP generalisation, "
            "scalability, language coverage, GST compliance, formal evaluation."
        ),
    },
]

#: Reviewed evidence snippets: ``question id -> [(page number, exact text)]``.
#:
#: Every snippet is a single line of the cleaned text (no embedded newlines)
#: and must occur exactly once in the document. Keep them verbatim.
GOLD_SNIPPETS: dict[str, list[tuple[int, str]]] = {
    "q1": [
        (
            1,
            "AI tackles these problems by offering a voice-first, "
            "offline-capable mobile application for Android based on natural "
            "voice commands",
        ),
        (1, "This paper proposes Vyapar AI, a voice-based mobile"),
        (
            1,
            "Abstract: India has a large number of micro, small, and medium "
            "enterprises (MSMEs) over 63 million in total, almost all of which",
        ),
    ],
    "q2": [
        (1, "There are over 63 million MSMEs in India, which contribute"),
        (1, "impact is significant, over 95% of these businesses do not use"),
        (1, "low digital literacy levels, complex UI/UX design,"),
        (2, "context of small businesses; however, none of them integrate"),
    ],
    "q3": [
        (4, "Frontend React Native CLI (Android only, no Expo)"),
        (4, "Speech-to-Text Vosk (offline, on-device, ~40 MB model)"),
        (4, "Backend FastAPI (Python, async, modular)"),
        (
            1,
            "The software combines React Native CLI frontend, Python FastAPI "
            "backend, and on-device",
        ),
    ],
    "q4": [
        (4, "The application was implemented and tested on Android"),
        (4, "hardware. All twelve core features were successfully"),
        (4, "summarises the feature validation status for the current"),
        (4, "User studies with relevant participants and latency"),
    ],
    "q5": [
        (4, "hardware. All twelve core features were successfully"),
        (4, "on-device with STT latency under 500 ms. Rule-based NLP"),
        (4, "The Vyapar AI system provides a proof-of-concept that voice"),
        (5, "Overall, this project has shown that a thoughtful application"),
    ],
    "q6": [
        (4, "it is limited to a predetermined number of"),
        (4, "supported, India's linguistic diversity means that a large share"),
        (4, "There is currently no functionality regarding the computation"),
        (4, "User studies with relevant participants and latency"),
        (4, "workload for MSME owners compared to current GUI"),
    ],
}


def _page_texts(
    documents: Sequence[Mapping[str, object]],
) -> dict[int, str]:
    """Map ``page_number -> cleaned text`` from page documents."""
    texts: dict[int, str] = {}
    for document in documents:
        metadata = document.get("metadata")
        text = document.get("text")
        if not isinstance(metadata, Mapping) or not isinstance(text, str):
            raise TypeError("documents must contain 'text' and 'metadata'.")
        page_number = metadata.get("page_number")
        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise ValueError("document metadata['page_number'] must be an integer.")
        texts[page_number] = text
    return texts


def resolve_gold(
    documents: Sequence[Mapping[str, object]],
) -> dict[str, list[GoldPassage]]:
    """Resolve reviewed snippets to offsets in the supplied page text.

    Args:
        documents: Page documents with ``text`` and ``metadata.page_number``.
            Pass the **cleaned** pages when evaluating ``evidence_chunks_v2``.

    Returns:
        ``question id -> [GoldPassage]`` in reviewed order.

    Raises:
        ValueError: If a snippet is absent, appears more than once, or refers
            to a page that is not present. Labels are never silently dropped.
    """
    page_texts = _page_texts(documents)

    resolved: dict[str, list[GoldPassage]] = {}
    for question_id, snippets in GOLD_SNIPPETS.items():
        passages: list[GoldPassage] = []
        for page_number, snippet in snippets:
            text = page_texts.get(page_number)
            if text is None:
                raise ValueError(
                    f"{question_id}: page {page_number} is not in the document."
                )
            first = text.find(snippet)
            if first < 0:
                raise ValueError(
                    f"{question_id}: snippet not found on page {page_number}: "
                    f"{snippet!r}"
                )
            second = text.find(snippet, first + 1)
            if second >= 0:
                raise ValueError(
                    f"{question_id}: snippet is ambiguous on page {page_number} "
                    f"(found at {first} and {second}): {snippet!r}"
                )
            passages.append(
                {
                    "page_number": page_number,
                    "start": first,
                    "end": first + len(snippet),
                    "text": snippet,
                }
            )
        resolved[question_id] = passages

    return resolved
