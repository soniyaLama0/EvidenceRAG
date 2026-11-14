"""Pure, testable retrieval metrics for Stage 6D.

Everything here works on plain integers and dictionaries, so it can be tested
without Chroma, without a model and without the paper. The evaluation script
does the I/O and calls these functions.

Relevance rule
--------------
A retrieved chunk is relevant to a question when, on the same page, it overlaps
a reviewed gold passage by at least ``threshold`` of *either*:

* the chunk's own length (the chunk is mostly evidence), or
* the gold passage's length (the chunk carries most of the evidence).

The two-sided rule matters because a 60-character evidence sentence can sit
inside a 1000-character chunk. A one-sided "chunk is mostly evidence" test
would label almost nothing relevant at these chunk sizes, and a one-sided
"passage is mostly covered" test would accept a chunk that only brushes the
start of the sentence. ``threshold=0.5`` is the primary setting; the evaluation
also reports 0.3 and 0.7 so the conclusion can be checked for threshold
sensitivity.

Chunk-level metrics (recall / precision / MRR) use each collection's own gold
chunk set, because the two collections chunk the paper differently. Passage
coverage (``passage_coverage``) is collection-independent: it asks whether the
top-k chunks together expose at least most of the reviewed passage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Default two-sided overlap threshold.
DEFAULT_THRESHOLD = 0.5

#: A passage counts as surfaced when the union of retrieved chunks covers this
#: fraction of it. 0.7 tolerates chunk-boundary losses.
DEFAULT_COVERAGE = 0.7


def overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    """Half-open interval overlap length, never negative."""
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def is_relevant(
    chunk: Mapping[str, Any],
    passages: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Return whether ``chunk`` carries reviewed evidence for a question.

    Args:
        chunk: ``{"page_number": int, "start": int, "end": int}`` in cleaned
            page coordinates.
        passages: Gold passages for one question, same coordinate system.
        threshold: Two-sided overlap fraction in ``(0, 1]``.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}.")

    page_number = chunk["page_number"]
    start = chunk["start"]
    end = chunk["end"]
    length = end - start
    if length <= 0:
        return False

    for passage in passages:
        if passage["page_number"] != page_number:
            continue
        shared = overlap(start, end, passage["start"], passage["end"])
        if shared == 0:
            continue

        passage_length = passage["end"] - passage["start"]
        if shared / length >= threshold:
            return True
        if passage_length > 0 and shared / passage_length >= threshold:
            return True

    return False


def covered_length(
    intervals: Iterable[tuple[int, int]],
) -> int:
    """Total length of the union of half-open intervals."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    total = 0
    current_start: int | None = None
    current_end: int | None = None

    for start, end in ordered:
        if current_start is None:
            current_start, current_end = start, end
            continue
        assert current_end is not None
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)

    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def passage_coverage(
    chunks: Sequence[Mapping[str, Any]],
    passage: Mapping[str, Any],
) -> float:
    """Fraction of one gold passage covered by the retrieved chunks."""
    page_number = passage["page_number"]
    intervals = [
        (
            max(chunk["start"], passage["start"]),
            min(chunk["end"], passage["end"]),
        )
        for chunk in chunks
        if chunk["page_number"] == page_number
    ]
    length = passage["end"] - passage["start"]
    if length <= 0:
        return 0.0
    return min(1.0, covered_length(intervals) / length)


def precision_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive.")
    # Short result lists are effectively padded with non-relevant positions,
    # so a collection cannot gain precision by returning fewer results.
    top = retrieved[:k]
    return len([item for item in top if item in gold]) / k


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float | None:
    """Fraction of gold chunks found in the top-k, or ``None`` if gold is empty."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not gold:
        return None
    return len(set(retrieved[:k]) & gold) / len(gold)


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none is relevant."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(
    retrieved: Sequence[str],
    gold: set[str],
    *,
    k: int,
    passages: Sequence[Mapping[str, Any]] | None = None,
    chunks: Mapping[str, Mapping[str, Any]] | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE,
) -> dict[str, Any]:
    """Compute all per-question metrics for one ranking cut at ``k``.

    Args:
        retrieved: Ordered chunk IDs (best first).
        gold: Gold chunk IDs for this question and collection.
        k: Rank cut-off.
        passages: Optional gold passages, enables coverage recall.
        chunks: Optional ``id -> {page_number,start,end}`` map, needed to
            compute coverage.
        coverage_threshold: Passage coverage needed to count as surfaced.
    """
    top = list(retrieved[:k])
    relevant_hits = [chunk_id for chunk_id in top if chunk_id in gold]

    result: dict[str, Any] = {
        "k": k,
        "returned": len(top),
        "gold_chunks": len(gold),
        "relevant_retrieved": len(relevant_hits),
        "precision": precision_at_k(retrieved, gold, k),
        "recall": recall_at_k(retrieved, gold, k),
        "reciprocal_rank": reciprocal_rank(retrieved, gold),
    }

    if passages is not None and chunks is not None:
        retrieved_chunks = [chunks[chunk_id] for chunk_id in top if chunk_id in chunks]
        covered = [
            passage_coverage(retrieved_chunks, passage)
            for passage in passages
        ]
        surfaced = [value for value in covered if value >= coverage_threshold]
        result["passages"] = len(passages)
        result["passages_surfaced"] = len(surfaced)
        result["passage_recall"] = (
            len(surfaced) / len(passages) if passages else None
        )
        result["mean_passage_coverage"] = (
            sum(covered) / len(covered) if covered else 0.0
        )

    return result


def macro_average(
    values: Iterable[float | None],
) -> float | None:
    """Mean of the non-``None`` values, or ``None`` when nothing is defined."""
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)
