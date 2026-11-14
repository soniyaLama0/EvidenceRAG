"""Tests for the Stage 6D evaluation modules.

The metric and relevance tests are pure and fast. The integration test runs the
full ``evaluate()`` wiring against a fake in-memory store and a generated
one-page PDF, so it exercises the script's real control flow without touching
``data/chroma`` and without loading the embedding model.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

import evaluation.relevance as relevance
import evaluation.run_retrieval_eval as eval_mod
from evaluation.metrics import (
    covered_length,
    evaluate_ranking,
    is_relevant,
    macro_average,
    overlap,
    passage_coverage,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.run_retrieval_eval import (
    collection_configs,
    gold_chunks,
    read_chunk_index,
    retrieved_rows,
    run_queries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def passage(page: int, start: int, end: int) -> dict:
    return {"page_number": page, "start": start, "end": end, "text": "x" * (end - start)}


def chunk(page: int, start: int, end: int) -> dict:
    return {"page_number": page, "chunk_index": 0, "start": start, "end": end}


# ---------------------------------------------------------------------------
# Gold label resolution
# ---------------------------------------------------------------------------


def test_resolve_gold_returns_exact_offsets(monkeypatch) -> None:
    monkeypatch.setattr(relevance, "GOLD_SNIPPETS", {"q1": [(1, "voice-first")]})
    documents = [{"text": "A voice-first assistant.", "metadata": {"page_number": 1}}]

    gold = relevance.resolve_gold(documents)

    assert gold == {
        "q1": [
            {
                "page_number": 1,
                "start": 2,
                "end": 13,
                "text": "voice-first",
            }
        ]
    }


def test_resolve_gold_rejects_missing_snippet(monkeypatch) -> None:
    monkeypatch.setattr(relevance, "GOLD_SNIPPETS", {"q1": [(1, "not in the paper")]})
    documents = [{"text": "A voice-first assistant.", "metadata": {"page_number": 1}}]

    with pytest.raises(ValueError, match="snippet not found"):
        relevance.resolve_gold(documents)


def test_resolve_gold_rejects_ambiguous_snippet(monkeypatch) -> None:
    monkeypatch.setattr(relevance, "GOLD_SNIPPETS", {"q1": [(1, "voice")]})
    documents = [
        {"text": "voice and voice", "metadata": {"page_number": 1}},
    ]

    with pytest.raises(ValueError, match="ambiguous"):
        relevance.resolve_gold(documents)


def test_resolve_gold_rejects_missing_page(monkeypatch) -> None:
    monkeypatch.setattr(relevance, "GOLD_SNIPPETS", {"q1": [(2, "voice")]})
    documents = [{"text": "voice", "metadata": {"page_number": 1}}]

    with pytest.raises(ValueError, match="page 2 is not in the document"):
        relevance.resolve_gold(documents)


def test_reviewed_questions_and_snippets_are_consistent() -> None:
    question_ids = {question["id"] for question in relevance.QUESTIONS}
    assert question_ids == set(relevance.GOLD_SNIPPETS)
    assert len(relevance.QUESTIONS) == 6
    for question in relevance.QUESTIONS:
        assert question["sufficiency"] in (relevance.SUFFICIENT, relevance.PARTIAL)
        assert question["note"]
        assert relevance.GOLD_SNIPPETS[question["id"]]


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ((0, 10, 5, 15), 5),
        ((0, 10, 10, 20), 0),
        ((0, 10, 20, 30), 0),
        ((0, 100, 20, 30), 10),
        ((20, 30, 0, 100), 10),
    ],
)
def test_overlap(intervals: tuple[int, int, int, int], expected: int) -> None:
    assert overlap(*intervals) == expected


def test_covered_length_merges_intervals() -> None:
    assert covered_length([]) == 0
    assert covered_length([(0, 10)]) == 10
    assert covered_length([(0, 10), (5, 15)]) == 15
    assert covered_length([(0, 10), (10, 20)]) == 20
    assert covered_length([(0, 10), (20, 30)]) == 20
    assert covered_length([(5, 5), (0, 10)]) == 10


def test_passage_coverage_is_union_based() -> None:
    target = passage(1, 100, 200)

    assert passage_coverage([chunk(1, 0, 300)], target) == 1.0
    assert passage_coverage([chunk(1, 100, 150)], target) == pytest.approx(0.5)
    assert passage_coverage([chunk(1, 100, 150), chunk(1, 150, 200)], target) == 1.0
    assert passage_coverage([chunk(2, 100, 200)], target) == 0.0
    assert passage_coverage([chunk(1, 200, 300)], target) == 0.0


# ---------------------------------------------------------------------------
# Relevance rule
# ---------------------------------------------------------------------------


def test_relevance_when_chunk_is_mostly_evidence() -> None:
    target = passage(1, 0, 100)

    assert is_relevant(chunk(1, 0, 100), [target])
    assert is_relevant(chunk(1, 20, 120), [target])  # 80% of the chunk
    # 20% of this chunk is evidence and only 20% of the passage is covered.
    assert not is_relevant(chunk(1, 80, 280), [target])


def test_relevance_when_chunk_contains_most_of_the_passage() -> None:
    target = passage(1, 500, 550)

    assert is_relevant(chunk(1, 0, 1000), [target])


def test_relevance_requires_the_same_page() -> None:
    target = passage(1, 0, 100)
    assert not is_relevant(chunk(2, 0, 100), [target])


def test_relevance_ignores_zero_length_chunks() -> None:
    assert not is_relevant(chunk(1, 10, 10), [passage(1, 0, 100)])


def test_relevance_threshold_is_validated() -> None:
    with pytest.raises(ValueError, match="threshold"):
        is_relevant(chunk(1, 0, 10), [], threshold=0)
    with pytest.raises(ValueError, match="threshold"):
        is_relevant(chunk(1, 0, 10), [], threshold=1.5)


def test_threshold_sensitivity_changes_labels() -> None:
    target = passage(1, 0, 100)
    partial = chunk(1, 80, 140)  # 33% of the chunk, 20% of the passage

    assert not is_relevant(partial, [target], threshold=0.5)
    assert is_relevant(partial, [target], threshold=0.3)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def test_precision_recall_and_reciprocal_rank() -> None:
    gold = {"a", "b", "c"}

    assert precision_at_k(["a", "x", "b"], gold, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "x", "b"], gold, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["x", "a"], gold, 1) == 0.0
    assert reciprocal_rank(["x", "a", "b"], gold) == pytest.approx(0.5)
    assert reciprocal_rank(["x", "y"], gold) == 0.0


def test_recall_is_none_without_gold() -> None:
    assert recall_at_k(["a"], set(), 3) is None
    assert precision_at_k(["a"], set(), 3) == 0.0


def test_short_result_lists_do_not_inflate_precision() -> None:
    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1 / 3)


def test_evaluate_ranking_includes_passage_metrics() -> None:
    chunks = {
        "a": chunk(1, 0, 600),
        "b": chunk(1, 600, 1200),
    }
    target = passage(1, 0, 500)

    result = evaluate_ranking(
        ["a", "b"], {"a"}, k=2, passages=[target], chunks=chunks
    )

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == 1.0
    assert result["reciprocal_rank"] == 1.0
    assert result["passages_surfaced"] == 1
    assert result["mean_passage_coverage"] == 1.0


def test_macro_average_skips_undefined_values() -> None:
    assert macro_average([1.0, None, 0.0]) == pytest.approx(0.5)
    assert macro_average([]) is None
    assert macro_average([None]) is None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


class FakeCollection:
    def __init__(self, ids, metadatas) -> None:
        self._ids = ids
        self._metadatas = metadatas
        self.get_calls = 0

    def get(self, include=None):
        self.get_calls += 1
        return {"ids": list(self._ids), "metadatas": list(self._metadatas)}


class FakeStore:
    """Read-only store: implements only what evaluation is allowed to use."""

    def __init__(self, collection_name: str, records=None) -> None:
        self.collection_name = collection_name
        if records is None:
            records = (
                BASELINE_RECORDS
                if collection_name == "evidence_chunks"
                else CLEANED_RECORDS
            )
        self.records = records
        self.collection = FakeCollection(
            [record["id"] for record in records],
            [record["metadata"] for record in records],
        )

    @classmethod
    def for_cleaned_ingestion(cls):
        return cls("evidence_chunks_v2", CLEANED_RECORDS)

    def get_collection_count(self) -> int:
        return len(self.records)

    def search(self, query_embedding, n_results=5, where=None):
        selected = self.records
        if where:
            selected = [
                record
                for record in selected
                if all(record["metadata"].get(k) == v for k, v in where.items())
            ]
        results = []
        for rank, record in enumerate(selected[:n_results]):
            distance = 0.1 * rank
            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": record["metadata"],
                    "distance": distance,
                    "similarity": 1.0 - distance,
                }
            )
        return results


def make_metadata(page: int, index: int, start: int, end: int, section=None) -> dict:
    metadata = {
        "source": "paper.pdf",
        "source_path": "C:/data/paper.pdf",
        "page_number": page,
        "total_pages": 1,
        "chunk_index": index,
        "char_start": start,
        "char_end": end,
        "token_count": 10,
        "is_truncated": False,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }
    if section is not None:
        metadata["section"] = section
    return metadata


BASELINE_RECORDS = [
    {
        "id": "v1a",
        "text": "Evidence sentence here.",
        "metadata": make_metadata(1, 0, 0, 600),
    },
    {
        "id": "v1b",
        "text": "Unrelated filler.",
        "metadata": make_metadata(1, 1, 600, 1200),
    },
]
CLEANED_RECORDS = [
    {
        "id": "v2a",
        "text": "Evidence sentence here.",
        "metadata": make_metadata(1, 0, 0, 600, "body"),
    },
    {
        "id": "v2b",
        "text": "A reference entry.",
        "metadata": make_metadata(1, 1, 600, 1200, "references"),
    },
]


def test_read_chunk_index_and_gold_chunks() -> None:
    index = read_chunk_index(FakeStore("evidence_chunks", BASELINE_RECORDS))  # type: ignore[arg-type]

    assert index["v1a"]["start"] == 0
    assert index["v1a"]["section"] is None
    assert gold_chunks(index, [passage(1, 0, 20)]) == {"v1a"}


def test_collection_configs_include_primary_and_filters() -> None:
    configs = collection_configs()
    by_name = {config["config"]: config for config in configs}

    assert by_name["v1_unfiltered"]["collection"] == "evidence_chunks"
    assert by_name["v2_unfiltered"]["collection"] == "evidence_chunks_v2"
    assert by_name["v2_body_only"]["filter"] == {"section": "body"}
    assert by_name["v2_references_only"]["filter"] == {"section": "references"}
    assert by_name["v1_unfiltered"]["primary"] is True


class RecordingRetriever:
    def __init__(self, results) -> None:
        self.results = results
        self.calls: list[dict] = []

    def retrieve(self, query, top_k=5, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        return self.results


def test_run_queries_passes_top_k_and_filter_through() -> None:
    retriever = RecordingRetriever([{"id": "x"}])
    questions = [
        {"id": "q1", "text": "Question one?", "sufficiency": "sufficient", "note": ""},
        {"id": "q2", "text": "Question two?", "sufficiency": "sufficient", "note": ""},
    ]

    out = run_queries(retriever, questions, top_k=3, where={"section": "body"})  # type: ignore[arg-type]

    assert set(out) == {"q1", "q2"}
    assert [call["top_k"] for call in retriever.calls] == [3, 3]
    assert [call["where"] for call in retriever.calls] == [{"section": "body"}] * 2


def test_retrieved_rows_are_flat_and_complete() -> None:
    rows = retrieved_rows(
        config={
            "config": "v2_unfiltered",
            "collection": "evidence_chunks_v2",
            "filter": None,
        },
        question={
            "id": "q1",
            "text": "What is X?",
            "sufficiency": "sufficient",
            "note": "",
        },
        k=3,
        results=[
            {
                "id": "v2a",
                "text": "Evidence sentence here.",
                "metadata": {
                    **make_metadata(1, 0, 0, 600, "body"),
                    "section": "body",
                },
                "distance": 0.1,
                "similarity": 0.9,
            }
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == set(eval_mod.CSV_FIELDS)
    assert row["rank"] == 1
    assert row["chunk_id"] == "v2a"
    assert row["section"] == "body"
    assert row["similarity"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Full wiring (fake store + generated PDF + fake embeddings)
# ---------------------------------------------------------------------------


def make_one_page_pdf(path: Path, body: str) -> None:
    pdf = canvas.Canvas(str(path), pagesize=(612, 792))
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 720, body)
    pdf.showPage()
    pdf.save()


class FakeEmbeddingModel:
    def embed_texts(self, texts):
        import numpy as np

        return np.asarray([[0.0] * 383 + [1.0] for _ in texts], dtype="float32")


def test_evaluate_writes_json_and_csv_and_is_read_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_one_page_pdf(pdf_path, "Evidence sentence here and nothing else.")

    monkeypatch.setattr(eval_mod, "VectorStore", FakeStore)
    monkeypatch.setattr(eval_mod, "EmbeddingModel", FakeEmbeddingModel)
    monkeypatch.setattr(
        eval_mod,
        "QUESTIONS",
        [
            {
                "id": "q1",
                "text": "What is X?",
                "sufficiency": "sufficient",
                "note": "test",
            }
        ],
    )
    monkeypatch.setattr(
        eval_mod,
        "resolve_gold",
        lambda pages: {"q1": [passage(1, 0, 24)]},
    )
    monkeypatch.setattr(
        eval_mod,
        "EXPECTED_COUNTS",
        {"evidence_chunks": 2, "evidence_chunks_v2": 2},
    )

    report = eval_mod.evaluate(
        pdf_path=pdf_path,
        output_dir=tmp_path / "out",
        k_values=(1, 2),
        thresholds=(0.5,),
    )

    json_path = tmp_path / "out" / "stage6d_results.json"
    csv_path = tmp_path / "out" / "stage6d_retrieval.csv"
    assert json_path.exists()
    assert csv_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["counts_before"] == loaded["counts_after"] == {
        "evidence_chunks": 2,
        "evidence_chunks_v2": 2,
    }
    assert loaded["meta"]["k_values"] == [1, 2]
    assert "v1_unfiltered" in loaded["metrics"]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert set(rows[0]) == set(eval_mod.CSV_FIELDS)

    # The reference-only config must find no body evidence.
    assert report["metrics"]["v2_references_only"]["q1"]["1"]["precision"] == 0.0
    # The v2 body filter keeps the evidence chunk.
    assert report["metrics"]["v2_body_only"]["q1"]["1"]["precision"] == 1.0
    # FakeStore has no write methods, so a write attempt would have failed.
    assert "DATABASE" not in capsys.readouterr().out


def test_main_reports_safety_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    make_one_page_pdf(pdf_path, "Some body text.")

    monkeypatch.setattr(
        eval_mod,
        "evaluate",
        lambda **kwargs: {
            "counts_after": {"evidence_chunks": 38, "evidence_chunks_v2": 35},
            "counts_before": {"evidence_chunks": 39, "evidence_chunks_v2": 35},
        },
    )

    exit_code = eval_mod.main([str(pdf_path)])

    assert exit_code == 1
    assert "SAFETY CHECK FAILED" in capsys.readouterr().err


def test_main_rejects_missing_pdf(tmp_path: Path) -> None:
    assert eval_mod.main([str(tmp_path / "missing.pdf")]) == 2
