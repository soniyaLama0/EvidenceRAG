"""Stage 6D: read-only retrieval comparison of ``evidence_chunks`` vs ``evidence_chunks_v2``.

What it does
------------
1. Loads and cleans the sample PDF (in memory only) to resolve reviewed gold
   passages to cleaned-text offsets.
2. Opens both Chroma collections **read-only** through ``VectorStore`` and
   reads their metadata to build each collection's own gold chunk set.
3. Runs the six evaluation questions through ``Retriever`` with identical
   settings: same embedding model, same query text, ``k = 3`` and ``k = 5``,
   same cosine metric.
4. Writes ``evaluation/results/stage6d_results.json`` (full machine-readable
   record) and ``..._retrieval.csv`` (one row per retrieved chunk).
5. Prints a comparison table and verifies the collection counts are unchanged.

Supplementary configurations (v2 body-only / references-only) demonstrate the
section filter. They are exploratory and reported separately from the primary
v1-vs-v2 comparison.

Nothing in this module writes to Chroma. Only ``collection.get`` and
``retriever.retrieve`` are used.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m evaluation.run_retrieval_eval
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    DEFAULT_THRESHOLD,
    evaluate_ranking,
    is_relevant,
    macro_average,
)
from evaluation.relevance import QUESTIONS, Question, resolve_gold
from ingestion.embedding_model import EMBEDDING_MODEL_NAME, EmbeddingModel
from ingestion.page_cleaner import PageCleaner
from ingestion.pdf_loader import PDFLoader
from ingestion.section_segmenter import REFERENCES_SECTION, SectionSegmenter
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "research_paper.pdf"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

BASELINE_COLLECTION = "evidence_chunks"
CLEANED_COLLECTION = "evidence_chunks_v2"
EXPECTED_COUNTS = {BASELINE_COLLECTION: 39, CLEANED_COLLECTION: 35}

K_VALUES = (3, 5)
THRESHOLD_SENSITIVITY = (0.3, 0.5, 0.7)

CSV_FIELDS = (
    "config",
    "collection",
    "filter",
    "question_id",
    "question",
    "k",
    "rank",
    "chunk_id",
    "page_number",
    "chunk_index",
    "section",
    "char_start",
    "char_end",
    "similarity",
    "distance",
    "token_count",
    "is_truncated",
    "text",
)

#: Exact publisher-banner fragments. Used only for error analysis, never to
#: alter retrieval. A reference entry that merely cites "IJERT" does not match.
FURNITURE_MARKERS = (
    "Published by : International Journal of Engineering Research",
    "IJERTV15IS061078 Page",
    "An International Peer-Reviewed Journal Vol.",
    "(This work is licensed under a Creative Commons",
)


def collection_configs() -> list[dict[str, Any]]:
    """Primary comparison first, then the supplementary section filters."""
    return [
        {
            "config": "v1_unfiltered",
            "collection": BASELINE_COLLECTION,
            "filter": None,
            "primary": True,
        },
        {
            "config": "v2_unfiltered",
            "collection": CLEANED_COLLECTION,
            "filter": None,
            "primary": True,
        },
        {
            "config": "v2_body_only",
            "collection": CLEANED_COLLECTION,
            "filter": {"section": "body"},
            "primary": False,
        },
        {
            "config": "v2_references_only",
            "collection": CLEANED_COLLECTION,
            "filter": {"section": "references"},
            "primary": False,
        },
    ]


def read_chunk_index(store: VectorStore) -> dict[str, dict[str, Any]]:
    """Read one collection's chunk locations. Read-only."""
    data = store.collection.get(include=["metadatas"])
    chunks: dict[str, dict[str, Any]] = {}
    for chunk_id, metadata in zip(data["ids"], data["metadatas"]):
        chunks[chunk_id] = {
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "start": metadata["char_start"],
            "end": metadata["char_end"],
            "section": metadata.get("section"),
        }
    return chunks


def gold_chunks(
    chunk_index: Mapping[str, Mapping[str, Any]],
    passages: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> set[str]:
    """Collection-specific gold chunk IDs under the relevance rule."""
    return {
        chunk_id
        for chunk_id, chunk in chunk_index.items()
        if is_relevant(chunk, passages, threshold=threshold)
    }


def run_queries(
    retriever: Retriever,
    questions: Sequence[Question],
    *,
    top_k: int,
    where: Mapping[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve ``top_k`` results per question with identical settings."""
    return {
        question["id"]: retriever.retrieve(
            question["text"], top_k=top_k, where=where
        )
        for question in questions
    }


def retrieved_rows(
    *,
    config: Mapping[str, Any],
    question: Question,
    k: int,
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten one query's results into CSV/JSON rows."""
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        rows.append(
            {
                "config": config["config"],
                "collection": config["collection"],
                "filter": config["filter"],
                "question_id": question["id"],
                "question": question["text"],
                "k": k,
                "rank": rank,
                "chunk_id": result["id"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "section": metadata.get("section"),
                "char_start": metadata["char_start"],
                "char_end": metadata["char_end"],
                "similarity": round(float(result["similarity"]), 6),
                "distance": round(float(result["distance"]), 6),
                "token_count": metadata.get("token_count"),
                "is_truncated": metadata.get("is_truncated"),
                "text": result["text"],
            }
        )
    return rows


def reference_ranges(
    pages: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, int]]:
    """Reference-section ranges ``(page, start, end)`` from the segmenter.

    Reuses the Stage 6B segmentation rather than hard-coding page 5 / 2402, so
    the error analysis stays valid if the paper changes.
    """
    spans = SectionSegmenter().segment(pages)  # type: ignore[arg-type]
    return [
        (span["page_number"], span["start"], span["end"])
        for span in spans
        if span["section"] == REFERENCES_SECTION
    ]


def _is_reference_row(
    row: Mapping[str, Any], ranges: Sequence[tuple[int, int, int]]
) -> bool:
    """Row is mostly inside the references section."""
    length = row["char_end"] - row["char_start"]
    if length <= 0:
        return False
    for page, start, end in ranges:
        if row["page_number"] != page:
            continue
        shared = max(0, min(row["char_end"], end) - max(row["char_start"], start))
        if shared * 2 >= length:
            return True
    return False


def diagnostics(
    rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    *,
    headline_k: int,
    references: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    """Explain *why* the rankings differ: furniture, references, boundaries."""
    boundaries = [(page, start) for page, start, _ in references if start > 0]
    per_config: dict[str, Any] = {}

    for config in configs:
        label = config["config"]
        subset = [
            row
            for row in rows
            if row["config"] == label and row["k"] == headline_k
        ]
        per_config[label] = {
            "slots": len(subset),
            "furniture_slots": sum(
                1
                for row in subset
                if any(marker in row["text"] for marker in FURNITURE_MARKERS)
            ),
            "reference_slots": sum(
                1 for row in subset if _is_reference_row(row, references)
            ),
            "truncated_slots": sum(
                1 for row in subset if row.get("is_truncated") is True
            ),
            "boundary_slots": sum(
                1
                for row in subset
                if any(
                    row["page_number"] == page and row["char_start"] < point < row["char_end"]
                    for page, point in boundaries
                )
            ),
            "distinct_chunks": len(
                {(row["page_number"], row["chunk_index"]) for row in subset}
            ),
            "pages": sorted({row["page_number"] for row in subset}),
        }

    return {
        "headline_k": headline_k,
        "references_ranges": [list(item) for item in references],
        "boundaries": [list(item) for item in boundaries],
        "configs": per_config,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the machine-readable report as readable Markdown."""
    meta = report["meta"]
    k_values = meta["k_values"]
    lines: list[str] = [
        "# Stage 6D - Retrieval evaluation",
        "",
        f"- Generated: `{meta['generated_at']}`",
        f"- Embedding model: `{meta['embedding_model']}`",
        f"- Metric: {meta['metric']}",
        f"- k values: {k_values}",
        f"- Relevance rule: {meta['relevance_rule']}",
        "",
        "## Collection counts",
        "",
        "| collection | before | after |",
        "|---|---:|---:|",
    ]
    for name in report["counts_before"]:
        lines.append(
            f"| `{name}` | {report['counts_before'][name]} | "
            f"{report['counts_after'][name]} |"
        )

    lines += [
        "",
        "## Macro averages over the six questions",
        "",
        "P = precision@k, R = recall@k, MRR = mean reciprocal rank, "
        "PR = passage recall@k (0.7 coverage).",
        "",
        "| config | k | P | R | MRR | PR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}"

    for config in report["configs"]:
        label = config["config"]
        for k in k_values:
            values = report["macro"][label][str(k)]
            lines.append(
                f"| `{label}` | {k} | {cell(values['precision'])} | "
                f"{cell(values['recall'])} | {cell(values['mrr'])} | "
                f"{cell(values['passage_recall'])} |"
            )

    lines += [
        "",
        "## Per question (k = %d)" % max(k_values),
        "",
        "| question | config | P | R | RR | PR | gold chunks | hits |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    headline = str(max(k_values))
    for question in report["questions"]:
        qid = question["id"]
        for label in ("v1_unfiltered", "v2_unfiltered"):
            values = report["metrics"][label][qid][headline]
            lines.append(
                f"| {qid} | `{label}` | {values['precision']:.2f} | "
                f"{values['recall']:.2f} | {values['reciprocal_rank']:.2f} | "
                f"{values['passage_recall']:.2f} | {values['gold_chunks']} | "
                f"{values['relevant_retrieved']} |"
            )

    lines += [
        "",
        "## Error-analysis diagnostics (k = %d)" % max(k_values),
        "",
        "| config | slots | furniture | references | truncated | boundary |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in report["configs"]:
        label = config["config"]
        data = report["diagnostics"]["configs"][label]
        lines.append(
            f"| `{label}` | {data['slots']} | {data['furniture_slots']} | "
            f"{data['reference_slots']} | {data['truncated_slots']} | "
            f"{data['boundary_slots']} |"
        )

    lines += [
        "",
        "## Sufficiency",
        "",
        "| question | label | note |",
        "|---|---|---|",
    ]
    for question in report["questions"]:
        lines.append(
            f"| {question['id']} | {question['sufficiency']} | "
            f"{question['note']} |"
        )

    lines += [
        "",
        "## Files",
        "",
        f"- JSON: `{report['outputs']['json']}`",
        f"- CSV: `{report['outputs']['csv']}`",
        "",
    ]
    return "\n".join(lines)


def evaluate(
    *,
    pdf_path: Path,
    output_dir: Path,
    k_values: Sequence[int] = K_VALUES,
    thresholds: Sequence[float] = THRESHOLD_SENSITIVITY,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full comparison and write JSON + CSV. Returns the report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Gold passages in cleaned-text coordinates ---------------------
    pages = PageCleaner().clean_documents(
        PDFLoader(pdf_path).load(skip_empty_pages=False)
    )
    gold = resolve_gold(pages)

    # ---- 2. Read-only access to both collections --------------------------
    stores = {
        BASELINE_COLLECTION: VectorStore(collection_name=BASELINE_COLLECTION),
        CLEANED_COLLECTION: VectorStore.for_cleaned_ingestion(),
    }
    counts_before = {
        name: store.get_collection_count() for name, store in stores.items()
    }
    chunk_indexes = {
        name: read_chunk_index(store) for name, store in stores.items()
    }
    for name, index in chunk_indexes.items():
        if len(index) != counts_before[name]:
            raise RuntimeError(
                f"{name}: read {len(index)} chunks but count() says "
                f"{counts_before[name]}."
            )

    # ---- 3. Identical retrieval settings for every configuration ----------
    model = EmbeddingModel()
    retrievers = {
        name: Retriever(model, store) for name, store in stores.items()
    }
    configs = collection_configs()

    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}

    for config in configs:
        name = config["collection"]
        retriever = retrievers[name]
        chunk_index = chunk_indexes[name]
        metrics[config["config"]] = {}
        sensitivity[config["config"]] = {}

        for question in QUESTIONS:
            passages = gold[question["id"]]
            primary_gold = gold_chunks(chunk_index, passages)
            rankings: dict[int, list[str]] = {}
            per_k: dict[str, Any] = {}

            for k in k_values:
                results = retriever.retrieve(
                    question["text"],
                    top_k=k,
                    where=config["filter"],
                )
                rows.extend(
                    retrieved_rows(
                        config=config, question=question, k=k, results=results
                    )
                )
                rankings[k] = [result["id"] for result in results]
                per_k[str(k)] = evaluate_ranking(
                    rankings[k],
                    primary_gold,
                    k=k,
                    passages=passages,
                    chunks=chunk_index,
                )

            metrics[config["config"]][question["id"]] = per_k

            # Threshold sensitivity reuses the same rankings. Re-querying would
            # only add I/O and could introduce ordering noise.
            for threshold in thresholds:
                key = str(threshold)
                gold_at_threshold = gold_chunks(
                    chunk_index, passages, threshold=threshold
                )
                sensitivity[config["config"]].setdefault(key, {})
                sensitivity[config["config"]][key][question["id"]] = {
                    "gold_chunks": len(gold_at_threshold),
                    "k": {
                        str(k): evaluate_ranking(
                            rankings[k], gold_at_threshold, k=k
                        )
                        for k in k_values
                    },
                }

    # ---- 3b. Macro averages for the threshold sensitivity table ------------
    sensitivity_macro: dict[str, Any] = {}
    for label, per_threshold in sensitivity.items():
        sensitivity_macro[label] = {}
        for threshold, per_question in per_threshold.items():
            sensitivity_macro[label][threshold] = {}
            for k in k_values:
                key = str(k)
                sensitivity_macro[label][threshold][key] = {
                    "precision": macro_average(
                        per_question[q["id"]]["k"][key]["precision"]
                        for q in QUESTIONS
                    ),
                    "recall": macro_average(
                        per_question[q["id"]]["k"][key]["recall"]
                        for q in QUESTIONS
                    ),
                }

    # ---- 4. Macro averages over the six questions -------------------------
    macro: dict[str, Any] = {}
    for config in configs:
        label = config["config"]
        macro[label] = {}
        for k in k_values:
            key = str(k)
            macro[label][key] = {
                "precision": macro_average(
                    metrics[label][q["id"]][key]["precision"]
                    for q in QUESTIONS
                ),
                "recall": macro_average(
                    metrics[label][q["id"]][key]["recall"] for q in QUESTIONS
                ),
                "mrr": macro_average(
                    metrics[label][q["id"]][key]["reciprocal_rank"]
                    for q in QUESTIONS
                ),
                "passage_recall": macro_average(
                    metrics[label][q["id"]][key]["passage_recall"]
                    for q in QUESTIONS
                ),
            }

    counts_after = {
        name: store.get_collection_count() for name, store in stores.items()
    }

    references = reference_ranges(pages)
    report: dict[str, Any] = {
        "meta": {
            "stage": "6D",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pdf": str(pdf_path),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "k_values": list(k_values),
            "primary_threshold": DEFAULT_THRESHOLD,
            "threshold_sensitivity": list(thresholds),
            "metric": "cosine similarity (Chroma 'cosine' space)",
            "relevance_rule": (
                "chunk overlaps >= 50% of itself OR >= 50% of a reviewed "
                "gold passage, on the same page, in cleaned-text coordinates"
            ),
        },
        "questions": QUESTIONS,
        "gold_passages": gold,
        "counts_before": counts_before,
        "counts_after": counts_after,
        "configs": configs,
        "metrics": metrics,
        "macro": macro,
        "sensitivity": sensitivity,
        "sensitivity_macro": sensitivity_macro,
        "diagnostics": diagnostics(
            rows,
            configs,
            headline_k=max(k_values),
            references=references,
        ),
        "rows": rows,
    }

    json_path = output_dir / "stage6d_results.json"
    csv_path = output_dir / "stage6d_retrieval.csv"
    markdown_path = output_dir / "stage6d_report.md"

    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    if verbose:
        print_report(report)
    return report


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.2f}"


def print_report(report: Mapping[str, Any]) -> None:
    """Print a compact, ASCII-only summary."""
    meta = report["meta"]
    print("=" * 78)
    print("STAGE 6D - RETRIEVAL EVALUATION (READ-ONLY)")
    print("=" * 78)
    print(f"model       : {meta['embedding_model']}")
    print(f"k values    : {meta['k_values']}")
    print(f"threshold   : {meta['primary_threshold']} (sensitivity {meta['threshold_sensitivity']})")
    print(
        f"counts      : before={report['counts_before']} "
        f"after={report['counts_after']}"
    )

    print("\nMACRO AVERAGES OVER 6 QUESTIONS (P = precision, R = recall)")
    header = f"{'config':<20} " + "  ".join(
        f"k={k} P/R/MRR/PR" for k in meta["k_values"]
    )
    print(header)
    for config in report["configs"]:
        label = config["config"]
        cells = []
        for k in meta["k_values"]:
            values = report["macro"][label][str(k)]
            cells.append(
                f"{_fmt(values['precision'])}/{_fmt(values['recall'])}/"
                f"{_fmt(values['mrr'])}/{_fmt(values['passage_recall'])}"
            )
        print(f"{label:<20} " + "  ".join(cells))

    headline_k = str(meta["k_values"][-1])
    print(f"\nPER-QUESTION (k={headline_k}): P / R / RR / passage-recall")
    print(f"{'question':<6} {'v1_unfiltered':<26} {'v2_unfiltered':<26}")
    for question in report["questions"]:
        qid = question["id"]
        cells = []
        for label in ("v1_unfiltered", "v2_unfiltered"):
            values = report["metrics"][label][qid][headline_k]
            cells.append(
                f"{_fmt(values['precision'])}/{_fmt(values['recall'])}/"
                f"{_fmt(values['reciprocal_rank'])}/"
                f"{_fmt(values['passage_recall'])}"
            )
        print(f"{qid:<6} {cells[0]:<26} {cells[1]:<26}")

    print("\nSUFFICIENCY")
    for question in report["questions"]:
        print(f"  {question['id']}: {question['sufficiency']:<10} {question['note'][:90]}")

    print("\nTHRESHOLD SENSITIVITY (macro P/R)")
    for config in report["configs"]:
        label = config["config"]
        parts = []
        for threshold, per_k in report["sensitivity_macro"][label].items():
            for k, values in per_k.items():
                parts.append(
                    f"t={threshold},k={k}:{_fmt(values['precision'])}/"
                    f"{_fmt(values['recall'])}"
                )
        print(f"  {label:<20} " + "  ".join(parts))

    print("\nDIAGNOSTICS AT k=%d (slots = questions x k)" % report["diagnostics"]["headline_k"])
    print(f"  {'config':<20} {'slots':>6} {'furniture':>10} {'references':>11} {'truncated':>10}")
    for config in report["configs"]:
        data = report["diagnostics"]["configs"][config["config"]]
        print(
            f"  {config['config']:<20} {data['slots']:>6} "
            f"{data['furniture_slots']:>10} {data['reference_slots']:>11} "
            f"{data['truncated_slots']:>10}"
        )

    print(f"\nwrote {report['outputs']['json']}")
    print(f"wrote {report['outputs']['csv']}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = list(sys.argv[1:] if argv is None else argv)
    pdf_path = Path(args[0]) if args else DEFAULT_PDF
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    report = evaluate(pdf_path=pdf_path, output_dir=DEFAULT_OUTPUT_DIR)

    problems = [
        f"{name}: expected {expected}, found {report['counts_after'].get(name)}"
        for name, expected in EXPECTED_COUNTS.items()
        if report["counts_after"].get(name) != expected
    ]
    if report["counts_before"] != report["counts_after"]:
        problems.append("collection counts changed during evaluation")

    if problems:
        print("\nDATABASE SAFETY CHECK FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("\nDATABASE SAFETY CHECK PASSED (counts unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
