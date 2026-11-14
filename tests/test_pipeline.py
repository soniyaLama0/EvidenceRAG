"""Tests for ``ingestion.pipeline``.

Most tests inject the loader, embedding model and (sometimes) the vector store,
so they run offline and without the real 384-d model. Two tests use a real
persistent Chroma client in ``tmp_path`` to prove upsert idempotence and that
``evidence_chunks_v2`` is independent of the baseline collection.

Run from the project root with::

    .\\.venv\\Scripts\\python.exe -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import chromadb
import pytest
from chromadb.config import Settings

from ingestion.embedding_model import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from ingestion.page_cleaner import CLEANING_VERSION, PageCleaner
from ingestion.pipeline import IngestionPipeline, ingest_pdf
from ingestion.section_segmenter import (
    BODY_SECTION,
    REFERENCES_SECTION,
    SectionSegmenter,
)
from ingestion.text_chunker import TextChunker
from retrieval.vector_store import CLEANED_COLLECTION_NAME, VectorStore

TOTAL_PAGES = 2


# ---------------------------------------------------------------------------
# Synthetic pages that mirror the real paper's furniture and references
# ---------------------------------------------------------------------------


def footer_lines(page_number: int) -> list[str]:
    return [
        "Published by : International Journal of Engineering Research & Technology (IJERT)",
        "https://www.ijert.org/ ISSN: 2278-0181",
        "An International Peer-Reviewed Journal Vol. 15 Issue 06 , June - 2026",
        f"IJERTV15IS061078 Page {page_number}",
        "(This work is licensed under a Creative Commons Attribution 4.0 International License.)",
    ]


def page_with_footer(body: str, page_number: int) -> str:
    return body + "\n" + "\n".join(footer_lines(page_number))


BODY_ONE = "Page one body about MSMEs and voice commands. " * 20
BODY_TWO = "Page two body before the references. " * 24
REFERENCE_BLOCK = "\n".join(
    [
        "XIII. REFERENCES",
        "[1]  First reference. Available:",
        "https://example.org/one",
        "[2]  Second reference.",
        "[3]  Third reference.",
    ]
)


def make_page(text: str, page_number: int, *, total_pages: int = TOTAL_PAGES) -> dict:
    return {
        "text": text,
        "metadata": {
            "source": "research_paper.pdf",
            "source_path": "C:/data/research_paper.pdf",
            "page_number": page_number,
            "total_pages": total_pages,
        },
    }


def make_pages() -> list[dict]:
    """Two pages: body only, then body + confirmed references."""
    return [
        make_page(page_with_footer(BODY_ONE, 1), 1),
        make_page(page_with_footer(BODY_TWO + "\n" + REFERENCE_BLOCK, 2), 2),
    ]


# ---------------------------------------------------------------------------
# Fakes and recording wrappers
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Offline stand-in for ``EmbeddingModel`` that keeps all metadata."""

    model_name = EMBEDDING_MODEL_NAME

    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log
        self.batches: list[list[dict]] = []

    def embed_chunks(self, chunks):
        if self.log is not None:
            self.log.append("embed")
        chunk_list = list(chunks)
        self.batches.append(copy.deepcopy(chunk_list))

        embedded = []
        for index, chunk in enumerate(chunk_list):
            metadata = dict(chunk["metadata"])
            metadata.update(
                {
                    "token_count": len(chunk["text"].split()),
                    "is_truncated": False,
                    "embedding_model": self.model_name,
                }
            )
            vector = [0.0] * EMBEDDING_DIMENSION
            vector[index % EMBEDDING_DIMENSION] = 1.0
            embedded.append(
                {
                    "text": chunk["text"],
                    "metadata": metadata,
                    "embedding": vector,
                }
            )
        return embedded


class FakeVectorStore:
    """Minimal store that records what the pipeline submitted."""

    collection_name = "fake_collection"

    def __init__(self, log: list[str] | None = None) -> None:
        self.log = log
        self.records: list[dict] = []

    def add_chunks(self, chunks) -> int:
        if self.log is not None:
            self.log.append("store")
        chunk_list = copy.deepcopy(list(chunks))
        self.records.extend(chunk_list)
        return len(chunk_list)

    def get_collection_count(self) -> int:
        return len(self.records)


def make_loader(pages: list[dict], log: list[str] | None = None):
    """Return a loader factory that serves ``pages`` and records the load."""

    class _Loader:
        def __init__(self, path: object) -> None:
            self.path = path
            if log is not None:
                log.append("load")

        def load(self, skip_empty_pages: bool = True) -> list[dict]:
            assert skip_empty_pages is False, "pipeline must keep empty pages"
            return copy.deepcopy(pages)

    return _Loader


class RecordingCleaner:
    def __init__(self, inner: PageCleaner, log: list[str]) -> None:
        self.inner = inner
        self.log = log

    def clean_documents(self, documents):
        self.log.append("clean")
        return self.inner.clean_documents(documents)


class RecordingSegmenter:
    def __init__(self, inner: SectionSegmenter, log: list[str]) -> None:
        self.inner = inner
        self.log = log

    def segment(self, documents):
        self.log.append("segment")
        return self.inner.segment(documents)

    def assign_to_chunks(self, chunks, spans):
        self.log.append("assign")
        return self.inner.assign_to_chunks(chunks, spans)


class RecordingChunker:
    def __init__(self, inner: TextChunker, log: list[str]) -> None:
        self.inner = inner
        self.log = log

    def chunk_documents(self, documents):
        self.log.append("chunk")
        return self.inner.chunk_documents(documents)


class TamperingChunker:
    """Returns a chunk whose text does not match its offsets."""

    def chunk_documents(self, documents):
        document = documents[0]
        return [
            {
                "text": "tampered text",
                "metadata": {
                    **document["metadata"],
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": 13,
                },
            }
        ]


# ---------------------------------------------------------------------------
# Pipeline order and happy path
# ---------------------------------------------------------------------------


def test_pipeline_execution_order() -> None:
    log: list[str] = []
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(log),
        FakeVectorStore(log),
        cleaner=RecordingCleaner(PageCleaner(), log),
        segmenter=RecordingSegmenter(SectionSegmenter(), log),
        chunker=RecordingChunker(TextChunker(), log),
        loader_factory=make_loader(make_pages(), log),
    )

    result = pipeline.run("research_paper.pdf")

    assert log == ["load", "clean", "segment", "chunk", "assign", "embed", "store"]
    assert result["chunk_count"] > 0


def test_pipeline_chunks_clean_text_and_preserves_metadata() -> None:
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        store,
        loader_factory=make_loader(make_pages()),
    )

    result = pipeline.run("research_paper.pdf")

    cleaned = {
        document["metadata"]["page_number"]: document["text"]
        for document in result["documents"]
    }
    assert all("Published by" not in text for text in cleaned.values())
    assert result["cleaned_char_count"] < result["raw_char_count"]
    assert result["raw_page_count"] == 2
    assert result["cleaned_page_count"] == 2
    assert result["empty_page_numbers"] == []
    assert result["chunk_count"] == result["embedded_count"] == result["submitted_count"]
    assert result["collection_count"] == len(store.records)

    seen_indexes: set[int] = set()
    for chunk in result["chunks"]:
        metadata = chunk["metadata"]
        page_number = metadata["page_number"]
        text = cleaned[page_number]

        assert metadata["source"] == "research_paper.pdf"
        assert metadata["source_path"] == "C:/data/research_paper.pdf"
        assert metadata["total_pages"] == TOTAL_PAGES
        assert metadata["section"] in (BODY_SECTION, REFERENCES_SECTION)
        assert metadata["cleaning_version"] == CLEANING_VERSION
        assert metadata["char_end"] <= len(text)
        assert text[metadata["char_start"] : metadata["char_end"]] == chunk["text"]
        seen_indexes.add(metadata["chunk_index"])

    assert seen_indexes == set(range(result["chunk_count"]))


def test_sections_are_assigned_to_body_and_references() -> None:
    store = FakeVectorStore()
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        store,
        loader_factory=make_loader(make_pages()),
    )

    result = pipeline.run("research_paper.pdf")

    assert result["body_chunk_count"] >= 1
    assert result["reference_chunk_count"] >= 1
    assert (
        result["body_chunk_count"] + result["reference_chunk_count"]
        == result["chunk_count"]
    )

    reference_chunks = [
        chunk
        for chunk in result["chunks"]
        if chunk["metadata"]["section"] == REFERENCES_SECTION
    ]
    assert reference_chunks
    assert all(chunk["metadata"]["page_number"] == 2 for chunk in reference_chunks)
    assert any("[1]" in chunk["text"] for chunk in reference_chunks)

    # Reference text is retained, not deleted.
    assert "XIII. REFERENCES" in result["documents"][1]["text"]
    assert "[3]  Third reference." in result["documents"][1]["text"]


def test_embedded_chunks_keep_section_metadata() -> None:
    model = FakeEmbeddingModel()
    pipeline = IngestionPipeline(
        model,
        FakeVectorStore(),
        loader_factory=make_loader(make_pages()),
    )

    result = pipeline.run("research_paper.pdf")

    assert result["embedded_chunks"]
    for chunk in result["embedded_chunks"]:
        assert chunk["metadata"]["section"] in (BODY_SECTION, REFERENCES_SECTION)
        assert chunk["metadata"]["cleaning_version"] == CLEANING_VERSION
        assert len(chunk["embedding"]) == EMBEDDING_DIMENSION
    assert [chunk["text"] for chunk in result["embedded_chunks"]] == [
        chunk["text"] for chunk in result["chunks"]
    ]


def test_ingest_pdf_shortcut(tmp_path: Path) -> None:
    """The module-level helper works with the real loader and default components."""
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "shortcut.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(612, 792))
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 720, "Shortcut pipeline body text about MSME voice billing. " * 12)
    pdf.drawString(72, 700, "[1]  A citation in the body must survive.")
    pdf.showPage()
    pdf.save()

    store = FakeVectorStore()
    result = ingest_pdf(pdf_path, FakeEmbeddingModel(), store)

    assert result["source"] == "shortcut.pdf"
    assert result["raw_page_count"] == result["cleaned_page_count"] == 1
    assert result["chunk_count"] >= 1
    assert result["chunk_count"] == result["submitted_count"] == len(store.records)
    assert "[1]  A citation in the body must survive." in result["documents"][0]["text"]
    assert {chunk["metadata"]["section"] for chunk in result["chunks"]} == {
        BODY_SECTION
    }


def test_tampered_offsets_are_rejected() -> None:
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        chunker=TamperingChunker(),  # type: ignore[arg-type]
        loader_factory=make_loader(make_pages()),
    )

    with pytest.raises(ValueError, match="does not match cleaned page"):
        pipeline.run("research_paper.pdf")


# ---------------------------------------------------------------------------
# Empty pages and invalid input
# ---------------------------------------------------------------------------


def test_empty_page_is_kept_but_produces_no_chunks() -> None:
    pages = make_pages()
    pages.append(make_page("", 3, total_pages=3))

    result = IngestionPipeline(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        loader_factory=make_loader(pages),
    ).run("research_paper.pdf")

    assert result["raw_page_count"] == 3
    assert result["cleaned_page_count"] == 2
    assert result["empty_page_numbers"] == [3]
    assert {chunk["metadata"]["page_number"] for chunk in result["chunks"]} == {1, 2}


def test_document_with_no_pages_is_rejected() -> None:
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        loader_factory=make_loader([]),
    )

    with pytest.raises(ValueError, match="no pages"):
        pipeline.run("empty.pdf")


def test_document_without_extractable_text_is_rejected() -> None:
    pages = [make_page("   \n\n", 1, total_pages=1)]
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        loader_factory=make_loader(pages),
    )

    with pytest.raises(ValueError, match="no extractable text"):
        pipeline.run("scanned.pdf")


def test_page_containing_only_the_footer_is_rejected() -> None:
    pages = [
        {
            "text": "\n".join(footer_lines(1)),
            "metadata": {
                "source": "furniture.pdf",
                "source_path": "C:/data/furniture.pdf",
                "page_number": 1,
                "total_pages": 1,
            },
        }
    ]
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        FakeVectorStore(),
        loader_factory=make_loader(pages),
    )

    with pytest.raises(ValueError, match="after cleaning"):
        pipeline.run("furniture.pdf")


def test_missing_pdf_is_rejected() -> None:
    pipeline = IngestionPipeline(FakeEmbeddingModel(), FakeVectorStore())

    with pytest.raises(FileNotFoundError):
        pipeline.run("does_not_exist.pdf")


def test_non_pdf_file_is_rejected(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a pdf", encoding="utf-8")
    pipeline = IngestionPipeline(FakeEmbeddingModel(), FakeVectorStore())

    with pytest.raises(ValueError, match="Unsupported file type"):
        pipeline.run(text_file)


def test_non_path_input_is_rejected() -> None:
    pipeline = IngestionPipeline(FakeEmbeddingModel(), FakeVectorStore())

    with pytest.raises(TypeError, match="pdf_path"):
        pipeline.run(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Repeat ingestion and v2 independence (real Chroma in tmp_path)
# ---------------------------------------------------------------------------


def make_client(tmp_path: Path):
    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=Settings(anonymized_telemetry=False),
    )


def test_repeat_ingestion_does_not_duplicate_records(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    store = VectorStore.for_cleaned_ingestion(client=client)
    pipeline = IngestionPipeline(
        FakeEmbeddingModel(),
        store,
        loader_factory=make_loader(make_pages()),
    )

    first = pipeline.run("research_paper.pdf")
    second = pipeline.run("research_paper.pdf")

    assert first["chunk_count"] == second["chunk_count"] == 3
    assert first["submitted_count"] == second["submitted_count"] == 3
    assert first["collection_count"] == 3
    assert second["collection_count"] == 3
    assert store.get_collection_count() == 3

    records = store.collection.get()
    assert len(records["ids"]) == 3
    assert len(set(records["ids"])) == 3


def test_cleaned_collection_is_independent_of_baseline(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    baseline = VectorStore(client=client, collection_name="evidence_chunks")
    baseline_record = {
        "text": "legacy baseline chunk",
        "metadata": {
            "source": "legacy.pdf",
            "source_path": "C:/data/legacy.pdf",
            "page_number": 1,
            "total_pages": 1,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 19,
            "token_count": 3,
            "is_truncated": False,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
        "embedding": [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
    }
    assert baseline.add_chunks([baseline_record]) == 1

    cleaned_store = VectorStore.for_cleaned_ingestion(client=client)
    assert cleaned_store.collection_name == CLEANED_COLLECTION_NAME
    assert baseline.collection_name == "evidence_chunks"

    result = IngestionPipeline(
        FakeEmbeddingModel(),
        cleaned_store,
        loader_factory=make_loader(make_pages()),
    ).run("research_paper.pdf")

    assert cleaned_store.get_collection_count() == result["chunk_count"]
    assert baseline.get_collection_count() == 1

    old_record = baseline.collection.get(ids=[baseline.collection.get()["ids"][0]])
    assert old_record["documents"] == ["legacy baseline chunk"]
    assert old_record["metadatas"][0]["source"] == "legacy.pdf"


def test_cleaned_collection_name_constant() -> None:
    assert CLEANED_COLLECTION_NAME == "evidence_chunks_v2"
    assert CLEANED_COLLECTION_NAME != VectorStore(
        client=chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        ),
        collection_name=f"probe_{uuid.uuid4().hex}",
    ).collection_name
