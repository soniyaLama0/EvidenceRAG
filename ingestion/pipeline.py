"""End-to-end ingestion pipeline for EvidenceRAG.

The pipeline wires the existing components together without re-implementing
any of their logic::

    PDF -> PDFLoader -> PageCleaner -> SectionSegmenter -> TextChunker
        -> Section assignment -> EmbeddingModel -> VectorStore

Design rules:

* Page order and page numbers come from ``PDFLoader`` and are never rewritten.
* Chunk ``char_start`` / ``char_end`` refer to **cleaned** page text, because
  ``TextChunker`` runs on the output of ``PageCleaner``. The pipeline verifies
  ``cleaned_page_text[char_start:char_end] == chunk_text`` before embedding.
* Every chunk carries ``section`` and ``cleaning_version`` metadata. The
  reference list is retained (never deleted) so it stays available for citation
  lookup, while retrieval can filter on ``section``.
* Ingestion is repeatable: ``VectorStore`` derives deterministic IDs from the
  resolved source path, page number and chunk index, and upserts them, so
  running the same source twice does not create duplicate records.
* Cleaning and segmentation fail open inside their own modules: an uncertain
  footer block is kept, and a document without a confirmed references heading
  is treated entirely as body. The pipeline never drops pages for being empty
  or unreadable; it reports them instead.

This module intentionally stops before retrieval, reranking and answer
generation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict, cast

from ingestion.embedding_model import EmbeddedChunk, EmbeddingModel
from ingestion.page_cleaner import (
    CLEANING_VERSION,
    CleanedDocument,
    PageCleaner,
)
from ingestion.pdf_loader import Document, PDFLoader
from ingestion.section_segmenter import (
    BODY_SECTION,
    REFERENCES_SECTION,
    SectionSegmenter,
    SectionSpan,
)
from ingestion.text_chunker import Chunk, TextChunker
from retrieval.vector_store import VectorStore

LoaderFactory = Callable[[str | Path], PDFLoader]


class PipelineResult(TypedDict):
    """Everything produced by one ingestion run, including diagnostics.

    ``raw_*`` values describe ``PDFLoader`` output; ``cleaned_*`` values
    describe the pages actually chunked. Character offsets in ``chunks`` and
    ``embedded_chunks`` refer to ``documents[*]["text"]`` (cleaned pages).
    """

    source: str
    raw_page_count: int
    cleaned_page_count: int
    empty_page_numbers: list[int]
    raw_char_count: int
    cleaned_char_count: int
    chunk_count: int
    embedded_count: int
    submitted_count: int
    collection_count: int
    body_chunk_count: int
    reference_chunk_count: int
    documents: list[CleanedDocument]
    spans: list[SectionSpan]
    chunks: list[Chunk]
    embedded_chunks: list[EmbeddedChunk]


class IngestionPipeline:
    """Run the complete cleaned ingestion pipeline.

    Args:
        embedding_model: Existing ``EmbeddingModel`` (or a compatible object).
        vector_store: Target store. Use
            ``VectorStore.for_cleaned_ingestion()`` for ``evidence_chunks_v2``.
        cleaner: Optional ``PageCleaner`` replacement.
        segmenter: Optional ``SectionSegmenter`` replacement.
        chunker: Optional ``TextChunker`` replacement. The default keeps the
            project's existing 1000-character / 200-overlap strategy.
        loader_factory: Optional ``PDFLoader`` factory, used by tests to inject
            synthetic pages without touching the filesystem.

    All components are injected for testability; defaults are the real ones.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        *,
        cleaner: PageCleaner | None = None,
        segmenter: SectionSegmenter | None = None,
        chunker: TextChunker | None = None,
        loader_factory: LoaderFactory = PDFLoader,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.cleaner = cleaner if cleaner is not None else PageCleaner()
        self.segmenter = segmenter if segmenter is not None else SectionSegmenter()
        self.chunker = chunker if chunker is not None else TextChunker()
        self.loader_factory = loader_factory

    def run(self, pdf_path: str | Path) -> PipelineResult:
        """Ingest one PDF and return a diagnostic result.

        Raises:
            TypeError: If ``pdf_path`` is not a string or ``Path``.
            FileNotFoundError: If the PDF does not exist (from ``PDFLoader``).
            ValueError: If the file is not a readable PDF, has no pages, or has
                no extractable text (for example a scanned image PDF).
        """
        if not isinstance(pdf_path, (str, Path)):
            raise TypeError(
                f"pdf_path must be a string or Path, got {type(pdf_path).__name__}."
            )

        loader = self.loader_factory(pdf_path)
        raw_documents = loader.load(skip_empty_pages=False)

        if not raw_documents:
            raise ValueError(f"PDF contains no pages: {pdf_path}")

        if not any(document["text"].strip() for document in raw_documents):
            raise ValueError(
                f"PDF contains no extractable text (it may be scanned images): "
                f"{pdf_path}"
            )

        # 1. Clean page furniture. Pages are kept even when they become empty,
        #    so page numbers and offsets stay aligned with the PDF.
        cleaned_documents = self.cleaner.clean_documents(raw_documents)

        if not any(document["text"].strip() for document in cleaned_documents):
            raise ValueError(
                f"No ingestible text remains after cleaning: {pdf_path}"
            )

        # 2. Detect body/references spans over the cleaned text.
        spans = self.segmenter.segment(cleaned_documents)  # type: ignore[arg-type]

        # 3. Chunk the cleaned pages with the existing strategy.
        chunks = self.chunker.chunk_documents(cleaned_documents)  # type: ignore[arg-type]
        if not chunks:
            raise ValueError(
                f"Page text produced no chunks during ingestion: {pdf_path}"
            )

        # 4. Attach section + cleaning_version to every chunk.
        labelled = cast(
            list[Chunk],
            self.segmenter.assign_to_chunks(chunks, spans),
        )
        self._verify_sections(labelled)
        self._verify_offsets(cleaned_documents, labelled)

        # 5. Embed with the existing model, preserving text and metadata.
        embedded = self.embedding_model.embed_chunks(labelled)
        self._verify_embedded(labelled, embedded)

        # 6. Upsert into the target collection.
        submitted = self.vector_store.add_chunks(embedded)
        if submitted != len(embedded):
            raise RuntimeError(
                f"VectorStore accepted {submitted} of {len(embedded)} chunks."
            )

        body_count = sum(
            1 for chunk in labelled if chunk["metadata"]["section"] == BODY_SECTION
        )
        reference_count = sum(
            1
            for chunk in labelled
            if chunk["metadata"]["section"] == REFERENCES_SECTION
        )

        return {
            "source": raw_documents[0]["metadata"]["source"],
            "raw_page_count": len(raw_documents),
            "cleaned_page_count": sum(
                1
                for document in cleaned_documents
                if document["text"].strip()
            ),
            "empty_page_numbers": [
                document["metadata"]["page_number"]
                for document in cleaned_documents
                if not document["text"].strip()
            ],
            "raw_char_count": sum(
                len(document["text"]) for document in raw_documents
            ),
            "cleaned_char_count": sum(
                len(document["text"]) for document in cleaned_documents
            ),
            "chunk_count": len(labelled),
            "embedded_count": len(embedded),
            "submitted_count": submitted,
            "collection_count": self.vector_store.get_collection_count(),
            "body_chunk_count": body_count,
            "reference_chunk_count": reference_count,
            "documents": cleaned_documents,
            "spans": spans,
            "chunks": labelled,
            "embedded_chunks": embedded,
        }

    @staticmethod
    def _verify_sections(chunks: Sequence[Chunk]) -> None:
        """Every chunk must carry a known section and the current version."""
        for index, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            section = metadata.get("section")
            if section not in (BODY_SECTION, REFERENCES_SECTION):
                raise ValueError(
                    f"chunks[{index}] has invalid section metadata: {section!r}."
                )
            if metadata.get("cleaning_version") != CLEANING_VERSION:
                raise ValueError(
                    f"chunks[{index}] has cleaning_version "
                    f"{metadata.get('cleaning_version')!r}; expected "
                    f"{CLEANING_VERSION!r}."
                )

    @staticmethod
    def _verify_offsets(
        documents: Sequence[CleanedDocument],
        chunks: Sequence[Chunk],
    ) -> None:
        """Guarantee chunk offsets address the cleaned page text exactly."""
        page_text = {
            document["metadata"]["page_number"]: document["text"]
            for document in documents
        }

        for index, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            page_number = metadata["page_number"]
            start = metadata["char_start"]
            end = metadata["char_end"]

            text = page_text.get(page_number)
            if text is None:
                raise ValueError(
                    f"chunks[{index}] refers to page {page_number}, "
                    "which is missing from the cleaned documents."
                )
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"chunks[{index}] has non-integer offsets.")
            if start < 0 or end < start or end > len(text):
                raise ValueError(
                    f"chunks[{index}] offsets [{start}, {end}) fall outside "
                    f"cleaned page {page_number} (length {len(text)})."
                )
            if text[start:end] != chunk["text"]:
                raise ValueError(
                    f"chunks[{index}] text does not match cleaned page "
                    f"{page_number}[{start}:{end}]."
                )

    @staticmethod
    def _verify_embedded(
        chunks: Sequence[Chunk],
        embedded: Sequence[EmbeddedChunk],
    ) -> None:
        """Embedding must preserve text, section and cleaning version."""
        if len(embedded) != len(chunks):
            raise RuntimeError(
                f"Embedding returned {len(embedded)} chunks for "
                f"{len(chunks)} inputs."
            )
        for index, (source, output) in enumerate(zip(chunks, embedded)):
            if output["text"] != source["text"]:
                raise ValueError(
                    f"embedded_chunks[{index}] text differs from its input chunk."
                )
            for field in ("section", "cleaning_version"):
                if output["metadata"].get(field) != source["metadata"].get(field):
                    raise ValueError(
                        f"embedded_chunks[{index}] lost metadata field {field!r}."
                    )


def ingest_pdf(
    pdf_path: str | Path,
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
) -> PipelineResult:
    """Shortcut for ``IngestionPipeline(embedding_model, vector_store).run(path)``."""
    return IngestionPipeline(embedding_model, vector_store).run(pdf_path)
