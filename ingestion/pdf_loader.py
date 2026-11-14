"""Load PDF files into page-level text documents.

This module is intentionally small: it reads a PDF with ``pypdf`` and returns
a plain list of dictionaries, one per page, in a shape that is easy to feed
into a later chunking / embedding step::

    [
        {
            "text": "Text from page 1 ...",
            "metadata": {
                "source": "report.pdf",
                "source_path": "C:/data/report.pdf",
                "page_number": 1,
                "total_pages": 10,
            },
        },
        ...
    ]
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


class DocumentMetadata(TypedDict):
    """Metadata attached to every extracted page."""

    source: str  # File name only, e.g. "report.pdf"
    source_path: str  # Full path passed by the caller
    page_number: int  # 1-based page number
    total_pages: int  # Total number of pages in the PDF


class Document(TypedDict):
    """A single page of extracted text plus its metadata."""

    text: str
    metadata: DocumentMetadata


class PDFLoader:
    """Extract text from a PDF, one page at a time.

    Args:
        pdf_path: Path to a ``.pdf`` file (string or ``Path``).

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The path is not a file, does not end in ``.pdf``,
            is encrypted, or cannot be parsed.
    """

    ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

    def __init__(self, pdf_path: str | Path) -> None:
        # Store a proper Path object so we can use .exists(), .suffix, etc.
        self.pdf_path = Path(pdf_path).expanduser()
        self._validate_path()

    def _validate_path(self) -> None:
        """Check that the path points to an existing .pdf file."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        if not self.pdf_path.is_file():
            raise ValueError(f"Expected a file but got a directory: {self.pdf_path}")

        # .suffix includes the dot, e.g. ".pdf". Compare case-insensitively
        # so "REPORT.PDF" is accepted too.
        if self.pdf_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(self.ALLOWED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type {self.pdf_path.suffix!r} for "
                f"{self.pdf_path.name!r}. Expected one of: {allowed}"
            )

    def _open_reader(self) -> PdfReader:
        """Open the PDF and make sure it is readable and not locked."""
        try:
            reader = PdfReader(str(self.pdf_path))
        except Exception as exc:
            # pypdf can raise many different errors for broken files, so we
            # wrap them in one clear message for the caller.
            raise ValueError(f"Could not open PDF: {self.pdf_path}") from exc

        if reader.is_encrypted:
            # Some PDFs are "encrypted" with an empty password (e.g. permissions
            # only). Try the empty password before giving up.
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0

            if not unlocked:
                raise ValueError(
                    f"PDF is password-protected and cannot be read: {self.pdf_path}"
                )

        return reader

    def load(self, skip_empty_pages: bool = True) -> list[Document]:
        """Extract text from every page of the PDF.

        Args:
            skip_empty_pages: When ``True`` (the default), pages with no
                extractable text (for example scanned images) are left out.
                Set to ``False`` to keep them with ``text=""`` so page
                numbers stay consistent for the reader.

        Returns:
            A list of ``Document`` dictionaries, one per kept page, in order.
        """
        reader = self._open_reader()
        total_pages = len(reader.pages)

        source_name = self.pdf_path.name
        source_path = str(self.pdf_path)

        documents: list[Document] = []

        # enumerate(..., start=1) gives us the human-friendly 1-based page number.
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                # extract_text() returns None for pages with no text layer.
                raw_text = page.extract_text() or ""
            except Exception:
                # A single broken page should not stop the whole document.
                raw_text = ""

            text = raw_text.strip()

            # Empty pages are common in scanned PDFs; skip them unless asked not to.
            if not text and skip_empty_pages:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": source_name,
                        "source_path": source_path,
                        "page_number": page_number,
                        "total_pages": total_pages,
                    },
                }
            )

        return documents


def load_pdf(pdf_path: str | Path, skip_empty_pages: bool = True) -> list[Document]:
    """Shortcut for ``PDFLoader(pdf_path).load(skip_empty_pages)``."""
    return PDFLoader(pdf_path).load(skip_empty_pages=skip_empty_pages)
