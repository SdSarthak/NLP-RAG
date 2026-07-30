"""Document loading and chunking.

Supported inputs: plain text, Markdown, and PDF files (via ``pypdf`` / ``PyPDF2``),
individual paths or whole directories.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
HEADING_RE = re.compile(r"^(#{1,6}\s+\S.*)$", re.MULTILINE)
_DOT_LEADER_RE = re.compile(r"\.{4,}")
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# A chunk this sparse in real words is page furniture (contents pages, figure
# axes, reference numbering) rather than content worth retrieving.
MIN_ALPHA_RATIO = 0.45
MIN_WORDS = 2


class DocumentError(RuntimeError):
    """Raised when a document cannot be loaded."""


@dataclass
class RawDocument:
    """A source document before chunking."""

    source: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """An indexable piece of a document."""

    id: str
    text: str
    source: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "index": self.index,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            source=data.get("source", "unknown"),
            index=int(data.get("index", 0)),
            metadata=dict(data.get("metadata") or {}),
        )


# ----------------------------------------------------------------------
# Text normalisation and splitting
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Collapse horizontal whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _DOT_LEADER_RE.sub(" ", text)  # table-of-contents dot leaders
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _split_units(text: str) -> List[str]:
    """Split text into sentence-sized units, respecting paragraph breaks."""
    units: List[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.replace("\n", " ").strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def _hard_split(unit: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split an oversized unit into fixed-size overlapping windows."""
    step = max(1, chunk_size - chunk_overlap)
    pieces = []
    for start in range(0, len(unit), step):
        piece = unit[start : start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        if start + chunk_size >= len(unit):
            break
    return pieces


def _overlap_tail(text: str, chunk_overlap: int) -> str:
    """Return the trailing ``chunk_overlap`` characters, snapped to a word boundary."""
    if chunk_overlap <= 0 or not text:
        return ""
    tail = text[-chunk_overlap:]
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def _pack_units(units: Sequence[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack sentence units into overlapping chunks."""
    expanded: List[str] = []
    for unit in units:
        if len(unit) > chunk_size:
            expanded.extend(_hard_split(unit, chunk_size, chunk_overlap))
        else:
            expanded.append(unit)

    chunks: List[str] = []
    buffer = ""
    for unit in expanded:
        if not buffer:
            buffer = unit
            continue
        if len(buffer) + 1 + len(unit) <= chunk_size:
            buffer = f"{buffer} {unit}"
            continue

        chunks.append(buffer)
        tail = _overlap_tail(buffer, chunk_overlap)
        candidate = f"{tail} {unit}".strip() if tail else unit
        buffer = candidate if len(candidate) <= chunk_size else unit

    if buffer:
        chunks.append(buffer)
    return chunks


def split_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """Split Markdown text into ``(heading, body)`` sections.

    Text before the first heading is returned with a ``None`` heading.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections: List[Tuple[Optional[str], str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        sections.append((match.group(1).strip(), body))
    return sections


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    keep_headings: bool = True,
) -> List[str]:
    """Split ``text`` into overlapping chunks of at most ``chunk_size`` characters.

    Markdown sections are chunked independently and each chunk is prefixed with
    its section heading, so a chunk always carries the context of where it came
    from. Chunk boundaries otherwise prefer paragraph then sentence breaks; only
    units longer than ``chunk_size`` are split mid-sentence.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []

    chunks: List[str] = []
    for heading, body in split_sections(text):
        prefix = ""
        budget = chunk_size
        if heading and keep_headings:
            candidate_prefix = f"{heading}\n"
            if len(candidate_prefix) < chunk_size // 2:
                prefix = candidate_prefix
                budget = chunk_size - len(prefix)

        units = _split_units(body)
        if not units:
            if heading:
                chunks.append(heading)
            continue

        for piece in _pack_units(units, budget, min(chunk_overlap, budget - 1)):
            chunks.append(prefix + piece)
    return chunks


def is_meaningful(text: str) -> bool:
    """Reject page furniture: contents lines, figure axes, bare numbering."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(_WORD_RE.findall(stripped)) < MIN_WORDS:
        return False
    letters = sum(1 for character in stripped if character.isalpha())
    return letters / len(stripped) >= MIN_ALPHA_RATIO


def chunk_documents(
    documents: Sequence[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    drop_noise: bool = True,
) -> List[Chunk]:
    """Convert raw documents into indexable chunks with stable ids."""
    chunks: List[Chunk] = []
    dropped = 0
    for document in documents:
        pieces = split_text(document.text, chunk_size, chunk_overlap)
        index = 0
        for piece in pieces:
            if drop_noise and not is_meaningful(piece):
                dropped += 1
                continue
            chunks.append(
                Chunk(
                    id=f"{document.source}::{index}",
                    text=piece,
                    source=document.source,
                    index=index,
                    metadata=dict(document.metadata),
                )
            )
            index += 1
    logger.info(
        "Chunked %d document(s) into %d chunk(s) (%d low-content chunk(s) dropped)",
        len(documents),
        len(chunks),
        dropped,
    )
    return chunks


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_text_file(path: Path) -> RawDocument:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return RawDocument(
        source=str(path),
        text=text,
        metadata={"format": path.suffix.lstrip(".") or "txt"},
    )


def _pdf_reader_class():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentError(
            "Reading PDFs requires 'pypdf' (pip install pypdf)."
        ) from exc


def load_pdf(path: Path) -> RawDocument:
    """Extract text from a PDF, page by page."""
    path = Path(path)
    reader_cls = _pdf_reader_class()
    try:
        reader = reader_cls(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Failed to read PDF {path}: {exc}") from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        logger.warning(
            "No extractable text in %s - it may be a scanned document", path
        )
    return RawDocument(
        source=str(path),
        text=text,
        metadata={"format": "pdf", "pages": len(pages)},
    )


def load_document(path: Path | str) -> RawDocument:
    """Load a single file, dispatching on its extension."""
    path = Path(path)
    if not path.exists():
        raise DocumentError(f"No such file: {path}")
    if not path.is_file():
        raise DocumentError(f"Not a file: {path}")

    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf(path)
    if suffix in TEXT_SUFFIXES or suffix == "":
        return load_text_file(path)
    raise DocumentError(
        f"Unsupported file type {suffix!r}. Supported: "
        f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def load_directory(path: Path | str, recursive: bool = True) -> List[RawDocument]:
    """Load every supported file in a directory."""
    path = Path(path)
    if not path.is_dir():
        raise DocumentError(f"Not a directory: {path}")

    pattern = "**/*" if recursive else "*"
    documents: List[RawDocument] = []
    for candidate in sorted(path.glob(pattern)):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            documents.append(load_document(candidate))
        except DocumentError as exc:
            logger.warning("Skipping %s: %s", candidate, exc)
    return documents


def load_paths(paths: Iterable[Path | str], recursive: bool = True) -> List[RawDocument]:
    """Load a mix of files and directories."""
    documents: List[RawDocument] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            documents.extend(load_directory(path, recursive=recursive))
        else:
            documents.append(load_document(path))
    return documents
