"""Vector stores.

Indexes are persisted in a portable format (``vectors.npy`` + ``chunks.jsonl`` +
``meta.json``) so the same directory can be opened with either the NumPy or the
FAISS backend.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


class VectorStoreError(RuntimeError):
    """Raised when an index cannot be built, saved or loaded."""


class VectorStore(ABC):
    """Similarity search over dense vectors, with the chunks attached."""

    backend: str = "base"

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @abstractmethod
    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        """Add vectors and their chunks to the index."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        """Return the ``top_k`` most similar chunks with cosine scores."""

    @abstractmethod
    def vectors(self) -> np.ndarray:
        """Return all stored vectors as a ``(len(self), dim)`` matrix."""

    # ------------------------------------------------------------------
    def _validate(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise VectorStoreError("vectors must be a 2-D array")
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                f"got {vectors.shape[0]} vectors for {len(chunks)} chunks"
            )
        if vectors.shape[1] != self.dim:
            raise VectorStoreError(
                f"expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )
        return vectors

    # ------------------------------------------------------------------
    def save(self, directory: Path | str, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Persist the index to ``directory`` and return that path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / VECTORS_FILE, self.vectors())
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        meta: Dict[str, Any] = {
            "backend": self.backend,
            "dim": self.dim,
            "count": len(self.chunks),
        }
        if extra:
            meta.update(extra)
        (directory / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Saved index with %d chunk(s) to %s", len(self.chunks), directory)
        return directory


class NumpyVectorStore(VectorStore):
    """Exact brute-force cosine search. No extra dependencies."""

    backend = "numpy"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        vectors = self._validate(vectors, chunks)
        if not len(chunks):
            return
        self._matrix = np.vstack([self._matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if not self.chunks or top_k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise VectorStoreError(
                f"query has dim {query.shape[0]}, index has dim {self.dim}"
            )
        scores = self._matrix @ query
        limit = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in top]

    def vectors(self) -> np.ndarray:
        return self._matrix


class FaissVectorStore(VectorStore):
    """FAISS inner-product index (equivalent to cosine on normalised vectors)."""

    backend = "faiss"

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "faiss is not installed or failed to load. "
                "Run `pip install faiss-cpu` or use the numpy backend."
            ) from exc
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        vectors = self._validate(vectors, chunks)
        if not len(chunks):
            return
        self._index.add(vectors)
        self._matrix = np.vstack([self._matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if not self.chunks or top_k <= 0:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise VectorStoreError(
                f"query has dim {query.shape[1]}, index has dim {self.dim}"
            )
        scores, indices = self._index.search(query, min(top_k, len(self.chunks)))
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results

    def vectors(self) -> np.ndarray:
        return self._matrix


def get_vector_store(config: RAGConfig, dim: int) -> VectorStore:
    """Instantiate the vector store described by ``config``."""
    backend = config.vector_backend

    if backend in {"auto", "faiss"}:
        try:
            store = FaissVectorStore(dim)
            logger.info("Using FAISS vector store (dim=%d)", dim)
            return store
        except Exception as exc:
            if backend == "faiss":
                raise
            logger.info("FAISS unavailable (%s); using NumPy vector store", exc)

    logger.info("Using NumPy vector store (dim=%d)", dim)
    return NumpyVectorStore(dim)


def read_index_meta(directory: Path | str) -> Dict[str, Any]:
    directory = Path(directory)
    meta_path = directory / META_FILE
    if not meta_path.exists():
        raise VectorStoreError(f"No index metadata found at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_vector_store(
    directory: Path | str, config: Optional[RAGConfig] = None
) -> Tuple[VectorStore, Dict[str, Any]]:
    """Load a persisted index; returns the store and its metadata."""
    directory = Path(directory)
    meta = read_index_meta(directory)

    vectors_path = directory / VECTORS_FILE
    chunks_path = directory / CHUNKS_FILE
    if not vectors_path.exists() or not chunks_path.exists():
        raise VectorStoreError(f"Incomplete index directory: {directory}")

    vectors = np.load(vectors_path).astype(np.float32)
    chunks = [
        Chunk.from_dict(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dim = int(meta.get("dim") or (vectors.shape[1] if vectors.size else 0))
    if dim <= 0:
        raise VectorStoreError(f"Cannot determine embedding dim for index {directory}")

    store = (
        get_vector_store(config, dim)
        if config is not None
        else NumpyVectorStore(dim)
    )
    store.add(vectors, chunks)
    logger.info("Loaded index with %d chunk(s) from %s", len(chunks), directory)
    return store, meta
