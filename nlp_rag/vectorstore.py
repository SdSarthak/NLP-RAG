"""Vector stores.

Indexes are persisted in a portable format (``vectors.npy`` + ``chunks.jsonl`` +
``meta.json``) so the same directory can be opened with either the NumPy or the
FAISS backend.
"""

from __future__ import annotations

import json
import logging
import os
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
        """Persist the index to ``directory`` and return that path.

        The three files are written to temporary siblings and only then moved
        into place. An interrupted save (Ctrl-C, a full disk, a crash) therefore
        leaves the previous index intact instead of a half-written directory
        whose vectors and chunks no longer correspond.
        """
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VectorStoreError(f"Cannot create index directory {directory}: {exc}") from exc

        meta: Dict[str, Any] = {
            "backend": self.backend,
            "dim": self.dim,
            "count": len(self.chunks),
        }
        if extra:
            meta.update(extra)

        temporary: List[Path] = []
        try:
            vectors_tmp = directory / (VECTORS_FILE + ".tmp.npy")
            temporary.append(vectors_tmp)
            with vectors_tmp.open("wb") as handle:
                np.save(handle, self.vectors(), allow_pickle=False)

            chunks_tmp = directory / (CHUNKS_FILE + ".tmp")
            temporary.append(chunks_tmp)
            with chunks_tmp.open("w", encoding="utf-8") as handle:
                for chunk in self.chunks:
                    handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

            meta_tmp = directory / (META_FILE + ".tmp")
            temporary.append(meta_tmp)
            meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # os.replace is atomic on POSIX and Windows alike.
            os.replace(vectors_tmp, directory / VECTORS_FILE)
            os.replace(chunks_tmp, directory / CHUNKS_FILE)
            os.replace(meta_tmp, directory / META_FILE)
        except OSError as exc:
            for path in temporary:
                try:
                    path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
            raise VectorStoreError(f"Failed to save index to {directory}: {exc}") from exc

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
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStoreError(f"Corrupt index metadata {meta_path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise VectorStoreError(f"Index metadata {meta_path} is not a JSON object")
    return meta


def _read_chunks(path: Path) -> List[Chunk]:
    """Parse ``chunks.jsonl``, naming the line that fails."""
    chunks: List[Chunk] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise VectorStoreError(f"Cannot read {path}: {exc}") from exc
    with handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise VectorStoreError(f"{path}:{number} is not valid JSON: {exc}") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise VectorStoreError(
                    f"{path}:{number} is not a usable chunk record: {exc}"
                ) from exc
    return chunks


def load_vector_store(
    directory: Path | str, config: Optional[RAGConfig] = None
) -> Tuple[VectorStore, Dict[str, Any]]:
    """Load a persisted index; returns the store and its metadata."""
    directory = Path(directory)
    meta = read_index_meta(directory)

    vectors_path = directory / VECTORS_FILE
    chunks_path = directory / CHUNKS_FILE
    missing = [p.name for p in (vectors_path, chunks_path) if not p.exists()]
    if missing:
        raise VectorStoreError(
            f"Incomplete index directory {directory}: missing {', '.join(missing)}"
        )

    try:
        # allow_pickle stays off: an index directory is untrusted input and a
        # pickled .npy would execute arbitrary code on load.
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VectorStoreError(f"Corrupt vector file {vectors_path}: {exc}") from exc
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise VectorStoreError(
            f"{vectors_path} holds a {vectors.ndim}-D array; expected a 2-D matrix"
        )

    chunks = _read_chunks(chunks_path)
    if vectors.shape[0] != len(chunks):
        raise VectorStoreError(
            f"Index {directory} is inconsistent: {vectors.shape[0]} vector(s) for "
            f"{len(chunks)} chunk(s). Re-build it with `index`."
        )

    dim = int(meta.get("dim") or (vectors.shape[1] if vectors.size else 0))
    if dim <= 0:
        raise VectorStoreError(f"Cannot determine embedding dim for index {directory}")
    if vectors.size and vectors.shape[1] != dim:
        raise VectorStoreError(
            f"Index {directory} declares dim {dim} but stores {vectors.shape[1]}-D vectors"
        )

    store = (
        get_vector_store(config, dim)
        if config is not None
        else NumpyVectorStore(dim)
    )
    store.add(vectors, chunks)
    logger.info("Loaded index with %d chunk(s) from %s", len(chunks), directory)
    return store, meta
