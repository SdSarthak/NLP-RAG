"""Text embedding backends.

``HashingEmbedder`` is the always-available default: a deterministic, signed
feature-hashing vectoriser built on NumPy alone. When ``sentence-transformers``
is installed, ``SentenceTransformerEmbedder`` provides true semantic embeddings
and is selected automatically by :func:`get_embedder`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

from nlp_rag.config import RAGConfig

logger = logging.getLogger(__name__)

# Letters and digits in *any* script, underscore excluded so that it can be used
# as an unambiguous n-gram joiner. On pure-ASCII text this matches exactly what
# ``[a-z0-9]+`` matched, so indexes built by earlier versions stay valid.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by the lexical components.

    Unicode-aware: Cyrillic, Greek, CJK and accented Latin text produce tokens
    instead of being silently discarded.
    """
    return _TOKEN_RE.findall(text.lower())


def _hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class Embedder(ABC):
    """Turns text into L2-normalised float32 vectors."""

    name: str = "embedder"

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into a ``(len(texts), dim)`` matrix."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dim": self.dim}


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class HashingEmbedder(Embedder):
    """Signed feature hashing over word unigrams and bigrams.

    Deterministic, dependency-free, and good enough for lexical-semantic
    similarity. Pair it with BM25 through the hybrid retriever for best results.
    """

    name = "hashing"

    def __init__(self, dim: int = 384, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self.use_bigrams = use_bigrams

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> Counter:
        tokens = tokenize(text)
        features: Counter = Counter(tokens)
        if self.use_bigrams:
            # Tokens never contain "_", so a bigram key cannot collide with a
            # unigram key.
            features.update(
                f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
            )
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in self._features(text).items():
                h = _hash(feature)
                bucket = h % self._dim
                sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
                matrix[row, bucket] += sign * (1.0 + math.log(count))
        return _l2_normalize(matrix)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dim": self._dim, "bigrams": self.use_bigrams}


class SentenceTransformerEmbedder(Embedder):
    """Wrapper around a ``sentence-transformers`` model."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers` or use the hashing backend."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dim": self._dim, "model": self.model_name}


def get_embedder(config: RAGConfig) -> Embedder:
    """Instantiate the embedder described by ``config``.

    ``embedding_backend="auto"`` prefers sentence-transformers and silently falls
    back to the hashing backend when it is unavailable.
    """
    backend = config.embedding_backend

    if backend in {"auto", "sentence-transformers"}:
        try:
            embedder = SentenceTransformerEmbedder(
                model_name=config.embedding_model,
                batch_size=config.embedding_batch_size,
            )
            logger.info(
                "Using sentence-transformers embeddings (%s, dim=%d)",
                config.embedding_model,
                embedder.dim,
            )
            return embedder
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            logger.info(
                "sentence-transformers unavailable (%s); "
                "falling back to hashing embeddings",
                exc,
            )

    logger.info("Using hashing embeddings (dim=%d)", config.embedding_dim)
    return HashingEmbedder(dim=config.embedding_dim)
