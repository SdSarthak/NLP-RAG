"""Configuration for the RAG pipeline.

Every value can be supplied programmatically, through environment variables
(``NLP_RAG_*``), or through a ``.env`` file when ``python-dotenv`` is installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

ENV_PREFIX = "NLP_RAG_"

EMBEDDING_BACKENDS = ("auto", "sentence-transformers", "hashing")
VECTOR_BACKENDS = ("auto", "faiss", "numpy")
RETRIEVERS = ("hybrid", "dense", "bm25")
GENERATORS = ("extractive", "transformers", "anthropic")


def _load_dotenv_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(override=False)


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name.upper())
    if value is None:
        return None
    value = value.strip()
    return value or None


def _coerce(raw: str, target_type: Any) -> Any:
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is Path:
        return Path(raw).expanduser()
    return raw


@dataclass
class RAGConfig:
    """All tunable knobs of the pipeline."""

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings
    embedding_backend: str = "auto"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # only used by the hashing backend
    embedding_batch_size: int = 32

    # Vector store
    vector_backend: str = "auto"

    # Retrieval
    retriever: str = "hybrid"
    top_k: int = 5
    rrf_k: int = 60
    min_score: float = 0.0

    # Generation
    generator: str = "extractive"
    max_context_chars: int = 6000
    transformers_model: str = "gpt2"
    transformers_max_new_tokens: int = 160
    anthropic_model: str = "claude-opus-5"
    anthropic_max_tokens: int = 4096

    # Persistence
    index_dir: Path = field(default_factory=lambda: Path("storage"))

    def __post_init__(self) -> None:
        self.index_dir = Path(self.index_dir)
        self.validate()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides: Any) -> "RAGConfig":
        """Build a config from ``NLP_RAG_*`` env vars, then apply overrides."""
        _load_dotenv_if_available()

        values: Dict[str, Any] = {}
        for f in fields(cls):
            raw = _env(f.name)
            if raw is None:
                continue
            target = Path if f.name == "index_dir" else f.type
            if isinstance(target, str):  # postponed annotations
                target = {"int": int, "float": float, "bool": bool, "str": str}.get(
                    target, str
                )
            try:
                values[f.name] = _coerce(raw, target)
            except ValueError:
                logger.warning(
                    "Ignoring invalid value for %s%s: %r",
                    ENV_PREFIX,
                    f.name.upper(),
                    raw,
                )

        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def replace(self, **overrides: Any) -> "RAGConfig":
        """Return a copy with the given (non-None) fields replaced."""
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return RAGConfig(**data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["index_dir"] = str(self.index_dir)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(
                f"embedding_backend must be one of {EMBEDDING_BACKENDS}, "
                f"got {self.embedding_backend!r}"
            )
        if self.vector_backend not in VECTOR_BACKENDS:
            raise ValueError(
                f"vector_backend must be one of {VECTOR_BACKENDS}, "
                f"got {self.vector_backend!r}"
            )
        if self.retriever not in RETRIEVERS:
            raise ValueError(
                f"retriever must be one of {RETRIEVERS}, got {self.retriever!r}"
            )
        if self.generator not in GENERATORS:
            raise ValueError(
                f"generator must be one of {GENERATORS}, got {self.generator!r}"
            )
