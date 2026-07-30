"""Shared pytest fixtures.

Tests pin the dependency-free backends (hashing embeddings, NumPy vectors,
extractive generation) so results are deterministic and no model downloads or
API keys are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp_rag import RAGConfig, RAGPipeline  # noqa: E402
from nlp_rag.documents import RawDocument  # noqa: E402
from nlp_rag.samples import sample_documents  # noqa: E402


@pytest.fixture
def config(tmp_path: Path) -> RAGConfig:
    return RAGConfig(
        chunk_size=400,
        chunk_overlap=60,
        embedding_backend="hashing",
        embedding_dim=256,
        vector_backend="numpy",
        retriever="hybrid",
        generator="extractive",
        top_k=3,
        index_dir=tmp_path / "index",
    )


@pytest.fixture
def documents() -> list:
    return sample_documents()


@pytest.fixture
def pipeline(config: RAGConfig) -> RAGPipeline:
    pipe = RAGPipeline.build(config)
    pipe.index_documents(sample_documents())
    return pipe


@pytest.fixture
def corpus_path() -> Path:
    return ROOT / "data" / "sample_corpus.md"


@pytest.fixture
def tiny_documents() -> list:
    return [
        RawDocument(source="doc-a", text="Cats purr when they are content."),
        RawDocument(source="doc-b", text="Dogs bark to warn their owners."),
        RawDocument(source="doc-c", text="Vector databases store embeddings."),
    ]
