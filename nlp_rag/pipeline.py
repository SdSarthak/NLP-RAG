"""End-to-end RAG orchestration: index -> retrieve -> generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk, RawDocument, chunk_documents, load_paths
from nlp_rag.embeddings import Embedder, get_embedder
from nlp_rag.generation import Generator, get_generator
from nlp_rag.retrieval import (
    BM25Retriever,
    RetrievedChunk,
    build_retriever,
    content_tokens,
)
from nlp_rag.vectorstore import VectorStore, get_vector_store, load_vector_store

logger = logging.getLogger(__name__)

FOLLOW_UP_MARKERS = frozenset(
    {
        "it",
        "its",
        "that",
        "this",
        "they",
        "them",
        "these",
        "those",
        "he",
        "she",
        "him",
        "her",
        "one",
        "ones",
        "same",
        "there",
    }
)


@dataclass
class RAGAnswer:
    """The result of a full RAG query."""

    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retriever: str = "hybrid"
    generator: str = "extractive"
    rewritten_question: Optional[str] = None

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "answer": self.answer,
            "sources": self.sources,
            "num_sources": self.num_sources,
            "retriever": self.retriever,
            "generator": self.generator,
        }


class RAGPipeline:
    """Owns the embedder, the vector store, the BM25 index and the generator."""

    def __init__(
        self,
        config: RAGConfig,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        bm25: Optional[BM25Retriever] = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, config: Optional[RAGConfig] = None) -> "RAGPipeline":
        """Create an empty pipeline from a config (or the environment)."""
        config = config or RAGConfig.from_env()
        embedder = get_embedder(config)
        store = get_vector_store(config, embedder.dim)
        generator = get_generator(config)
        return cls(config, embedder, store, generator)

    @classmethod
    def load(
        cls, directory: Path | str, config: Optional[RAGConfig] = None
    ) -> "RAGPipeline":
        """Load a persisted index and rebuild the retrieval structures."""
        directory = Path(directory)
        store, meta = load_vector_store(directory, config=None)

        saved_config = meta.get("config") or {}
        if config is None:
            config = RAGConfig.from_dict(saved_config) if saved_config else RAGConfig.from_env()
        config = config.replace(index_dir=directory)

        embedder = get_embedder(config)
        if embedder.dim != store.dim:
            raise ValueError(
                f"Embedder dim {embedder.dim} does not match index dim {store.dim}. "
                "Re-index, or set the embedding backend used at index time "
                f"({meta.get('embedder', {}).get('name', 'unknown')})."
            )

        # Move the loaded vectors into the configured backend when they differ.
        target = get_vector_store(config, store.dim)
        if target.backend != store.backend:
            target.add(store.vectors(), store.chunks)
            store = target

        pipeline = cls(config, embedder, store, get_generator(config))
        pipeline.bm25.add(store.chunks)
        return pipeline

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        self.store.add(vectors, chunks)
        self.bm25.add(chunks)
        logger.info("Indexed %d chunk(s); index size is now %d", len(chunks), len(self.store))
        return len(chunks)

    def index_documents(self, documents: Sequence[RawDocument]) -> int:
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return self.index_chunks(chunks)

    def index_texts(self, texts: Sequence[str], source: str = "inline") -> int:
        documents = [
            RawDocument(source=f"{source}:{i}", text=text)
            for i, text in enumerate(texts)
            if text and text.strip()
        ]
        return self.index_documents(documents)

    def index_paths(self, paths: Iterable[Path | str], recursive: bool = True) -> int:
        documents = load_paths(paths, recursive=recursive)
        if not documents:
            logger.warning("No supported documents found in %s", list(paths))
            return 0
        return self.index_documents(documents)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or self.config.top_k
        retriever = build_retriever(self.config, self.store, self.embedder, self.bm25)
        results = retriever.retrieve(question, top_k)
        if self.config.min_score > 0.0:
            results = [r for r in results if r.score >= self.config.min_score]
        logger.info("Retrieved %d chunk(s) for %r", len(results), question[:60])
        return results

    def answer(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        results = self.retrieve(question, top_k)
        answer_text = self.generator.generate(question, results)
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=[result.to_dict() for result in results],
            retriever=self.config.retriever,
            generator=self.generator.name,
        )

    # ------------------------------------------------------------------
    # Persistence / introspection
    # ------------------------------------------------------------------
    def save(self, directory: Optional[Path | str] = None) -> Path:
        directory = Path(directory or self.config.index_dir)
        return self.store.save(
            directory,
            extra={
                "embedder": self.embedder.describe(),
                "config": self.config.to_dict(),
            },
        )

    def stats(self) -> Dict[str, Any]:
        sources = sorted({chunk.source for chunk in self.store.chunks})
        return {
            "chunks": len(self.store),
            "documents": len(sources),
            "sources": sources,
            "embedder": self.embedder.describe(),
            "vector_backend": self.store.backend,
            "retriever": self.config.retriever,
            "generator": self.generator.name,
        }


class ConversationalRAG:
    """Multi-turn wrapper that resolves follow-up questions against history."""

    def __init__(self, pipeline: RAGPipeline, history_window: int = 4) -> None:
        self.pipeline = pipeline
        self.history_window = max(1, history_window)
        self.history: List[Dict[str, str]] = []

    def reset(self) -> None:
        self.history.clear()

    def contextualize(self, question: str) -> str:
        """Expand terse follow-ups with terms from the previous question."""
        if not self.history:
            return question

        tokens = content_tokens(question)
        raw_tokens = question.lower().split()
        needs_context = len(tokens) < 3 or bool(
            FOLLOW_UP_MARKERS.intersection(raw_tokens)
        )
        if not needs_context:
            return question

        previous = self.history[-1]["question"]
        carry = [t for t in content_tokens(previous) if t not in set(tokens)]
        if not carry:
            return question
        return f"{question} {' '.join(carry[:8])}"

    def chat(self, message: str, top_k: Optional[int] = None) -> RAGAnswer:
        message = message.strip()
        if not message:
            raise ValueError("message must not be empty")

        rewritten = self.contextualize(message)
        result = self.pipeline.answer(rewritten, top_k)
        result.question = message
        result.rewritten_question = rewritten if rewritten != message else None

        self.history.append({"question": message, "answer": result.answer})
        if len(self.history) > self.history_window:
            self.history = self.history[-self.history_window :]
        return result
