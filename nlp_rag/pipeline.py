"""End-to-end RAG orchestration: index -> retrieve -> generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk, RawDocument, chunk_documents, load_paths
from nlp_rag.embeddings import Embedder, get_embedder, tokenize
from nlp_rag.generation import Generator, get_generator
from nlp_rag.retrieval import (
    BM25Retriever,
    RetrievedChunk,
    build_retriever,
    content_tokens,
)
from nlp_rag.vectorstore import (
    VectorStore,
    get_vector_store,
    load_vector_store,
    read_index_meta,
)

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
        # Chunk ids already in the store. Re-indexing a path must not append a
        # second copy of every chunk: duplicates waste memory, and because the
        # hybrid retriever fuses on chunk id they also collapse at query time,
        # so a top_k of 5 silently returns fewer than 5 distinct passages.
        self._indexed_ids = {chunk.id for chunk in self.store.chunks}

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
        # Read the metadata before the vectors: a dimension mismatch is then
        # reported without first loading (and copying) the whole matrix.
        meta = read_index_meta(directory)

        saved_config = meta.get("config") or {}
        if config is None:
            config = RAGConfig.from_dict(saved_config) if saved_config else RAGConfig.from_env()
        config = config.replace(index_dir=directory)

        embedder = get_embedder(config)
        index_dim = int(meta.get("dim") or 0)
        if index_dim and embedder.dim != index_dim:
            saved_embedder = meta.get("embedder") or {}
            raise ValueError(
                f"Embedder dim {embedder.dim} does not match index dim {index_dim}. "
                "Re-index, or set the embedding backend used at index time "
                f"({saved_embedder.get('name', 'unknown')})."
            )

        # Load straight into the configured backend rather than loading into a
        # NumPy store and copying: that held two full copies of the matrix.
        store, meta = load_vector_store(directory, config=config)
        if embedder.dim != store.dim:
            raise ValueError(
                f"Embedder dim {embedder.dim} does not match index dim {store.dim}. "
                "Re-index with a matching embedding backend."
            )

        pipeline = cls(config, embedder, store, get_generator(config))
        pipeline.bm25.add(store.chunks)
        return pipeline

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and index ``chunks``, skipping ids that are already present.

        Returns the number of chunks actually added.
        """
        if not chunks:
            return 0

        fresh: List[Chunk] = []
        seen: set = set()
        duplicates = 0
        for chunk in chunks:
            if chunk.id in self._indexed_ids or chunk.id in seen:
                duplicates += 1
                continue
            seen.add(chunk.id)
            fresh.append(chunk)

        if duplicates:
            logger.info(
                "Skipped %d chunk(s) already present in the index", duplicates
            )
        if not fresh:
            return 0

        vectors = self.embedder.encode([chunk.text for chunk in fresh])
        self.store.add(vectors, fresh)
        self.bm25.add(fresh)
        self._indexed_ids.update(seen)
        logger.info(
            "Indexed %d chunk(s); index size is now %d", len(fresh), len(self.store)
        )
        return len(fresh)

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
        # `top_k or self.config.top_k` silently rewrote an explicit 0 (and any
        # negative value) to the configured default instead of rejecting it.
        top_k = self.config.top_k if top_k is None else int(top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
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
        # Tokenize rather than split on whitespace: "What about them?" splits to
        # "them?" which never matches a marker, so the follow-up went unresolved.
        raw_tokens = tokenize(question)
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
