"""Retrievers: dense (embeddings), lexical (BM25) and hybrid (RRF fusion)."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk
from nlp_rag.embeddings import Embedder, tokenize
from nlp_rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# A small, dependency-free stop word list; enough to keep BM25 focused.
STOP_WORDS = frozenset(
    """
    a an the and or but if then than that this these those there here of in on at to
    for from by with without into onto about as is are was were be been being am do
    does did doing have has had having it its it's i you he she they we me him her
    them us my your his their our what which who whom whose when where why how not no
    nor so too very can will just should would could may might must shall s t don
    """.split()
)


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with provenance."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk.id,
            "source": self.chunk.source,
            "index": self.chunk.index,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "rank": self.rank,
            "retriever": self.retriever,
            "components": {k: round(v, 6) for k, v in self.components.items()},
        }


def content_tokens(text: str) -> List[str]:
    """Tokenize and drop stop words / single characters."""
    return [t for t in tokenize(text) if t not in STOP_WORDS and len(t) > 1]


class Retriever(ABC):
    name = "retriever"

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Return up to ``top_k`` chunks relevant to ``query``."""


class DenseRetriever(Retriever):
    """Cosine similarity over embedded chunks."""

    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not query.strip() or len(self.store) == 0:
            return []
        vector = self.embedder.encode_one(query)
        hits = self.store.search(vector, top_k)
        return [
            RetrievedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                retriever=self.name,
                components={"dense": score},
            )
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]


class BM25Retriever(Retriever):
    """Okapi BM25 over the same chunks, for exact-term matching."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._term_freqs: List[Counter] = []
        self._lengths: List[int] = []
        self._doc_freq: Counter = Counter()
        self._avg_len = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            tokens = content_tokens(chunk.text)
            freqs = Counter(tokens)
            self.chunks.append(chunk)
            self._term_freqs.append(freqs)
            self._lengths.append(len(tokens))
            self._doc_freq.update(freqs.keys())
        total = sum(self._lengths)
        self._avg_len = total / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._doc_freq.get(term, 0)
        # BM25+ style smoothing keeps the idf strictly positive.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        freqs = self._term_freqs[doc_index]
        length = self._lengths[doc_index] or 1
        norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
        total = 0.0
        for term in query_tokens:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            total += self._idf(term) * (tf * (self.k1 + 1)) / (tf + norm)
        return total

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query_tokens = content_tokens(query)
        if not query_tokens:
            return []

        scored = [
            (index, self.score(query_tokens, index))
            for index in range(len(self.chunks))
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                retriever=self.name,
                components={"bm25": score},
            )
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]


class HybridRetriever(Retriever):
    """Reciprocal rank fusion of several retrievers.

    RRF is scale-free, so dense cosine scores and BM25 scores can be combined
    without normalising either of them.
    """

    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []
        candidates = top_k * self.candidate_multiplier

        fused: Dict[str, float] = defaultdict(float)
        components: Dict[str, Dict[str, float]] = defaultdict(dict)
        chunks: Dict[str, Chunk] = {}

        for retriever in self.retrievers:
            for hit in retriever.retrieve(query, candidates):
                key = hit.chunk.id
                chunks[key] = hit.chunk
                fused[key] += 1.0 / (self.rrf_k + hit.rank)
                components[key][retriever.name] = hit.score

        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [
            RetrievedChunk(
                chunk=chunks[key],
                score=score,
                rank=rank,
                retriever=self.name,
                components=components[key],
            )
            for rank, (key, score) in enumerate(ordered[:top_k], start=1)
        ]


def build_retriever(
    config: RAGConfig,
    store: VectorStore,
    embedder: Embedder,
    bm25: BM25Retriever,
) -> Retriever:
    """Assemble the retriever described by ``config``."""
    dense = DenseRetriever(store, embedder)
    if config.retriever == "dense":
        return dense
    if config.retriever == "bm25":
        return bm25
    return HybridRetriever([dense, bm25], rrf_k=config.rrf_k)
