"""Retrievers: dense (embeddings), lexical (BM25) and hybrid (RRF fusion)."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

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
        if top_k <= 0 or not query.strip() or len(self.store) == 0:
            return []
        vector = self.embedder.encode_one(query)
        # A query with no in-vocabulary features (punctuation, an unsupported
        # script for a lexical embedder) embeds to the zero vector, which scores
        # every chunk 0.0 and would return an arbitrary slice of the index as if
        # it were relevant.
        if not np.any(vector):
            logger.debug("Query %r produced an empty embedding; no dense hits", query[:60])
            return []
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
    """Okapi BM25 over the same chunks, for exact-term matching.

    Scoring goes through an inverted index, so a query only touches the chunks
    that actually contain one of its terms. Scanning every chunk (the obvious
    implementation) costs the same whether the query matches three documents or
    three hundred thousand.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self._term_freqs: List[Counter] = []
        self._lengths: List[int] = []
        self._doc_freq: Counter = Counter()
        self._avg_len = 0.0
        # term -> [(doc index, term frequency), ...]
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Per-document length normalisation. Depends on the average document
        # length, so adding chunks invalidates it; it is rebuilt on demand
        # rather than on every add, keeping add() linear in the new chunks only.
        self._denominators: List[float] = []
        self._dirty = True

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            tokens = content_tokens(chunk.text)
            freqs = Counter(tokens)
            doc_index = len(self.chunks)
            self.chunks.append(chunk)
            self._term_freqs.append(freqs)
            self._lengths.append(len(tokens))
            self._doc_freq.update(freqs.keys())
            for term, tf in freqs.items():
                self._postings[term].append((doc_index, tf))
        total = sum(self._lengths)
        self._avg_len = total / len(self._lengths) if self._lengths else 0.0
        self._dirty = True

    def __len__(self) -> int:
        return len(self.chunks)

    def _refresh(self) -> None:
        if not self._dirty:
            return
        average = self._avg_len or 1.0
        k1, b = self.k1, self.b
        self._denominators = [
            k1 * (1 - b + b * (length or 1) / average) for length in self._lengths
        ]
        self._dirty = False

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self._doc_freq.get(term, 0)
        # BM25+ style smoothing keeps the idf strictly positive.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document. Kept for introspection and testing."""
        if not 0 <= doc_index < len(self.chunks):
            raise IndexError(f"no document at index {doc_index}")
        self._refresh()
        freqs = self._term_freqs[doc_index]
        norm = self._denominators[doc_index]
        total = 0.0
        for term in set(query_tokens):
            tf = freqs.get(term, 0)
            if not tf:
                continue
            total += self._idf(term) * (tf * (self.k1 + 1)) / (tf + norm)
        return total

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not self.chunks or top_k <= 0:
            return []
        query_tokens = set(content_tokens(query))
        if not query_tokens:
            return []

        self._refresh()
        denominators = self._denominators
        k1_plus_one = self.k1 + 1.0
        scores: Dict[int, float] = defaultdict(float)
        for term in query_tokens:
            postings = self._postings.get(term)
            if not postings:
                continue
            weight = self._idf(term) * k1_plus_one
            for doc_index, tf in postings:
                scores[doc_index] += weight * tf / (tf + denominators[doc_index])

        ranked = sorted(
            ((index, score) for index, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                retriever=self.name,
                components={"bm25": score},
            )
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
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
