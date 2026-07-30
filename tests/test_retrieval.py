import numpy as np
import pytest

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk
from nlp_rag.embeddings import HashingEmbedder
from nlp_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    build_retriever,
    content_tokens,
)
from nlp_rag.vectorstore import NumpyVectorStore

CORPUS = [
    "BM25 is a lexical ranking function based on term frequency.",
    "Transformers use self-attention to process tokens in parallel.",
    "Vector embeddings power semantic search with cosine similarity.",
    "Chunk overlap preserves sentences that straddle a boundary.",
]


def make_chunks():
    return [
        Chunk(id=f"c{i}", text=text, source="corpus", index=i)
        for i, text in enumerate(CORPUS)
    ]


@pytest.fixture
def chunks():
    return make_chunks()


@pytest.fixture
def dense(chunks):
    embedder = HashingEmbedder(dim=512)
    store = NumpyVectorStore(dim=embedder.dim)
    store.add(embedder.encode([c.text for c in chunks]), chunks)
    return DenseRetriever(store, embedder)


@pytest.fixture
def bm25(chunks):
    retriever = BM25Retriever()
    retriever.add(chunks)
    return retriever


def test_content_tokens_drops_stop_words():
    assert content_tokens("What is the meaning of it all?") == ["meaning", "all"]


def test_bm25_ranks_keyword_match_first(bm25):
    results = bm25.retrieve("lexical ranking function", top_k=2)
    assert results
    assert results[0].chunk.id == "c0"
    assert results[0].score > 0
    assert results[0].rank == 1


def test_bm25_empty_query_returns_nothing(bm25):
    assert bm25.retrieve("the of and", top_k=3) == []
    assert bm25.retrieve("", top_k=3) == []


def test_bm25_unknown_terms_return_nothing(bm25):
    assert bm25.retrieve("zebra xylophone", top_k=3) == []


def test_dense_retriever_finds_related_chunk(dense):
    results = dense.retrieve("semantic search with embeddings", top_k=2)
    assert results
    assert results[0].chunk.id == "c2"
    assert results[0].retriever == "dense"


def test_dense_retriever_on_empty_store():
    embedder = HashingEmbedder(dim=32)
    retriever = DenseRetriever(NumpyVectorStore(dim=32), embedder)
    assert retriever.retrieve("anything", top_k=3) == []


def test_hybrid_fuses_both_retrievers(dense, bm25):
    hybrid = HybridRetriever([dense, bm25], rrf_k=60)
    results = hybrid.retrieve("self-attention transformers", top_k=3)

    assert results
    assert results[0].chunk.id == "c1"
    assert results[0].retriever == "hybrid"
    assert set(results[0].components) & {"dense", "bm25"}
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_requires_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_build_retriever_selects_strategy(dense, bm25):
    store, embedder = dense.store, dense.embedder
    assert build_retriever(RAGConfig(retriever="dense"), store, embedder, bm25).name == "dense"
    assert build_retriever(RAGConfig(retriever="bm25"), store, embedder, bm25).name == "bm25"
    assert build_retriever(RAGConfig(retriever="hybrid"), store, embedder, bm25).name == "hybrid"


def test_retrieved_chunk_serialisation(bm25):
    payload = bm25.retrieve("lexical ranking", top_k=1)[0].to_dict()
    assert payload["id"] == "c0"
    assert payload["retriever"] == "bm25"
    assert isinstance(payload["score"], float)


def test_bm25_scores_are_non_negative(bm25, chunks):
    tokens = content_tokens("embeddings cosine similarity")
    scores = [bm25.score(tokens, i) for i in range(len(chunks))]
    assert all(score >= 0 for score in scores)
    assert scores[2] == max(scores)
    assert not np.isnan(scores).any()
