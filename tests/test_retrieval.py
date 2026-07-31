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


# ----------------------------------------------------------------------
# BM25 inverted index
# ----------------------------------------------------------------------
def test_bm25_retrieve_agrees_with_scoring_every_document(bm25, chunks):
    """The inverted index must reproduce an exhaustive scan exactly."""
    query = "lexical ranking function term frequency"
    tokens = content_tokens(query)

    exhaustive = sorted(
        ((i, bm25.score(tokens, i)) for i in range(len(chunks))),
        key=lambda item: (-item[1], item[0]),
    )
    expected = [(f"c{i}", s) for i, s in exhaustive if s > 0.0]

    results = bm25.retrieve(query, top_k=len(chunks))
    assert [r.chunk.id for r in results] == [i for i, _ in expected]
    for result, (_, score) in zip(results, expected):
        assert result.score == pytest.approx(score)


def test_bm25_length_normalisation_is_refreshed_after_a_later_add(bm25):
    """Average document length changes on add; stale norms would skew scores."""
    before = bm25.retrieve("lexical ranking function", top_k=1)[0].score

    bm25.add(
        [
            Chunk(
                id=f"long{i}",
                text=" ".join(["filler"] * 200),
                source="corpus",
                index=100 + i,
            )
            for i in range(5)
        ]
    )
    after = bm25.retrieve("lexical ranking function", top_k=1)[0]

    tokens = content_tokens("lexical ranking function")
    assert after.score == pytest.approx(bm25.score(tokens, 0))
    assert after.score != pytest.approx(before)


def test_bm25_only_touches_documents_containing_a_query_term(bm25):
    postings = bm25._postings
    assert set(postings["lexical"]) == {(0, 1)}
    assert "zebra" not in postings or not postings["zebra"]


def test_bm25_repeated_query_terms_do_not_double_count(bm25):
    once = bm25.retrieve("lexical", top_k=1)[0].score
    twice = bm25.retrieve("lexical lexical lexical", top_k=1)[0].score
    assert once == pytest.approx(twice)


def test_bm25_score_rejects_an_out_of_range_document(bm25, chunks):
    with pytest.raises(IndexError):
        bm25.score(["lexical"], len(chunks))


def test_bm25_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Retriever(k1=-1.0)
    with pytest.raises(ValueError):
        BM25Retriever(b=1.5)


def test_bm25_on_an_empty_index():
    assert BM25Retriever().retrieve("anything", top_k=3) == []


def test_dense_retriever_ignores_a_termless_query(dense):
    """A zero embedding scored every chunk 0.0 and returned an arbitrary slice."""
    assert dense.retrieve("???", top_k=3) == []
    assert dense.retrieve("anything", top_k=0) == []
