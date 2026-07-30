import numpy as np
import pytest

from nlp_rag.config import RAGConfig
from nlp_rag.embeddings import HashingEmbedder, get_embedder, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! RAG-2 rocks.") == [
        "hello",
        "world",
        "rag",
        "2",
        "rocks",
    ]


def test_encode_shape_and_normalisation():
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.encode(["first text", "second text here"])

    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    first = embedder.encode(["retrieval augmented generation"])
    second = embedder.encode(["retrieval augmented generation"])
    assert np.allclose(first, second)


def test_empty_text_yields_zero_vector():
    embedder = HashingEmbedder(dim=32)
    vector = embedder.encode_one("")
    assert np.allclose(vector, 0.0)


def test_related_texts_are_closer_than_unrelated():
    embedder = HashingEmbedder(dim=512)
    query = embedder.encode_one("vector embeddings for semantic search")
    related = embedder.encode_one(
        "semantic search compares vector embeddings with cosine similarity"
    )
    unrelated = embedder.encode_one("the cat sat quietly on the warm windowsill")

    assert float(query @ related) > float(query @ unrelated)


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_get_embedder_honours_hashing_backend():
    config = RAGConfig(embedding_backend="hashing", embedding_dim=128)
    embedder = get_embedder(config)
    assert embedder.name == "hashing"
    assert embedder.dim == 128
    assert embedder.describe()["dim"] == 128
