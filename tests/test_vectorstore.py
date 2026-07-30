import numpy as np
import pytest

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk
from nlp_rag.vectorstore import (
    NumpyVectorStore,
    VectorStoreError,
    get_vector_store,
    load_vector_store,
)


def make_chunks(n: int):
    return [
        Chunk(id=f"c{i}", text=f"chunk {i}", source="unit-test", index=i)
        for i in range(n)
    ]


def test_add_and_search_orders_by_similarity():
    store = NumpyVectorStore(dim=3)
    vectors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071, 0.7071, 0.0]], dtype=np.float32
    )
    store.add(vectors, make_chunks(3))

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert [chunk.id for chunk, _ in results] == ["c0", "c2"]
    assert results[0][1] > results[1][1]
    assert len(store) == 3


def test_search_on_empty_store_returns_empty():
    store = NumpyVectorStore(dim=4)
    assert store.search(np.zeros(4, dtype=np.float32), top_k=3) == []


def test_dimension_mismatch_raises():
    store = NumpyVectorStore(dim=3)
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((1, 4), dtype=np.float32), make_chunks(1))

    store.add(np.zeros((1, 3), dtype=np.float32), make_chunks(1))
    with pytest.raises(VectorStoreError):
        store.search(np.zeros(5, dtype=np.float32), top_k=1)


def test_vector_count_must_match_chunk_count():
    store = NumpyVectorStore(dim=2)
    with pytest.raises(VectorStoreError):
        store.add(np.zeros((3, 2), dtype=np.float32), make_chunks(2))


def test_save_and_load_roundtrip(tmp_path):
    store = NumpyVectorStore(dim=3)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.add(vectors, make_chunks(2))
    store.save(tmp_path / "idx", extra={"embedder": {"name": "hashing", "dim": 3}})

    config = RAGConfig(vector_backend="numpy", embedding_backend="hashing")
    loaded, meta = load_vector_store(tmp_path / "idx", config=config)

    assert len(loaded) == 2
    assert meta["dim"] == 3
    assert meta["embedder"]["name"] == "hashing"
    assert np.allclose(loaded.vectors(), vectors)
    assert [c.id for c in loaded.chunks] == ["c0", "c1"]


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(VectorStoreError):
        load_vector_store(tmp_path / "nope")


def test_get_vector_store_honours_numpy_backend():
    store = get_vector_store(RAGConfig(vector_backend="numpy"), dim=8)
    assert store.backend == "numpy"
    assert store.dim == 8
