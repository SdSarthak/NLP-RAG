import json

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


# ----------------------------------------------------------------------
# A persisted index is untrusted input
# ----------------------------------------------------------------------
@pytest.fixture
def saved_index(tmp_path):
    store = NumpyVectorStore(dim=3)
    store.add(np.eye(3, dtype=np.float32), make_chunks(3))
    return store.save(tmp_path / "idx", extra={"embedder": {"name": "hashing"}})


def test_save_is_atomic_and_leaves_no_temporary_files(saved_index):
    assert sorted(p.name for p in saved_index.iterdir()) == [
        "chunks.jsonl",
        "meta.json",
        "vectors.npy",
    ]


def test_save_does_not_clobber_a_good_index_when_it_fails(tmp_path, monkeypatch):
    directory = tmp_path / "idx"
    good = NumpyVectorStore(dim=3)
    good.add(np.eye(3, dtype=np.float32), make_chunks(3))
    good.save(directory)
    original = (directory / "chunks.jsonl").read_text(encoding="utf-8")

    broken = NumpyVectorStore(dim=3)
    broken.add(np.zeros((1, 3), dtype=np.float32), make_chunks(1))
    monkeypatch.setattr(
        "nlp_rag.vectorstore.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(VectorStoreError):
        broken.save(directory)

    assert (directory / "chunks.jsonl").read_text(encoding="utf-8") == original
    loaded, _ = load_vector_store(directory)
    assert len(loaded) == 3


def test_corrupt_metadata_is_reported(saved_index):
    (saved_index / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(VectorStoreError, match="Corrupt index metadata"):
        load_vector_store(saved_index)


def test_metadata_must_be_an_object(saved_index):
    (saved_index / "meta.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(VectorStoreError):
        load_vector_store(saved_index)


def test_corrupt_vector_file_is_reported(saved_index):
    (saved_index / "vectors.npy").write_bytes(b"not a numpy file")
    with pytest.raises(VectorStoreError, match="Corrupt vector file"):
        load_vector_store(saved_index)


def test_malformed_chunk_line_names_the_line(saved_index):
    (saved_index / "chunks.jsonl").write_text(
        '{"id": "a", "text": "x"}\n{broken\n', encoding="utf-8"
    )
    with pytest.raises(VectorStoreError, match="chunks.jsonl:2"):
        load_vector_store(saved_index)


def test_chunk_record_missing_required_fields_is_reported(saved_index):
    (saved_index / "chunks.jsonl").write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(VectorStoreError, match="not a usable chunk record"):
        load_vector_store(saved_index)


def test_vector_and_chunk_counts_must_agree(saved_index):
    (saved_index / "chunks.jsonl").write_text(
        '{"id": "a", "text": "x"}\n', encoding="utf-8"
    )
    with pytest.raises(VectorStoreError, match="inconsistent"):
        load_vector_store(saved_index)


def test_declared_dim_must_match_the_stored_vectors(saved_index):
    meta = json.loads((saved_index / "meta.json").read_text(encoding="utf-8"))
    meta["dim"] = 99
    (saved_index / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorStoreError, match="declares dim"):
        load_vector_store(saved_index)


def test_incomplete_directory_names_the_missing_file(saved_index):
    (saved_index / "vectors.npy").unlink()
    with pytest.raises(VectorStoreError, match="missing vectors.npy"):
        load_vector_store(saved_index)


def test_an_empty_index_round_trips(tmp_path):
    directory = NumpyVectorStore(dim=4).save(tmp_path / "empty")
    loaded, meta = load_vector_store(directory)
    assert len(loaded) == 0
    assert meta["dim"] == 4
    assert loaded.search(np.ones(4, dtype=np.float32), top_k=3) == []
