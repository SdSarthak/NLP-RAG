import pytest

from nlp_rag.config import RAGConfig


def test_defaults_are_valid():
    config = RAGConfig()
    assert config.chunk_overlap < config.chunk_size
    assert config.top_k > 0


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        RAGConfig(chunk_size=100, chunk_overlap=100)


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        RAGConfig(embedding_backend="magic")
    with pytest.raises(ValueError):
        RAGConfig(generator="telepathy")


def test_from_env_reads_prefixed_variables(monkeypatch):
    monkeypatch.setenv("NLP_RAG_TOP_K", "9")
    monkeypatch.setenv("NLP_RAG_RETRIEVER", "bm25")
    monkeypatch.setenv("NLP_RAG_INDEX_DIR", "custom-index")

    config = RAGConfig.from_env()

    assert config.top_k == 9
    assert config.retriever == "bm25"
    assert str(config.index_dir) == "custom-index"


def test_from_env_overrides_win(monkeypatch):
    monkeypatch.setenv("NLP_RAG_TOP_K", "9")
    config = RAGConfig.from_env(top_k=2)
    assert config.top_k == 2


def test_from_env_ignores_unparsable_values(monkeypatch):
    monkeypatch.setenv("NLP_RAG_TOP_K", "not-a-number")
    assert RAGConfig.from_env().top_k == RAGConfig().top_k


def test_roundtrip_dict():
    config = RAGConfig(top_k=7, retriever="dense")
    restored = RAGConfig.from_dict(config.to_dict())
    assert restored.top_k == 7
    assert restored.retriever == "dense"


def test_replace_ignores_none():
    config = RAGConfig(top_k=4)
    assert config.replace(top_k=None).top_k == 4
    assert config.replace(top_k=8).top_k == 8


def test_invalid_env_value_names_the_variable(monkeypatch):
    """"top_k must be positive" alone gives no hint where the value came from."""
    monkeypatch.setenv("NLP_RAG_TOP_K", "-3")
    with pytest.raises(ValueError, match="NLP_RAG_TOP_K"):
        RAGConfig.from_env()


def test_invalid_env_backend_names_the_variable(monkeypatch):
    monkeypatch.setenv("NLP_RAG_RETRIEVER", "telepathy")
    with pytest.raises(ValueError, match="NLP_RAG_RETRIEVER"):
        RAGConfig.from_env()


def test_programmatic_error_is_not_blamed_on_the_environment(monkeypatch):
    monkeypatch.delenv("NLP_RAG_TOP_K", raising=False)
    with pytest.raises(ValueError) as excinfo:
        RAGConfig.from_env(top_k=-1)
    assert "NLP_RAG_" not in str(excinfo.value)


def test_unknown_override_is_rejected():
    with pytest.raises(TypeError):
        RAGConfig.from_env(chunk_sze=100)


def test_blank_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("NLP_RAG_RETRIEVER", "   ")
    assert RAGConfig.from_env().retriever == RAGConfig().retriever


def test_env_index_dir_expands_user(monkeypatch):
    monkeypatch.setenv("NLP_RAG_INDEX_DIR", "~/rag-index")
    assert "~" not in str(RAGConfig.from_env().index_dir)
