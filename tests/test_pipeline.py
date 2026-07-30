import pytest

from nlp_rag.config import RAGConfig
from nlp_rag.pipeline import ConversationalRAG, RAGPipeline


def test_indexing_populates_stores(pipeline):
    stats = pipeline.stats()
    assert stats["chunks"] > 0
    assert stats["chunks"] == len(pipeline.bm25)
    assert stats["vector_backend"] == "numpy"
    assert stats["generator"] == "extractive"


def test_answer_returns_sources(pipeline):
    result = pipeline.answer("What is retrieval-augmented generation?")

    assert result.num_sources > 0
    assert result.answer
    assert any("retrieval" in source["text"].lower() for source in result.sources)
    payload = result.to_dict()
    assert payload["question"].startswith("What is")
    assert payload["num_sources"] == result.num_sources


def test_top_k_is_respected(pipeline):
    assert len(pipeline.retrieve("transformers", top_k=2)) <= 2


def test_empty_question_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.answer("   ")


def test_index_texts_and_paths(config, tmp_path):
    pipe = RAGPipeline.build(config)
    assert pipe.index_texts(["Alpha beta gamma.", "  ", ""]) == 1

    doc = tmp_path / "extra.md"
    doc.write_text("Delta epsilon zeta is a distinct topic.", encoding="utf-8")
    assert pipe.index_paths([doc]) == 1
    assert pipe.stats()["documents"] == 2


def test_save_and_load_roundtrip(pipeline, config):
    directory = pipeline.save()
    assert (directory / "meta.json").exists()

    reloaded = RAGPipeline.load(directory, config=config)

    assert len(reloaded.store) == len(pipeline.store)
    assert len(reloaded.bm25) == len(pipeline.bm25)

    question = "Why combine BM25 with dense retrieval?"
    before = [r.chunk.id for r in pipeline.retrieve(question)]
    after = [r.chunk.id for r in reloaded.retrieve(question)]
    assert before == after


def test_load_rejects_mismatched_embedding_dim(pipeline, config):
    directory = pipeline.save()
    mismatched = config.replace(embedding_dim=config.embedding_dim * 2)
    with pytest.raises(ValueError):
        RAGPipeline.load(directory, config=mismatched)


def test_conversational_expands_follow_up(pipeline):
    session = ConversationalRAG(pipeline)
    session.chat("What is retrieval-augmented generation?")
    result = session.chat("Why does it help?")

    assert result.rewritten_question is not None
    assert "retrieval" in result.rewritten_question.lower()
    assert result.question == "Why does it help?"
    assert len(session.history) == 2


def test_conversational_leaves_standalone_questions_alone(pipeline):
    session = ConversationalRAG(pipeline)
    session.chat("What is retrieval-augmented generation?")
    result = session.chat("Which metrics measure retrieval quality?")
    assert result.rewritten_question is None


def test_conversational_reset(pipeline):
    session = ConversationalRAG(pipeline)
    session.chat("What are transformers?")
    session.reset()
    assert session.history == []


def test_history_window_is_bounded(pipeline):
    session = ConversationalRAG(pipeline, history_window=2)
    for question in ("What is NLP?", "What is RAG?", "What is BM25?"):
        session.chat(question)
    assert len(session.history) == 2


def test_bm25_only_pipeline(config):
    pipe = RAGPipeline.build(config.replace(retriever="bm25"))
    pipe.index_texts(["Reciprocal rank fusion merges ranked lists."])
    results = pipe.retrieve("reciprocal rank fusion")
    assert results and results[0].retriever == "bm25"


def test_dense_only_pipeline(config):
    pipe = RAGPipeline.build(config.replace(retriever="dense"))
    pipe.index_texts(["Reciprocal rank fusion merges ranked lists."])
    results = pipe.retrieve("reciprocal rank fusion")
    assert results and results[0].retriever == "dense"


def test_build_uses_configured_backends():
    pipe = RAGPipeline.build(
        RAGConfig(embedding_backend="hashing", vector_backend="numpy")
    )
    assert pipe.embedder.name == "hashing"
    assert pipe.store.backend == "numpy"
