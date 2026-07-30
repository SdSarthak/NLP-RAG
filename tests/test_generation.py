from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk
from nlp_rag.generation import (
    NO_CONTEXT_ANSWER,
    ExtractiveGenerator,
    build_prompt,
    format_context,
    get_generator,
)
from nlp_rag.retrieval import RetrievedChunk


def make_result(text: str, rank: int = 1, source: str = "corpus") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"{source}::{rank}", text=text, source=source, index=rank),
        score=1.0 / rank,
        rank=rank,
        retriever="hybrid",
    )


def test_extractive_generator_selects_relevant_sentence():
    results = [
        make_result(
            "Chunking splits documents into pieces. "
            "Reciprocal rank fusion merges two ranked lists into one.",
            rank=1,
        ),
        make_result("Transformers rely on self-attention.", rank=2),
    ]
    answer = ExtractiveGenerator().generate(
        "What does reciprocal rank fusion do?", results
    )

    assert "reciprocal rank fusion" in answer.lower()
    assert "[1]" in answer


def test_extractive_generator_without_context():
    assert ExtractiveGenerator().generate("anything", []) == NO_CONTEXT_ANSWER


def test_extractive_generator_flags_missing_answer():
    results = [make_result("Penguins huddle together to stay warm.", rank=1)]
    answer = ExtractiveGenerator().generate("How is BM25 scored?", results)
    assert "does not directly answer" in answer


def test_extractive_generator_respects_sentence_budget():
    text = " ".join(f"Retrieval fact number {i} about retrieval." for i in range(10))
    answer = ExtractiveGenerator(max_sentences=2).generate("retrieval", [make_result(text)])
    assert answer.count("[1]") == 2


def test_extractive_generator_ignores_markdown_headings():
    results = [
        make_result(
            "## Transformers\nTransformers are built on self-attention.", rank=1
        )
    ]
    answer = ExtractiveGenerator().generate("What are transformers built on?", results)

    assert answer.startswith("Transformers are built on self-attention.")
    assert "##" not in answer


def test_extractive_generator_leads_with_the_best_sentence():
    results = [
        make_result("Cats purr. Dogs bark loudly.", rank=1),
        make_result("Reciprocal rank fusion merges ranked lists.", rank=2),
    ]
    answer = ExtractiveGenerator(max_sentences=2).generate(
        "What does reciprocal rank fusion merge?", results
    )
    assert answer.startswith("Reciprocal rank fusion merges ranked lists. [2]")


def test_format_context_numbers_and_truncates():
    results = [make_result("alpha " * 50, rank=1), make_result("beta", rank=2)]
    context = format_context(results, max_chars=120)

    assert context.startswith("[1] (source: corpus)")
    assert len(context) <= 130
    assert context.endswith("...")


def test_build_prompt_contains_question_and_context():
    prompt = build_prompt("Why?", "[1] because")
    assert "Question: Why?" in prompt
    assert "[1] because" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_get_generator_defaults_to_extractive():
    generator = get_generator(RAGConfig(generator="extractive"))
    assert generator.name == "extractive"


def test_get_generator_falls_back_when_backend_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    generator = get_generator(RAGConfig(generator="anthropic"))
    assert generator.name == "extractive"
