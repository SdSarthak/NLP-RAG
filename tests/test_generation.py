import pytest

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk
from nlp_rag.generation import (
    NO_CONTEXT_ANSWER,
    AnthropicGenerator,
    ExtractiveGenerator,
    GenerationError,
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


# ----------------------------------------------------------------------
# Anthropic generator: response handling, without touching the network
# ----------------------------------------------------------------------
class _Block:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.messages = _FakeMessages(response, error)


def _anthropic(response=None, error=None, **kwargs):
    return AnthropicGenerator(client=_FakeClient(response, error), **kwargs)


def test_anthropic_generator_returns_text():
    generator = _anthropic(_Response([_Block("Grounded answer. [1]")]))
    assert generator.generate("q", [make_result("context")]) == "Grounded answer. [1]"


def test_anthropic_generator_flags_a_truncated_answer():
    """A response cut off at max_tokens must not be presented as complete."""
    generator = _anthropic(
        _Response([_Block("The answer begins but then st")], stop_reason="max_tokens")
    )
    answer = generator.generate("q", [make_result("context")])
    assert answer.startswith("The answer begins but then st")
    assert "truncated" in answer


def test_anthropic_generator_handles_a_refusal():
    generator = _anthropic(_Response([], stop_reason="refusal"))
    assert "declined" in generator.generate("q", [make_result("context")])


def test_anthropic_generator_handles_an_empty_response():
    generator = _anthropic(_Response([]))
    assert generator.generate("q", [make_result("context")]) == NO_CONTEXT_ANSWER


def test_anthropic_generator_ignores_non_text_blocks():
    generator = _anthropic(
        _Response([_Block("", type="thinking"), _Block("Visible answer.")])
    )
    assert generator.generate("q", [make_result("context")]) == "Visible answer."


def test_anthropic_generator_skips_the_api_without_context():
    generator = _anthropic(_Response([_Block("should not be called")]))
    assert generator.generate("q", []) == NO_CONTEXT_ANSWER
    assert generator._client.messages.calls == []


def test_anthropic_generator_wraps_transport_errors_with_a_hint():
    generator = _anthropic(error=_named_error("APIConnectionError"))
    with pytest.raises(GenerationError) as excinfo:
        generator.generate("q", [make_result("context")])
    assert "Could not reach the Claude API" in str(excinfo.value)


def test_anthropic_generator_wraps_unknown_errors():
    generator = _anthropic(error=RuntimeError("boom"))
    with pytest.raises(GenerationError, match="boom"):
        generator.generate("q", [make_result("context")])


def test_anthropic_generator_rejects_bad_max_tokens():
    with pytest.raises(ValueError):
        _anthropic(_Response([]), max_tokens=0)


def test_anthropic_generator_sends_the_configured_model_and_limit():
    generator = _anthropic(
        _Response([_Block("ok")]), model="claude-opus-5", max_tokens=1234
    )
    generator.generate("q", [make_result("context")])
    call = generator._client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 1234
    assert call["messages"][0]["role"] == "user"


def _named_error(name):
    return type(name, (Exception,), {})("transport failure")
