"""Non-ASCII corpora must index, retrieve and print without loss or crashes.

Before these fixes a Russian or Japanese document produced zero chunks (the
noise filter counted ``[A-Za-z]`` words only), non-Latin queries embedded to the
zero vector, and printing any answer containing a character outside the console
code page raised ``UnicodeEncodeError`` on Windows.

This file is UTF-8; Python 3 reads source as UTF-8 by default.
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from nlp_rag.cli import configure_stdio, main, print_answer
from nlp_rag.documents import RawDocument, chunk_documents, is_meaningful, split_text
from nlp_rag.embeddings import HashingEmbedder, tokenize
from nlp_rag.pipeline import RAGAnswer, RAGPipeline

BACKENDS = ["--embedding-backend", "hashing", "--vector-backend", "numpy"]

RUSSIAN = (
    "Семантический "
    "поиск сравнива"
    "ет векторы коси"
    "нусным расстоя"
    "нием."
)
JAPANESE = (
    "埋め込みはベクトルです。"
    "意味検索に使います。"
)
GREEK = "Δοκιμή ελληνικού κειμένου."
FRENCH = "Le café est très chaud."
ARROW = "Implication → entailment."


@pytest.mark.parametrize("text", [RUSSIAN, JAPANESE, GREEK, FRENCH])
def test_tokenize_handles_non_ascii_scripts(text):
    tokens = tokenize(text)
    assert tokens, f"no tokens produced for {text!r}"
    assert all(token == token.lower() for token in tokens)


def test_tokenize_keeps_accented_letters_inside_a_word():
    # "café" used to be truncated to "caf" and "très" split into "tr"/"s".
    assert tokenize(FRENCH) == ["le", "café", "est", "très", "chaud"]


def test_tokenize_is_unchanged_for_ascii_text():
    """Existing persisted indexes stay valid: ASCII tokenisation must not move."""
    text = "BM25 and dense retrieval, fused with RRF (k=60)!"
    assert tokenize(text) == ["bm25", "and", "dense", "retrieval", "fused", "with", "rrf", "k", "60"]


@pytest.mark.parametrize("text", [RUSSIAN, JAPANESE, GREEK])
def test_non_latin_text_is_not_treated_as_noise(text):
    assert is_meaningful(text)


def test_page_furniture_is_still_rejected():
    assert not is_meaningful("1 2 3 4 5 6 7 8 9 10 11 12")
    assert not is_meaningful("x")
    assert not is_meaningful("  ")


@pytest.mark.parametrize("text", [RUSSIAN, JAPANESE, GREEK])
def test_non_latin_documents_produce_chunks(text):
    chunks = chunk_documents([RawDocument(source="doc", text=text)])
    assert len(chunks) == 1
    assert chunks[0].text


def test_cjk_sentences_split_on_their_own_terminator():
    units = split_text(JAPANESE, chunk_size=14, chunk_overlap=2)
    assert len(units) == 2


def test_non_latin_text_embeds_to_a_non_zero_vector():
    embedder = HashingEmbedder(dim=64)
    for text in (RUSSIAN, JAPANESE, GREEK):
        vector = embedder.encode_one(text)
        assert float(abs(vector).sum()) > 0.0


def test_unicode_spaces_are_normalised():
    from nlp_rag.documents import normalize_text

    assert normalize_text("a  b c") == "a b c"


def test_non_latin_corpus_indexes_and_retrieves(config):
    pipeline = RAGPipeline.build(config)
    added = pipeline.index_documents(
        [
            RawDocument(source="ru", text=RUSSIAN),
            RawDocument(source="ja", text=JAPANESE),
            RawDocument(source="en", text="Dense retrieval compares embeddings."),
        ]
    )
    assert added == 3

    results = pipeline.retrieve(
        "векторы", top_k=3
    )
    assert results
    assert results[0].chunk.source == "ru"


def test_cli_prints_non_encodable_answers_without_crashing(tmp_path, capsys):
    source = tmp_path / "doc.md"
    source.write_text(
        f"# Notes\n\n{ARROW} Vectors compare by cosine similarity.\n\n# Notes two\n\n{RUSSIAN}\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "idx"

    assert main(["index", str(source), "--index-dir", str(index_dir), *BACKENDS]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "query",
                "векторы",
                "--index-dir",
                str(index_dir),
                "--json",
                *BACKENDS,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_sources"] > 0


def test_print_answer_survives_a_legacy_code_page(monkeypatch):
    """The real failure: cp1252 stdout raised UnicodeEncodeError and killed the run."""
    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", buffer)
    monkeypatch.setattr(sys, "stderr", buffer)
    configure_stdio()

    print_answer(RAGAnswer(question="q", answer=ARROW + " " + RUSSIAN), show_sources=False)
    buffer.flush()
    assert buffer.buffer.getvalue()


def test_configure_stdio_is_idempotent(monkeypatch):
    buffer = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
    monkeypatch.setattr(sys, "stdout", buffer)
    configure_stdio()
    configure_stdio()
    assert buffer.errors == "replace"
