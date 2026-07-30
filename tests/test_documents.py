from pathlib import Path

import pytest

from nlp_rag.documents import (
    Chunk,
    DocumentError,
    RawDocument,
    chunk_documents,
    is_meaningful,
    load_directory,
    load_document,
    normalize_text,
    split_sections,
    split_text,
)


def test_split_text_respects_chunk_size():
    text = " ".join(f"Sentence number {i}." for i in range(120))
    chunks = split_text(text, chunk_size=200, chunk_overlap=40)

    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_split_text_overlaps_consecutive_chunks():
    text = " ".join(f"Token{i} words here." for i in range(60))
    chunks = split_text(text, chunk_size=150, chunk_overlap=50)

    assert len(chunks) > 1
    tail_words = set(chunks[0].split()[-4:])
    assert tail_words & set(chunks[1].split())


def test_split_text_handles_oversized_unit():
    text = "x" * 1000
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "".join(chunks).count("x") >= 1000


def test_split_text_empty_and_whitespace():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_split_text_validates_arguments():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0)


def test_split_sections_finds_markdown_headings():
    text = "intro line\n\n# One\n\nbody one\n\n## Two\n\nbody two"
    sections = split_sections(text)

    assert [heading for heading, _ in sections] == [None, "# One", "## Two"]
    assert sections[1][1] == "body one"


def test_chunks_carry_their_section_heading():
    text = "## Transformers\n\nThey rely on self-attention.\n\n## BM25\n\nIt is lexical."
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)

    assert chunks == [
        "## Transformers\nThey rely on self-attention.",
        "## BM25\nIt is lexical.",
    ]


def test_headings_can_be_dropped():
    text = "## Transformers\n\nThey rely on self-attention."
    chunks = split_text(text, chunk_size=200, chunk_overlap=20, keep_headings=False)
    assert chunks == ["They rely on self-attention."]


def test_normalize_text_preserves_paragraphs():
    normalized = normalize_text("a  b\n\n\n\nc\td")
    assert normalized == "a b\n\nc d"


def test_dot_leaders_are_collapsed():
    normalized = normalize_text("Introduction......................... 12")
    assert "...." not in normalized
    assert normalized == "Introduction 12"


def test_is_meaningful_rejects_page_furniture():
    assert is_meaningful("Semantic search compares vectors by meaning.")
    assert is_meaningful("Cats purr.")
    assert not is_meaningful("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16")
    assert not is_meaningful("   ")
    assert not is_meaningful("x")


def test_chunk_documents_drops_noise_chunks():
    documents = [RawDocument(source="table", text="1 2 3 4 5 6 7 8 9 10 11 12")]
    assert chunk_documents(documents) == []
    assert len(chunk_documents(documents, drop_noise=False)) == 1


def test_chunk_documents_produces_stable_ids():
    documents = [RawDocument(source="doc", text="One. Two. Three.")]
    chunks = chunk_documents(documents, chunk_size=12, chunk_overlap=4)

    assert [chunk.id for chunk in chunks] == [
        f"doc::{i}" for i in range(len(chunks))
    ]
    assert all(chunk.source == "doc" for chunk in chunks)


def test_chunk_roundtrip_dict():
    chunk = chunk_documents([RawDocument(source="s", text="Hello world.")])[0]
    assert Chunk.from_dict(chunk.to_dict()) == chunk


def test_load_document_and_directory(tmp_path):
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta content", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    document = load_document(tmp_path / "a.txt")
    assert document.text == "alpha content"

    documents = load_directory(tmp_path)
    assert len(documents) == 2
    assert {Path(d.source).name for d in documents} == {"a.txt", "b.md"}


def test_load_document_errors(tmp_path):
    with pytest.raises(DocumentError):
        load_document(tmp_path / "missing.txt")

    unsupported = tmp_path / "thing.bin"
    unsupported.write_bytes(b"\x00")
    with pytest.raises(DocumentError):
        load_document(unsupported)
