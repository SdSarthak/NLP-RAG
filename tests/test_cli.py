import json

import pytest

from nlp_rag.cli import build_parser, main

BACKENDS = ["--embedding-backend", "hashing", "--vector-backend", "numpy"]


@pytest.mark.parametrize(
    "argv",
    [
        ["index", "docs"],
        ["query", "question"],
        ["chat"],
        ["eval", "data.jsonl"],
        ["info"],
        ["demo"],
    ],
)
def test_parser_exposes_all_commands(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.func)


def test_demo_runs_without_an_index(capsys):
    assert main(["demo", "--no-chat", *BACKENDS]) == 0
    output = capsys.readouterr().out
    assert "demo over the built-in sample corpus" in output
    assert "Answer:" in output


def test_index_then_query(tmp_path, corpus_path, capsys):
    index_dir = tmp_path / "idx"

    assert main(["index", str(corpus_path), "--index-dir", str(index_dir), *BACKENDS]) == 0
    assert (index_dir / "meta.json").exists()
    assert "Indexed" in capsys.readouterr().out

    assert (
        main(
            [
                "query",
                "Why combine BM25 with dense retrieval?",
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
    assert payload["answer"]


def test_index_append_grows_the_index(tmp_path, corpus_path, capsys):
    index_dir = tmp_path / "idx"
    extra = tmp_path / "extra.md"
    extra.write_text("Quokkas are famously photogenic marsupials.", encoding="utf-8")

    main(["index", str(corpus_path), "--index-dir", str(index_dir), *BACKENDS])
    capsys.readouterr()

    main(["index", str(extra), "--index-dir", str(index_dir), "--append", *BACKENDS])
    capsys.readouterr()

    assert main(["info", "--index-dir", str(index_dir), *BACKENDS]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 2


def test_eval_command(tmp_path, corpus_path, capsys):
    index_dir = tmp_path / "idx"
    dataset = corpus_path.parent / "eval_sample.jsonl"

    main(["index", str(corpus_path), "--index-dir", str(index_dir), *BACKENDS])
    capsys.readouterr()

    assert (
        main(
            [
                "eval",
                str(dataset),
                "--index-dir",
                str(index_dir),
                "--k",
                "3",
                "--json",
                *BACKENDS,
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["num_examples"] == 6
    assert report["metrics"]["hit_rate@k"] >= 0.8


def test_query_without_index_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["query", "anything", "--index-dir", str(tmp_path / "missing")])


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["nonsense"])
