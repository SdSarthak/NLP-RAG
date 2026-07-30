import json
import math

import pytest

from nlp_rag.evaluation import (
    EvalExample,
    context_recall,
    evaluate,
    hit_rate_at_k,
    load_eval_dataset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
    token_f1,
)


def test_precision_and_recall():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d", "z"]

    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, [], 4) == 0.0


def test_hit_rate_and_reciprocal_rank():
    assert hit_rate_at_k(["a", "b"], ["b"], 2) == 1.0
    assert hit_rate_at_k(["a", "b"], ["z"], 2) == 0.0
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["z"]) == 0.0


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_ndcg_penalises_late_hits():
    early = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert early > late
    assert late == pytest.approx(1 / math.log2(4))


def test_rouge_l_and_token_f1():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("", "anything") == 0.0
    assert 0.0 < rouge_l("the cat sat on a mat", "the cat sat") < 1.0
    assert token_f1("alpha beta", "beta gamma") == pytest.approx(0.5)
    assert token_f1("alpha", "gamma") == 0.0


def test_context_recall():
    assert context_recall("alpha beta", ["alpha gamma", "beta"]) == pytest.approx(1.0)
    assert context_recall("alpha beta", ["gamma"]) == 0.0


def test_eval_example_matching():
    from nlp_rag.documents import Chunk

    example = EvalExample(
        question="q", relevant_ids=["x::0"], relevant_substrings=["Reciprocal Rank"]
    )
    assert example.matches(Chunk(id="x::0", text="unrelated", source="x", index=0))
    assert example.matches(
        Chunk(id="y::1", text="uses reciprocal rank fusion", source="y", index=1)
    )
    assert not example.matches(Chunk(id="z::2", text="nothing", source="z", index=2))


def test_load_eval_dataset_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"question": "one", "relevant_substrings": ["a"]},
                {"question": "two", "relevant_ids": ["c::0"], "answer": "yes"},
            ]
        ),
        encoding="utf-8",
    )

    examples = load_eval_dataset(path)
    assert [e.question for e in examples] == ["one", "two"]
    assert examples[1].answer == "yes"


def test_load_eval_dataset_json_array(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "one"}]), encoding="utf-8")
    assert len(load_eval_dataset(path)) == 1


def test_load_eval_dataset_requires_question(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"answer": "no question"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_dataset(path)


def test_evaluate_over_pipeline(pipeline):
    examples = [
        EvalExample(
            question="What is retrieval-augmented generation?",
            relevant_substrings=["Retrieval-Augmented Generation (RAG) combines"],
            answer="RAG combines retrieval with generation and cites its sources.",
        ),
        EvalExample(
            question="Which metrics measure retrieval quality?",
            relevant_substrings=["precision@k"],
        ),
    ]

    report = evaluate(pipeline, examples, k=3, generate_answers=True)

    assert report.num_examples == 2
    assert report.k == 3
    assert report.metrics["hit_rate@k"] == pytest.approx(1.0)
    assert 0.0 < report.metrics["mrr"] <= 1.0
    assert "answer_rouge_l" in report.metrics
    assert "context_recall" in report.metrics
    assert len(report.per_example) == 2
    assert "Evaluated 2 question(s)" in report.format()
    assert report.to_dict()["metrics"]["hit_rate@k"] == 1.0


def test_evaluate_rejects_bad_k(pipeline):
    with pytest.raises(ValueError):
        evaluate(pipeline, [], k=0)
