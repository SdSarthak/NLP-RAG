import json
import logging
import math
import types

import pytest

from nlp_rag.documents import Chunk
from nlp_rag.retrieval import RetrievedChunk
from nlp_rag.evaluation import (
    EvalDatasetError,
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


# ----------------------------------------------------------------------
# Ground truth must come from the corpus, never from the prediction
# ----------------------------------------------------------------------
class _FakePipeline:
    """Retriever that always returns the wrong chunks, over a known corpus."""

    def __init__(self, corpus, returns):
        self.store = types.SimpleNamespace(chunks=corpus)
        self._returns = returns

    def retrieve(self, question, top_k):
        return [
            RetrievedChunk(chunk=chunk, score=1.0 / rank, rank=rank, retriever="fake")
            for rank, chunk in enumerate(self._returns[:top_k], start=1)
        ]


def _chunk(identifier, text):
    return Chunk(id=identifier, text=text, source="s", index=0)


def test_missing_a_relevant_chunk_is_scored_as_a_miss():
    """The old harness derived relevance from the results, so this scored 1.0."""
    relevant = _chunk("s::0", "Reciprocal rank fusion merges two rankings.")
    noise = [_chunk(f"s::{i}", "Completely unrelated filler.") for i in range(1, 4)]
    pipeline = _FakePipeline([relevant, *noise], noise)

    example = EvalExample(
        question="q", relevant_substrings=["reciprocal rank fusion"]
    )
    report = evaluate(pipeline, [example], k=3)

    assert report.per_example[0]["num_relevant"] == 1
    assert report.metrics["hit_rate@k"] == 0.0
    assert report.metrics["recall@k"] == 0.0
    assert report.metrics["mrr"] == 0.0
    assert report.metrics["ndcg@k"] == 0.0


def test_ndcg_accounts_for_relevant_chunks_that_were_not_retrieved():
    """Two relevant chunks exist; only the second-ranked one is returned."""
    hits = [_chunk("s::0", "alpha marker"), _chunk("s::1", "alpha marker again")]
    noise = _chunk("s::2", "filler")
    pipeline = _FakePipeline([*hits, noise], [noise, hits[1]])

    example = EvalExample(question="q", relevant_substrings=["alpha marker"])
    report = evaluate(pipeline, [example], k=2)

    assert report.per_example[0]["num_relevant"] == 2
    # dcg = 1/log2(3) with an ideal of 1 + 1/log2(3): a genuine penalty.
    assert report.metrics["ndcg@k"] == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )
    assert report.metrics["recall@k"] == pytest.approx(0.5)


def test_recall_is_reported_for_substring_labels(pipeline):
    """Recall used to be silently omitted unless explicit ids were supplied."""
    example = EvalExample(
        question="What is retrieval-augmented generation?",
        relevant_substrings=["Retrieval-Augmented Generation (RAG) combines"],
    )
    report = evaluate(pipeline, [example], k=3)
    assert "recall@k" in report.metrics


def test_explicit_ids_outside_the_index_are_flagged(pipeline, caplog):
    example = EvalExample(question="anything", relevant_ids=["not::a::chunk"])
    with caplog.at_level(logging.WARNING):
        evaluate(pipeline, [example], k=3)
    assert "not in the index" in caplog.text


def test_unmatched_labels_are_flagged(pipeline, caplog):
    example = EvalExample(question="anything", relevant_substrings=["zzz nothing zzz"])
    with caplog.at_level(logging.WARNING):
        report = evaluate(pipeline, [example], k=3)
    assert report.per_example[0]["num_relevant"] == 0
    assert "matches no chunk" in caplog.text


def test_evaluate_rejects_a_pipeline_without_retrieve():
    with pytest.raises(TypeError):
        evaluate(object(), [], k=3)


def test_generate_answers_requires_an_answer_method():
    pipeline = _FakePipeline([], [])
    with pytest.raises(TypeError):
        evaluate(pipeline, [], k=3, generate_answers=True)


# ----------------------------------------------------------------------
# Dataset parsing
# ----------------------------------------------------------------------
def test_malformed_jsonl_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"question": "ok"}\n{"question": "broken"\n', encoding="utf-8"
    )
    with pytest.raises(EvalDatasetError) as excinfo:
        load_eval_dataset(path)
    assert "bad.jsonl:2" in str(excinfo.value)


def test_missing_dataset_is_reported_clearly(tmp_path):
    with pytest.raises(EvalDatasetError) as excinfo:
        load_eval_dataset(tmp_path / "nope.jsonl")
    assert "No such evaluation dataset" in str(excinfo.value)


def test_json_array_of_non_objects_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('["just a string"]', encoding="utf-8")
    with pytest.raises(EvalDatasetError):
        load_eval_dataset(path)


def test_empty_dataset_is_not_an_error(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    assert load_eval_dataset(path) == []
