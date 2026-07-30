"""Evaluation harness: retrieval and answer-quality metrics."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nlp_rag.documents import Chunk
from nlp_rag.embeddings import tokenize

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Ranking metrics (binary relevance)
# ----------------------------------------------------------------------
def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return 1.0 if any(item in relevant for item in list(retrieved)[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(list(retrieved)[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


# ----------------------------------------------------------------------
# Answer metrics
# ----------------------------------------------------------------------
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over whitespace/word tokens."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1, the standard extractive-QA overlap metric."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred_tokens))
    recall = len(common) / len(set(ref_tokens))
    return 2 * precision * recall / (precision + recall)


def context_recall(answer_reference: str, contexts: Sequence[str]) -> float:
    """Fraction of reference-answer tokens present in the retrieved context."""
    ref_tokens = set(tokenize(answer_reference))
    if not ref_tokens:
        return 0.0
    context_tokens = set()
    for context in contexts:
        context_tokens.update(tokenize(context))
    return len(ref_tokens & context_tokens) / len(ref_tokens)


# ----------------------------------------------------------------------
# Dataset + report
# ----------------------------------------------------------------------
@dataclass
class EvalExample:
    """One labelled question.

    Relevance can be expressed either as explicit chunk ids or as substrings that
    a relevant chunk must contain (handy when chunk ids are not known upfront).
    """

    question: str
    relevant_ids: List[str] = field(default_factory=list)
    relevant_substrings: List[str] = field(default_factory=list)
    answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalExample":
        if "question" not in data:
            raise ValueError("eval example is missing the 'question' field")
        return cls(
            question=str(data["question"]),
            relevant_ids=[str(x) for x in data.get("relevant_ids", [])],
            relevant_substrings=[str(x) for x in data.get("relevant_substrings", [])],
            answer=data.get("answer"),
        )

    def matches(self, chunk: Chunk) -> bool:
        if chunk.id in self.relevant_ids:
            return True
        lowered = chunk.text.lower()
        return any(s.lower() in lowered for s in self.relevant_substrings)


def load_eval_dataset(path: Path | str) -> List[EvalExample]:
    """Read a JSONL (or JSON list) evaluation dataset."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [EvalExample.from_dict(record) for record in records]


@dataclass
class EvaluationReport:
    """Averaged metrics plus per-example detail."""

    k: int
    num_examples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    per_example: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "num_examples": self.num_examples,
            "metrics": {name: round(value, 4) for name, value in self.metrics.items()},
            "per_example": self.per_example,
        }

    def format(self) -> str:
        lines = [
            f"Evaluated {self.num_examples} question(s) at k={self.k}",
            "-" * 46,
        ]
        for name, value in self.metrics.items():
            lines.append(f"{name:<22} {value:.4f}")
        return "\n".join(lines)


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    pipeline: Any,
    examples: Sequence[EvalExample],
    k: int = 5,
    generate_answers: bool = False,
) -> EvaluationReport:
    """Run retrieval (and optionally generation) over a labelled dataset.

    ``pipeline`` is any object exposing ``retrieve(question, top_k)`` and, when
    ``generate_answers`` is set, ``answer(question, top_k)``.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    buckets: Dict[str, List[float]] = {
        "precision@k": [],
        "recall@k": [],
        "hit_rate@k": [],
        "mrr": [],
        "ndcg@k": [],
        "context_recall": [],
        "answer_rouge_l": [],
        "answer_token_f1": [],
    }
    per_example: List[Dict[str, Any]] = []

    for example in examples:
        results = pipeline.retrieve(example.question, k)
        retrieved_ids = [result.chunk.id for result in results]
        relevant_ids = {
            result.chunk.id for result in results if example.matches(result.chunk)
        }
        relevant_ids.update(example.relevant_ids)

        record: Dict[str, Any] = {
            "question": example.question,
            "retrieved": retrieved_ids,
        }

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        hit = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k)
        buckets["precision@k"].append(precision)
        buckets["hit_rate@k"].append(hit)
        buckets["mrr"].append(rr)
        buckets["ndcg@k"].append(ndcg)
        record.update(
            {"precision@k": precision, "hit_rate@k": hit, "mrr": rr, "ndcg@k": ndcg}
        )

        if example.relevant_ids:
            recall = recall_at_k(retrieved_ids, example.relevant_ids, k)
            buckets["recall@k"].append(recall)
            record["recall@k"] = recall

        if example.answer:
            recall_ctx = context_recall(
                example.answer, [result.chunk.text for result in results]
            )
            buckets["context_recall"].append(recall_ctx)
            record["context_recall"] = recall_ctx

            if generate_answers:
                generated = pipeline.answer(example.question, k).answer
                rouge = rouge_l(generated, example.answer)
                f1 = token_f1(generated, example.answer)
                buckets["answer_rouge_l"].append(rouge)
                buckets["answer_token_f1"].append(f1)
                record.update(
                    {
                        "generated_answer": generated,
                        "answer_rouge_l": rouge,
                        "answer_token_f1": f1,
                    }
                )

        per_example.append(record)

    metrics = {
        name: _average(values) for name, values in buckets.items() if values
    }
    return EvaluationReport(
        k=k,
        num_examples=len(examples),
        metrics=metrics,
        per_example=per_example,
    )
