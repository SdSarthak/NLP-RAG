"""Evaluation harness: retrieval and answer-quality metrics."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

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


class EvalDatasetError(ValueError):
    """Raised when an evaluation dataset cannot be parsed."""


def load_eval_dataset(path: Path | str) -> List[EvalExample]:
    """Read a JSONL (or JSON list) evaluation dataset.

    Parse failures name the offending line so a one-character typo in a large
    dataset does not turn into an unlocatable ``JSONDecodeError``.
    """
    path = Path(path)
    if not path.exists():
        raise EvalDatasetError(f"No such evaluation dataset: {path}")
    if path.is_dir():
        raise EvalDatasetError(f"Evaluation dataset is a directory: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []

    records: List[Any]
    if raw.lstrip().startswith("["):
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(records, list):
            raise EvalDatasetError(f"{path} must contain a list of examples")
    else:
        records = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvalDatasetError(
                    f"{path}:{number} is not valid JSON: {exc}"
                ) from exc

    examples: List[EvalExample] = []
    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise EvalDatasetError(
                f"{path}: example {number} must be an object, got {type(record).__name__}"
            )
        try:
            examples.append(EvalExample.from_dict(record))
        except ValueError as exc:
            raise EvalDatasetError(f"{path}: example {number}: {exc}") from exc
    return examples


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


def corpus_chunks(pipeline: Any) -> Optional[List[Chunk]]:
    """Best-effort access to every indexed chunk, for grounding relevance."""
    store = getattr(pipeline, "store", None)
    chunks = getattr(store, "chunks", None)
    if chunks is None:
        chunks = getattr(pipeline, "chunks", None)
    if chunks is None:
        return None
    return list(chunks)


def resolve_relevant(
    example: EvalExample, corpus: Optional[Sequence[Chunk]]
) -> Set[str]:
    """Ground-truth chunk ids for ``example``.

    Substring labels are resolved against the **whole corpus**. Resolving them
    against the retrieved results instead (as this harness originally did) makes
    the ground truth a subset of the prediction, which silently inflates nDCG and
    makes recall unmeasurable - a retriever that misses a relevant chunk is never
    penalised for it because that chunk never enters the relevance set.
    """
    relevant: Set[str] = set(example.relevant_ids)
    if corpus is not None and example.relevant_substrings:
        relevant.update(chunk.id for chunk in corpus if example.matches(chunk))
    return relevant


def evaluate(
    pipeline: Any,
    examples: Sequence[EvalExample],
    k: int = 5,
    generate_answers: bool = False,
    corpus: Optional[Sequence[Chunk]] = None,
) -> EvaluationReport:
    """Run retrieval (and optionally generation) over a labelled dataset.

    ``pipeline`` is any object exposing ``retrieve(question, top_k)`` and, when
    ``generate_answers`` is set, ``answer(question, top_k)``. ``corpus`` defaults
    to every chunk the pipeline has indexed and is what substring labels are
    matched against.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not hasattr(pipeline, "retrieve"):
        raise TypeError("pipeline must expose retrieve(question, top_k)")
    if generate_answers and not hasattr(pipeline, "answer"):
        raise TypeError(
            "generate_answers=True requires a pipeline exposing answer(question, top_k)"
        )

    if corpus is None:
        corpus = corpus_chunks(pipeline)
    if corpus is None:
        logger.warning(
            "Could not read the indexed chunks from %s; substring labels will be "
            "matched against retrieved results only, which overstates ranking "
            "quality. Pass corpus=... to measure it properly.",
            type(pipeline).__name__,
        )
    corpus_ids = {chunk.id for chunk in corpus} if corpus is not None else None

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

    unlabelled = 0
    for example in examples:
        results = pipeline.retrieve(example.question, k)
        retrieved_ids = [result.chunk.id for result in results]

        relevant_ids = resolve_relevant(example, corpus)
        if corpus is None:
            # Degraded mode: no corpus available, fall back to the retrieved set.
            relevant_ids.update(
                result.chunk.id for result in results if example.matches(result.chunk)
            )
        elif corpus_ids is not None:
            missing = [i for i in example.relevant_ids if i not in corpus_ids]
            if missing:
                logger.warning(
                    "Question %r labels chunk id(s) %s that are not in the index",
                    example.question[:60],
                    ", ".join(sorted(missing)[:5]),
                )

        record: Dict[str, Any] = {
            "question": example.question,
            "retrieved": retrieved_ids,
            "num_relevant": len(relevant_ids),
        }

        if not relevant_ids:
            unlabelled += 1
            logger.warning(
                "Question %r matches no chunk in the index; it is scored as a miss",
                example.question[:60],
            )

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

        if relevant_ids:
            recall = recall_at_k(retrieved_ids, relevant_ids, k)
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

    if unlabelled:
        logger.warning(
            "%d of %d question(s) had no relevant chunk in the index - check the "
            "dataset labels against what was actually indexed",
            unlabelled,
            len(examples),
        )

    metrics = {
        name: _average(values) for name, values in buckets.items() if values
    }
    return EvaluationReport(
        k=k,
        num_examples=len(examples),
        metrics=metrics,
        per_example=per_example,
    )
