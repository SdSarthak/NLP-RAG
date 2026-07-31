"""Answer generation backends.

* ``ExtractiveGenerator`` (default) - no model downloads, no API keys. It selects
  and stitches the most query-relevant sentences from the retrieved context and
  cites them.
* ``TransformersGenerator`` - local Hugging Face causal LM (e.g. GPT-2).
* ``AnthropicGenerator`` - grounded answers from the Claude API.
"""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from nlp_rag.config import RAGConfig
from nlp_rag.documents import SENTENCE_RE
from nlp_rag.retrieval import RetrievedChunk, content_tokens

logger = logging.getLogger(__name__)

_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+")

NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in the knowledge base to answer that. "
    "Try rephrasing the question or indexing more documents."
)

SYSTEM_PROMPT = (
    "You answer questions strictly from the numbered context passages you are "
    "given. Cite the passages you rely on inline as [1], [2], and so on. If the "
    "context does not contain the answer, say so plainly instead of guessing. "
    "Keep answers concise and factual."
)


class GenerationError(RuntimeError):
    """Raised when an answer could not be generated."""


def format_context(
    results: Sequence[RetrievedChunk], max_chars: int = 6000
) -> str:
    """Render retrieved chunks as a numbered context block."""
    parts: List[str] = []
    used = 0
    for position, result in enumerate(results, start=1):
        header = f"[{position}] (source: {result.chunk.source})\n"
        body = result.chunk.text.strip()
        budget = max_chars - used - len(header)
        if budget <= 0:
            break
        if len(body) > budget:
            body = body[:budget].rstrip() + "..."
        block = header + body
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Prompt template shared by the LM-backed generators."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


class Generator(ABC):
    """Turns a question plus retrieved context into an answer."""

    name = "generator"

    @abstractmethod
    def generate(self, question: str, results: Sequence[RetrievedChunk]) -> str:
        """Generate an answer for ``question`` grounded in ``results``."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


class ExtractiveGenerator(Generator):
    """Query-focused extractive summariser with inline citations."""

    name = "extractive"

    def __init__(self, max_sentences: int = 3, max_context_chars: int = 6000) -> None:
        self.max_sentences = max(1, max_sentences)
        self.max_context_chars = max_context_chars

    @staticmethod
    def _sentences(text: str) -> List[str]:
        """Sentences worth quoting: Markdown headings are context, not answers."""
        sentences: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or _HEADING_LINE_RE.match(line):
                continue
            sentences.extend(
                part.strip() for part in SENTENCE_RE.split(line) if part.strip()
            )
        return sentences

    @staticmethod
    def _term_weights(
        query_terms: Sequence[str], results: Sequence[RetrievedChunk]
    ) -> Dict[str, float]:
        """Weight query terms by how rare they are across the retrieved context.

        A term that appears in only one passage is far more discriminative than
        one that appears in all of them, so it should dominate sentence scoring.
        """
        total = len(results) or 1
        chunk_terms = [set(content_tokens(r.chunk.text)) for r in results]
        weights: Dict[str, float] = {}
        for term in query_terms:
            df = sum(1 for terms in chunk_terms if term in terms)
            weights[term] = math.log(1.0 + total / (df if df else 0.5))
        return weights

    def generate(self, question: str, results: Sequence[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_ANSWER

        query_terms = set(content_tokens(question))
        weights = self._term_weights(sorted(query_terms), results)
        total_weight = sum(weights.values())
        scored: List[tuple] = []

        for position, result in enumerate(results, start=1):
            for order, sentence in enumerate(self._sentences(result.chunk.text)):
                sentence_terms = set(content_tokens(sentence))
                if not sentence_terms:
                    continue
                overlap = query_terms & sentence_terms
                matched_weight = sum(weights[term] for term in overlap)
                coverage = matched_weight / total_weight if total_weight else 0.0
                density = len(overlap) / len(sentence_terms)
                rank_bonus = 1.0 / position
                score = 2.0 * coverage + density + 0.5 * rank_bonus
                scored.append((score, position, order, sentence))

        if not scored:
            return NO_CONTEXT_ANSWER

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        best_score = scored[0][0]
        if best_score <= 0.5:  # nothing but the rank bonus matched
            leading = self._sentences(results[0].chunk.text)[: self.max_sentences]
            if not leading:
                return NO_CONTEXT_ANSWER
            return (
                "The knowledge base does not directly answer that. "
                f"The closest passage says: {' '.join(leading)} [1]"
            )

        selected: List[tuple] = []
        seen: set = set()
        for score, position, order, sentence in scored:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append((score, position, order, sentence))
            if len(selected) >= self.max_sentences:
                break

        # Lead with the best-matching sentence, then keep sentences from the same
        # passage in their original reading order.
        lead_position = selected[0][1]
        selected.sort(
            key=lambda item: (item[1] != lead_position, item[1], item[2])
        )
        answer = " ".join(
            f"{sentence} [{position}]" for _score, position, _order, sentence in selected
        )
        if len(answer) > self.max_context_chars:
            answer = answer[: self.max_context_chars].rstrip() + "..."
        return answer

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "max_sentences": self.max_sentences}


class TransformersGenerator(Generator):
    """Local Hugging Face causal language model."""

    name = "transformers"

    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 160,
        max_context_chars: int = 6000,
        device: int = -1,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is not installed. Run `pip install transformers torch` "
                "or use the extractive generator."
            ) from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self._pipeline = pipeline(
            "text-generation", model=model_name, device=device
        )

    def generate(self, question: str, results: Sequence[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_ANSWER
        prompt = build_prompt(
            question, format_context(results, self.max_context_chars)
        )
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )
        except Exception as exc:
            raise GenerationError(f"transformers generation failed: {exc}") from exc

        text = (outputs[0].get("generated_text") or "").strip()
        return text or NO_CONTEXT_ANSWER

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model_name}


class AnthropicGenerator(Generator):
    """Grounded answers from the Claude API.

    Requires ``ANTHROPIC_API_KEY`` in the environment (or a `.env` file).
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        max_tokens: int = 4096,
        max_context_chars: int = 6000,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "anthropic is not installed. Run `pip install anthropic` "
                "or use the extractive generator."
            ) from exc

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise GenerationError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill "
                "it in, or choose a different generator."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars
        self._client = anthropic.Anthropic()

    def generate(self, question: str, results: Sequence[RetrievedChunk]) -> str:
        if not results:
            return NO_CONTEXT_ANSWER

        context = format_context(results, self.max_context_chars)
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above, citing passages as [1], [2], ..."
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            raise GenerationError(f"Claude API request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            return (
                "The model declined to answer this request. "
                "Try rephrasing the question."
            )

        text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        return text or NO_CONTEXT_ANSWER

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model}


def get_generator(config: RAGConfig) -> Generator:
    """Instantiate the generator described by ``config``.

    Optional backends fall back to the extractive generator with a warning rather
    than crashing an otherwise working pipeline.
    """
    if config.generator == "transformers":
        try:
            return TransformersGenerator(
                model_name=config.transformers_model,
                max_new_tokens=config.transformers_max_new_tokens,
                max_context_chars=config.max_context_chars,
            )
        except (ImportError, GenerationError) as exc:
            logger.warning("Falling back to extractive generation: %s", exc)

    elif config.generator == "anthropic":
        try:
            return AnthropicGenerator(
                model=config.anthropic_model,
                max_tokens=config.anthropic_max_tokens,
                max_context_chars=config.max_context_chars,
            )
        except (ImportError, GenerationError) as exc:
            logger.warning("Falling back to extractive generation: %s", exc)

    return ExtractiveGenerator(max_context_chars=config.max_context_chars)
