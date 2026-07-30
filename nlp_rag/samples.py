"""A tiny built-in corpus so the demo works with zero setup."""

from __future__ import annotations

from typing import List

from nlp_rag.documents import RawDocument

SAMPLE_DOCUMENTS: List[str] = [
    "Natural Language Processing (NLP) is a branch of artificial intelligence that "
    "helps computers understand, interpret and manipulate human language. It covers "
    "tasks such as tokenization, part-of-speech tagging, named entity recognition, "
    "parsing, machine translation and question answering.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with text "
    "generation. A retriever first finds passages relevant to the user question, and "
    "a generator then writes an answer conditioned on those passages. Because the "
    "answer is grounded in retrieved documents, RAG reduces hallucination and lets a "
    "system cite its sources.",
    "A RAG pipeline usually has five stages: query processing, document retrieval, "
    "context extraction, response generation and answer synthesis. Documents are "
    "split into chunks, embedded into vectors and stored in a vector index so that "
    "semantically similar passages can be found quickly.",
    "Vector embeddings represent text as numerical vectors in a high-dimensional "
    "space. Semantic search compares those vectors with cosine similarity, so it can "
    "match passages that mean the same thing even when they share no keywords.",
    "BM25 is a classic lexical ranking function based on term frequency and inverse "
    "document frequency. It excels at exact keyword matching, which makes it a strong "
    "complement to dense embedding retrieval. Combining both with reciprocal rank "
    "fusion is known as hybrid retrieval.",
    "Transformers are a neural network architecture built on self-attention. They "
    "process every token in a sequence in parallel and learn how strongly each token "
    "should attend to the others, which is why they replaced recurrent networks for "
    "most NLP tasks.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained "
    "encoder that reads text in both directions at once. It is typically fine-tuned "
    "for classification, extraction and sentence-embedding tasks such as SBERT.",
    "GPT (Generative Pre-trained Transformer) models are decoder-only transformers "
    "trained to predict the next token. They excel at open-ended text generation and "
    "are commonly used as the generation component of a RAG system.",
    "Retrieval quality is measured with ranking metrics such as precision@k, "
    "recall@k, mean reciprocal rank and nDCG. Generation quality is measured with "
    "overlap metrics such as ROUGE, BLEU and token-level F1 against reference "
    "answers.",
    "Chunking strategy matters: chunks that are too large dilute the embedding and "
    "waste context, while chunks that are too small lose the surrounding meaning. "
    "Overlapping chunks of a few hundred characters are a common default because they "
    "preserve sentences that straddle a boundary.",
]


def sample_documents() -> List[RawDocument]:
    """Return the built-in corpus as :class:`RawDocument` objects."""
    return [
        RawDocument(
            source=f"sample:{index}",
            text=text,
            metadata={"corpus": "builtin"},
        )
        for index, text in enumerate(SAMPLE_DOCUMENTS)
    ]
