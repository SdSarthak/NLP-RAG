# Sample corpus

A small knowledge base used by the tests, the demo and the example evaluation
set. Index it with:

    python main.py index data/sample_corpus.md --index-dir storage

## Natural language processing

Natural Language Processing (NLP) is a branch of artificial intelligence that
helps computers understand, interpret and manipulate human language. Classic NLP
tasks include tokenization, part-of-speech tagging, named entity recognition,
dependency parsing, machine translation and question answering.

## Retrieval-augmented generation

Retrieval-Augmented Generation (RAG) combines information retrieval with text
generation. A retriever first finds passages relevant to the user question, and a
generator then writes an answer conditioned on those passages. Because the answer
is grounded in retrieved documents, RAG reduces hallucination and lets the system
cite its sources.

## Pipeline stages

A RAG pipeline usually has five stages: query processing, document retrieval,
context extraction, response generation and answer synthesis. Documents are split
into chunks, embedded into vectors and stored in a vector index so that
semantically similar passages can be found quickly.

## Chunking

Chunking strategy matters. Chunks that are too large dilute the embedding and
waste context, while chunks that are too small lose the surrounding meaning.
Overlapping chunks of a few hundred characters are a common default because the
overlap preserves sentences that straddle a chunk boundary.

## Embeddings and semantic search

Vector embeddings represent text as numerical vectors in a high-dimensional
space. Semantic search compares those vectors with cosine similarity, so it can
match passages that mean the same thing even when they share no keywords.
Sentence-BERT is a popular model family for producing such sentence embeddings.

## Lexical retrieval and hybrid search

BM25 is a classic lexical ranking function based on term frequency and inverse
document frequency. It excels at exact keyword matching, which makes it a strong
complement to dense embedding retrieval. Fusing the two ranked lists with
reciprocal rank fusion is known as hybrid retrieval, and it is usually more
robust than either retriever alone.

## Transformers

Transformers are a neural network architecture built on self-attention. They
process every token in a sequence in parallel and learn how strongly each token
should attend to the others, which is why they replaced recurrent networks for
most NLP tasks. BERT is an encoder-only transformer, while GPT models are
decoder-only transformers trained to predict the next token.

## Evaluation

Retrieval quality is measured with ranking metrics such as precision@k,
recall@k, mean reciprocal rank and nDCG. Generation quality is measured with
overlap metrics such as ROUGE, BLEU and token-level F1 against reference answers,
and with grounding checks that verify the answer is supported by the retrieved
context.
