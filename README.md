# NLP RAG

A Retrieval-Augmented Generation toolkit for question answering over your own
documents. It ships hybrid retrieval (dense + BM25 fused with reciprocal rank
fusion), heading-aware chunking, cited answers, a persistent index, a CLI and an
evaluation harness.

**It runs with nothing but NumPy.** The default stack — hashing embeddings,
brute-force cosine search and extractive generation — needs no model downloads
and no API keys. Install an optional backend and the pipeline picks it up
automatically: sentence-transformers for semantic embeddings, FAISS for the
vector index, transformers or the Claude API for generation.

```
$ python main.py demo
Question: Why combine BM25 with dense retrieval?

Answer: BM25 is a classic lexical ranking function based on term frequency and
inverse document frequency. [1] It excels at exact keyword matching, which makes
it a strong complement to dense embedding retrieval. [1] Combining both with
reciprocal rank fusion is known as hybrid retrieval. [1]
```

## Quick start

```bash
git clone https://github.com/SdSarthak/NLP-RAG.git
cd NLP-RAG
pip install -r requirements.txt

python main.py demo                       # zero-setup tour over a built-in corpus
```

Point it at your own documents:

```bash
python main.py index data/ --index-dir storage
python main.py query "Why do RAG systems use overlapping chunks?"
python main.py chat
```

Or ask questions about a single PDF in one shot:

```bash
python bot.py --pdf "NLP Journal.pdf" --question "What is sentiment analysis?"
python bot.py --pdf paper.pdf --chat
```

## How it works

```
documents ─▶ load ─▶ chunk ─▶ embed ─▶ vector index ─┐
   (.txt .md .pdf)   (heading-aware)                 ├─▶ hybrid retrieval ─▶ generate ─▶ cited answer
                     └──────────────▶ BM25 index ────┘        (RRF)
```

| Stage | What happens | Where |
| --- | --- | --- |
| Load | `.txt`, `.md`, `.pdf` (pypdf), single files or whole trees | `nlp_rag/documents.py` |
| Chunk | Sections split on Markdown headings; each chunk keeps its heading; sentence-aware packing with overlap; page furniture dropped | `nlp_rag/documents.py` |
| Embed | Signed feature hashing (default) or sentence-transformers | `nlp_rag/embeddings.py` |
| Index | NumPy brute-force cosine or FAISS `IndexFlatIP`; portable on-disk format | `nlp_rag/vectorstore.py` |
| Retrieve | Dense, BM25 (Okapi), or both fused with reciprocal rank fusion | `nlp_rag/retrieval.py` |
| Generate | Extractive (default), local transformers LM, or the Claude API | `nlp_rag/generation.py` |
| Evaluate | precision@k, recall@k, hit rate, MRR, nDCG, context recall, ROUGE-L, token F1 | `nlp_rag/evaluation.py` |

### Why hybrid retrieval

Dense embeddings match meaning, BM25 matches exact terms, and neither wins
everywhere. Reciprocal rank fusion combines the two ranked lists without needing
to normalise their very different score scales:

```
score(chunk) = Σ  1 / (k + rank_in_retriever)      # k = 60 by default
```

### Why the answers have `[1]` markers

Every answer cites the numbered passages it came from, and `query` prints those
passages underneath. The extractive generator scores each candidate sentence by
how much of the question it covers — weighting rare query terms above common
ones — plus how focused the sentence is and how highly its passage ranked.

## Usage

### Python API

```python
from nlp_rag import RAGConfig, RAGPipeline

pipeline = RAGPipeline.build(RAGConfig(top_k=5, retriever="hybrid"))
pipeline.index_paths(["data/", "notes/handbook.pdf"])
pipeline.save("storage")

result = pipeline.answer("What is retrieval-augmented generation?")
print(result.answer)
for source in result.sources:
    print(source["source"], source["score"])
```

Multi-turn conversations resolve follow-up questions against the history:

```python
from nlp_rag import ConversationalRAG

session = ConversationalRAG(pipeline)
session.chat("What is retrieval-augmented generation?")
session.chat("Why does it help?")   # expanded to include "retrieval augmented generation"
```

Reload a saved index instead of rebuilding it:

```python
pipeline = RAGPipeline.load("storage")
```

### CLI

| Command | Purpose |
| --- | --- |
| `python main.py demo [--no-chat]` | Run the built-in corpus demo; no index needed |
| `python main.py index PATH... [--append] [--chunk-size N]` | Build or extend an index |
| `python main.py query "question" [--json] [--no-sources]` | One-shot question |
| `python main.py chat` | Interactive session (`reset` clears history, `quit` exits) |
| `python main.py eval DATASET [--k 5] [--generate]` | Score a labelled dataset |
| `python main.py info` | Index statistics |

Shared flags: `--index-dir`, `--top-k`, `--retriever {hybrid,dense,bm25}`,
`--generator {extractive,transformers,anthropic}`, `--embedding-backend`,
`--vector-backend`, `-v`.

After `pip install -e .` the same CLI is available as `nlp-rag`.

### Evaluation

A dataset is JSONL (or a JSON array). Mark relevance either by chunk id or, more
conveniently, by a substring a relevant chunk must contain:

```json
{"question": "Which metrics measure retrieval quality?",
 "relevant_substrings": ["precision@k"],
 "answer": "Precision@k, recall@k, MRR and nDCG."}
```

```bash
python main.py index data/sample_corpus.md --index-dir storage
python main.py eval data/eval_sample.jsonl --k 3

Evaluated 6 question(s) at k=3
----------------------------------------------
precision@k            0.3333     # one relevant chunk per question, so 1/3 is the ceiling
hit_rate@k             1.0000
mrr                    1.0000
ndcg@k                 1.0000
context_recall         0.8831
```

Add `--generate` to also generate answers and score them against the reference
answers with ROUGE-L and token F1.

## Configuration

Every knob has a default, and can be set in code, through `NLP_RAG_*`
environment variables, or in a `.env` file. Copy `.env.example` to `.env` to see
the full list.

| Variable | Default | Notes |
| --- | --- | --- |
| `NLP_RAG_CHUNK_SIZE` / `NLP_RAG_CHUNK_OVERLAP` | `800` / `120` | Characters |
| `NLP_RAG_EMBEDDING_BACKEND` | `auto` | `auto`, `sentence-transformers`, `hashing` |
| `NLP_RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers only |
| `NLP_RAG_EMBEDDING_DIM` | `384` | hashing backend only |
| `NLP_RAG_VECTOR_BACKEND` | `auto` | `auto`, `faiss`, `numpy` |
| `NLP_RAG_RETRIEVER` | `hybrid` | `hybrid`, `dense`, `bm25` |
| `NLP_RAG_TOP_K` / `NLP_RAG_RRF_K` | `5` / `60` | Retrieval depth, RRF constant |
| `NLP_RAG_GENERATOR` | `extractive` | `extractive`, `transformers`, `anthropic` |
| `NLP_RAG_ANTHROPIC_MODEL` | `claude-opus-5` | Used when `generator=anthropic` |
| `NLP_RAG_INDEX_DIR` | `storage` | Where the index lives |
| `ANTHROPIC_API_KEY` | — | Only for `generator=anthropic` |

`auto` means "use the best backend that is installed". Optional backends never
break the pipeline: if `sentence-transformers`, `faiss`, `transformers` or
`anthropic` is missing or misconfigured, the run logs a warning and continues on
the pure-NumPy path.

## Optional backends

```bash
pip install sentence-transformers torch   # semantic embeddings
pip install faiss-cpu                     # faster vector index
pip install transformers torch            # local LM generation (GPT-2 by default)
pip install anthropic                     # Claude-backed generation
```

```bash
NLP_RAG_EMBEDDING_BACKEND=sentence-transformers python main.py index data/
python main.py query "..." --generator anthropic     # needs ANTHROPIC_API_KEY
```

Embeddings are baked into the index, so switching embedding backends requires
re-indexing; the pipeline refuses to load an index whose dimensionality does not
match the configured embedder.

## Project layout

```
nlp_rag/
  config.py       RAGConfig: defaults, env vars, validation
  documents.py    loaders (txt/md/pdf), normalisation, heading-aware chunking
  embeddings.py   hashing + sentence-transformers backends
  vectorstore.py  NumPy and FAISS stores, portable save/load
  retrieval.py    dense, BM25, hybrid (RRF)
  generation.py   extractive, transformers, Anthropic generators
  pipeline.py     RAGPipeline and ConversationalRAG
  evaluation.py   ranking + answer metrics, dataset loading
  cli.py          argparse CLI
  samples.py      built-in demo corpus
main.py           entrypoint (delegates to the CLI)
bot.py            single-PDF question answering
data/             sample corpus + example evaluation set
tests/            pytest suite
```

## Development

```bash
pip install -r requirements.txt
python -m pytest            # 97 tests, no network or model downloads required
```

The tests pin the dependency-free backends so they are deterministic.

## Requirements

- Python 3.8+
- `numpy`, `pypdf`, `python-dotenv` (see `requirements.txt`)
- Optional: `sentence-transformers`, `faiss-cpu`, `transformers`, `anthropic`

## Notes

`NLP Journal.pdf` is the research paper this project was originally built
around; `bot.py` defaults to it so `python bot.py` works out of the box.

## License

MIT
