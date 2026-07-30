"""NLP RAG entrypoint.

Running this file with no arguments starts the zero-setup demo; every CLI
subcommand is also available:

    python main.py demo
    python main.py index docs/ --index-dir storage
    python main.py query "What is retrieval-augmented generation?"
    python main.py chat
    python main.py eval data/eval_sample.jsonl --k 3
"""

from __future__ import annotations

import sys

from nlp_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
