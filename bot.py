"""Ask questions about a PDF.

This is the original notebook-style script rebuilt on top of the ``nlp_rag``
package: load a PDF, chunk it, index it, then answer questions with citations.

    python bot.py                                  # uses "NLP Journal.pdf"
    python bot.py --pdf paper.pdf --question "What is NLP?"
    python bot.py --pdf paper.pdf --chat
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nlp_rag.cli import configure_stdio, print_answer, run_chat
from nlp_rag.config import GENERATORS, RETRIEVERS, RAGConfig
from nlp_rag.documents import DocumentError, load_pdf
from nlp_rag.pipeline import RAGPipeline

DEFAULT_PDF = Path(__file__).with_name("NLP Journal.pdf")
DEFAULT_QUESTIONS = [
    "What is NLP?",
    "What is retrieval-augmented generation?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to index (default: {DEFAULT_PDF.name})",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to ask; repeat for several",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    parser.add_argument("--generator", choices=GENERATORS, default="extractive")
    parser.add_argument(
        "--chat", action="store_true", help="Start an interactive session afterwards"
    )
    parser.add_argument(
        "--save-index",
        type=Path,
        default=None,
        help="Persist the index to this directory for reuse",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_stdio()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.pdf.exists():
        print(
            f"PDF not found: {args.pdf}\n"
            "Pass one with --pdf, or run `python main.py demo` for the "
            "zero-setup sample corpus.",
            file=sys.stderr,
        )
        return 1

    config = RAGConfig.from_env(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retriever=args.retriever,
        generator=args.generator,
        top_k=args.top_k,
    )

    try:
        document = load_pdf(args.pdf)
    except DocumentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not document.text.strip():
        print(
            f"No text could be extracted from {args.pdf} - it is probably a scan.",
            file=sys.stderr,
        )
        return 1

    pipeline = RAGPipeline.build(config)
    chunks = pipeline.index_documents([document])
    print(
        f"Indexed {chunks} chunk(s) from {args.pdf.name} "
        f"({document.metadata.get('pages', '?')} pages)."
    )

    if args.save_index:
        pipeline.save(args.save_index)
        print(f"Index saved to {args.save_index}")

    for question in args.questions or DEFAULT_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print_answer(pipeline.answer(question))

    if args.chat:
        print()
        return run_chat(pipeline, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
