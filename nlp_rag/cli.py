"""Command line interface for the NLP RAG toolkit.

    python main.py demo
    python main.py index "NLP Journal.pdf" --index-dir storage
    python main.py query "What is retrieval-augmented generation?"
    python main.py chat
    python main.py eval data/eval_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from nlp_rag.config import (
    EMBEDDING_BACKENDS,
    GENERATORS,
    RETRIEVERS,
    VECTOR_BACKENDS,
    RAGConfig,
)
from nlp_rag.evaluation import evaluate, load_eval_dataset
from nlp_rag.pipeline import ConversationalRAG, RAGAnswer, RAGPipeline
from nlp_rag.samples import sample_documents
from nlp_rag.vectorstore import META_FILE

logger = logging.getLogger(__name__)

BANNER = "=" * 70


def configure_stdio() -> None:
    """Make stdout/stderr survive text the local encoding cannot represent.

    On Windows the console encoding defaults to a legacy code page (cp1252),
    so printing an answer that quotes an arrow, a dash or any non-Latin script
    raises ``UnicodeEncodeError`` and kills the command. Redirected output is
    switched to UTF-8 (the right thing for files and pipes); a real console
    keeps its encoding but stops raising.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - replaced stream in tests
            continue
        encoding = (getattr(stream, "encoding", None) or "").lower()
        try:
            if encoding.replace("-", "") in {"utf8", "utf8mb4"}:
                reconfigure(errors="replace")
            elif stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def build_config(args: argparse.Namespace) -> RAGConfig:
    """Merge env-var config with CLI overrides."""
    return RAGConfig.from_env(
        index_dir=getattr(args, "index_dir", None),
        chunk_size=getattr(args, "chunk_size", None),
        chunk_overlap=getattr(args, "chunk_overlap", None),
        embedding_backend=getattr(args, "embedding_backend", None),
        vector_backend=getattr(args, "vector_backend", None),
        retriever=getattr(args, "retriever", None),
        generator=getattr(args, "generator", None),
        top_k=getattr(args, "top_k", None),
    )


def load_pipeline(args: argparse.Namespace) -> RAGPipeline:
    """Load a persisted index, or exit with an actionable message."""
    config = build_config(args)
    index_dir = Path(config.index_dir)
    if not (index_dir / META_FILE).exists():
        raise SystemExit(
            f"No index found at '{index_dir}'.\n"
            "Build one first, e.g.:\n"
            "  python main.py index path/to/docs\n"
            "or try the zero-setup demo:\n"
            "  python main.py demo"
        )
    return RAGPipeline.load(index_dir, config=config)


def print_answer(result: RAGAnswer, show_sources: bool = True) -> None:
    print(f"\nAnswer: {result.answer}")
    if result.rewritten_question:
        print(f"(resolved question: {result.rewritten_question})")
    if show_sources and result.sources:
        print(f"\nSources ({result.num_sources}):")
        for position, source in enumerate(result.sources, start=1):
            snippet = source["text"].replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:160].rstrip() + "..."
            print(f"  [{position}] {source['source']} (score {source['score']:.4f})")
            print(f"      {snippet}")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def cmd_index(args: argparse.Namespace) -> int:
    config = build_config(args)
    index_dir = Path(config.index_dir)

    if args.append and (index_dir / META_FILE).exists():
        pipeline = RAGPipeline.load(index_dir, config=config)
    else:
        pipeline = RAGPipeline.build(config)

    added = pipeline.index_paths(args.paths, recursive=not args.no_recursive)
    if not added:
        print("Nothing was indexed - no supported documents found.")
        return 1

    pipeline.save(index_dir)
    stats = pipeline.stats()
    print(
        f"Indexed {added} new chunk(s). "
        f"Index now holds {stats['chunks']} chunk(s) from "
        f"{stats['documents']} document(s) at '{index_dir}'."
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args)
    result = pipeline.answer(args.question, top_k=args.top_k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print_answer(result, show_sources=not args.no_sources)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args)
    return run_chat(pipeline, top_k=args.top_k)


def run_chat(pipeline: RAGPipeline, top_k: Optional[int] = None) -> int:
    session = ConversationalRAG(pipeline)
    stats = pipeline.stats()

    print(BANNER)
    print("NLP RAG chat")
    print(BANNER)
    print(
        f"{stats['chunks']} chunk(s) from {stats['documents']} document(s) | "
        f"retriever={stats['retriever']} | generator={stats['generator']}"
    )
    print("Commands: 'reset' clears history, 'quit' exits.")
    print(BANNER)

    while True:
        try:
            message = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return 0

        if not message:
            continue
        if message.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            return 0
        if message.lower() == "reset":
            session.reset()
            print("Conversation history cleared.")
            continue

        try:
            result = session.chat(message, top_k=top_k)
        except Exception as exc:  # keep the REPL alive
            logger.exception("Query failed")
            print(f"Error: {exc}")
            continue
        print_answer(result)


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args)
    examples = load_eval_dataset(args.dataset)
    if not examples:
        print(f"No examples found in {args.dataset}")
        return 1

    report = evaluate(
        pipeline, examples, k=args.k, generate_answers=args.generate
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.format())
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    pipeline = load_pipeline(args)
    print(json.dumps(pipeline.stats(), indent=2, ensure_ascii=False))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    config = build_config(args)
    pipeline = RAGPipeline.build(config)
    pipeline.index_documents(sample_documents())

    print(BANNER)
    print("NLP RAG - demo over the built-in sample corpus")
    print(BANNER)
    stats = pipeline.stats()
    print(
        f"Indexed {stats['chunks']} chunk(s) | embedder={stats['embedder']['name']} "
        f"| vectors={stats['vector_backend']} | retriever={stats['retriever']} "
        f"| generator={stats['generator']}"
    )

    demo_questions = [
        "What is Natural Language Processing?",
        "How does retrieval-augmented generation reduce hallucination?",
        "Why combine BM25 with dense retrieval?",
        "Which metrics measure retrieval quality?",
    ]
    for question in demo_questions:
        print(f"\n{BANNER}\nQuestion: {question}")
        print_answer(pipeline.answer(question), show_sources=False)

    if args.no_chat:
        return 0

    print(f"\n{BANNER}\nStarting interactive mode...")
    return run_chat(pipeline, top_k=args.top_k)


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory holding the persisted index (default: storage)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve")
    parser.add_argument(
        "--retriever", choices=RETRIEVERS, default=None, help="Retrieval strategy"
    )
    parser.add_argument(
        "--generator", choices=GENERATORS, default=None, help="Answer generator"
    )
    parser.add_argument(
        "--embedding-backend",
        choices=EMBEDDING_BACKENDS,
        default=None,
        help="Embedding backend",
    )
    parser.add_argument(
        "--vector-backend",
        choices=VECTOR_BACKENDS,
        default=None,
        help="Vector index backend",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlp-rag",
        description="Retrieval-Augmented Generation over your own documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Index files or directories")
    index_parser.add_argument("paths", nargs="+", help="Files and/or directories")
    index_parser.add_argument("--chunk-size", type=int, default=None)
    index_parser.add_argument("--chunk-overlap", type=int, default=None)
    index_parser.add_argument(
        "--append", action="store_true", help="Add to the existing index"
    )
    index_parser.add_argument(
        "--no-recursive", action="store_true", help="Do not walk sub-directories"
    )
    _add_common(index_parser)
    index_parser.set_defaults(func=cmd_index)

    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question")
    query_parser.add_argument("--json", action="store_true", help="Emit JSON")
    query_parser.add_argument("--no-sources", action="store_true")
    _add_common(query_parser)
    query_parser.set_defaults(func=cmd_query)

    chat_parser = subparsers.add_parser("chat", help="Interactive multi-turn chat")
    _add_common(chat_parser)
    chat_parser.set_defaults(func=cmd_chat)

    eval_parser = subparsers.add_parser("eval", help="Score a labelled dataset")
    eval_parser.add_argument("dataset", help="JSON or JSONL evaluation file")
    eval_parser.add_argument("--k", type=int, default=5)
    eval_parser.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers and score them against the references",
    )
    eval_parser.add_argument("--json", action="store_true", help="Emit JSON")
    _add_common(eval_parser)
    eval_parser.set_defaults(func=cmd_eval)

    info_parser = subparsers.add_parser("info", help="Show index statistics")
    _add_common(info_parser)
    info_parser.set_defaults(func=cmd_info)

    demo_parser = subparsers.add_parser(
        "demo", help="Run the built-in corpus demo (no index required)"
    )
    demo_parser.add_argument(
        "--no-chat", action="store_true", help="Skip the interactive prompt"
    )
    _add_common(demo_parser)
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv or ["demo"])
    configure_logging(getattr(args, "verbose", False))

    if not getattr(args, "func", None):  # pragma: no cover - argparse guards this
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Command failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
