"""NLP RAG - Retrieval-Augmented Generation toolkit.

A dependency-light RAG stack that works out of the box (pure NumPy retrieval and
extractive generation) and transparently upgrades when optional backends are
installed (sentence-transformers, FAISS, transformers, Anthropic).
"""

from nlp_rag.config import RAGConfig
from nlp_rag.documents import Chunk, RawDocument, chunk_documents, split_text
from nlp_rag.embeddings import Embedder, HashingEmbedder, get_embedder
from nlp_rag.generation import (
    ExtractiveGenerator,
    Generator,
    GenerationError,
    get_generator,
)
from nlp_rag.pipeline import ConversationalRAG, RAGAnswer, RAGPipeline
from nlp_rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievedChunk,
)
from nlp_rag.vectorstore import NumpyVectorStore, VectorStore, get_vector_store

__version__ = "0.2.0"

__all__ = [
    "BM25Retriever",
    "Chunk",
    "ConversationalRAG",
    "DenseRetriever",
    "Embedder",
    "ExtractiveGenerator",
    "GenerationError",
    "Generator",
    "HashingEmbedder",
    "HybridRetriever",
    "NumpyVectorStore",
    "RAGAnswer",
    "RAGConfig",
    "RAGPipeline",
    "RawDocument",
    "RetrievedChunk",
    "VectorStore",
    "__version__",
    "chunk_documents",
    "get_embedder",
    "get_generator",
    "get_vector_store",
    "split_text",
]
