"""
NLP RAG - Natural Language Processing with Retrieval-Augmented Generation
A complete RAG system for question answering with document retrieval
"""

import os
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentStore:
    """Simple document storage and retrieval"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]):
        """Add documents to the store"""
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            return []
        
        # Simple keyword-based search (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        # Score each document
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            # Calculate overlap score
            overlap = len(query_words.intersection(doc_words))
            score = overlap / max(len(query_words), 1)
            
            if score > 0:
                scored_docs.append({
                    'id': idx,
                    'content': doc,
                    'score': score
                })
        
        # Sort by score and return top-k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]


class RAGSystem:
    """Retrieval-Augmented Generation system"""
    
    def __init__(self):
        self.document_store = DocumentStore()
        logger.info("RAG system initialized")
    
    def load_documents(self, documents: List[str]):
        """Load documents into the system"""
        self.document_store.add_documents(documents)
    
    def load_from_file(self, filepath: str):
        """Load documents from a text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (simple paragraph-based splitting)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            self.load_documents(chunks)
            logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        results = self.document_store.search(query, top_k)
        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer based on query and context
        
        Args:
            query: User question
            context: Retrieved context documents
            
        Returns:
            Generated answer
        """
        if not context:
            return "I don't have enough information to answer that question."
        
        # Simple answer generation (in production, use LLM)
        answer_parts = [
            f"Based on the available information:",
            f"\n\nRelevant context:",
        ]
        
        for i, ctx in enumerate(context[:2], 1):
            answer_parts.append(f"\n{i}. {ctx[:200]}...")
        
        return "\n".join(answer_parts)
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Process a query end-to-end
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k)
        
        # Extract context
        context = [doc['content'] for doc in retrieved_docs]
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs)
        }


class InteractiveRAG:
    """Interactive RAG chatbot"""
    
    def __init__(self):
        self.rag = RAGSystem()
        self.conversation_history = []
    
    def load_knowledge_base(self, documents: List[str] = None, filepath: str = None):
        """Load knowledge base from documents or file"""
        if documents:
            self.rag.load_documents(documents)
        elif filepath:
            self.rag.load_from_file(filepath)
        else:
            # Load sample documents
            sample_docs = [
                "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.",
                "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to produce more accurate and contextual responses.",
                "Machine learning models can be trained on large datasets to recognize patterns and make predictions.",
                "Deep learning uses neural networks with multiple layers to process complex data representations.",
                "Transformers are a type of neural network architecture that has revolutionized NLP tasks.",
                "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model.",
                "GPT (Generative Pre-trained Transformer) models excel at text generation tasks.",
                "Vector embeddings represent text as numerical vectors in high-dimensional space.",
                "Semantic search uses embeddings to find documents based on meaning rather than keywords.",
                "Question answering systems use NLP to understand questions and retrieve relevant answers."
            ]
            self.rag.load_documents(sample_docs)
    
    def chat(self, message: str) -> str:
        """Process a chat message"""
        result = self.rag.query(message)
        self.conversation_history.append({
            'question': message,
            'answer': result['answer']
        })
        return result['answer']
    
    def run(self):
        """Run interactive chat loop"""
        print("=" * 70)
        print("NLP RAG Chatbot")
        print("=" * 70)
        print("Ask questions about the knowledge base (type 'quit' to exit)")
        print("=" * 70)
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                answer = self.chat(question)
                print(f"\nBot: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("NLP RAG - Natural Language Processing with RAG")
    print("=" * 70)
    
    # Initialize RAG system
    bot = InteractiveRAG()
    
    # Load sample knowledge base
    print("\nLoading knowledge base...")
    bot.load_knowledge_base()
    
    # Run demo queries
    demo_queries = [
        "What is Natural Language Processing?",
        "Tell me about RAG systems",
        "How do transformers work?"
    ]
    
    print("\nRunning demo queries...")
    print("=" * 70)
    
    for query in demo_queries:
        print(f"\nQuery: {query}")
        result = bot.rag.query(query)
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['num_sources']}")
    
    print("\n" + "=" * 70)
    print("Demo complete! Starting interactive mode...")
    print("=" * 70)
    
    # Start interactive mode
    bot.run()


if __name__ == "__main__":
    main()
