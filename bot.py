# Import libraries
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import BaseRetriever, Document
from typing import List

# Step 1: Load and preprocess the PDF
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load your PDF
pdf_text = load_pdf("C:/Users/sarth/OneDrive/Desktop/Projects/NLP RAG/NLP Journal.pdf")  # Replace with your PDF file path

# Step 2: Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(pdf_text)

# Step 3: Create embeddings and FAISS index
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
embeddings = embedder.encode(texts)

# Build FAISS index with float32 conversion
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Step 4: Set up the retriever
def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [texts[i] for i in indices[0]]
    return relevant_chunks

# Step 5: Set up GPT-2 for generation
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_answer(query, context):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Create a generator pipeline
generator_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    device="cpu"  # Switch to "cuda" if GPU is available
)

# Create a LangChain LLM wrapper
llm = HuggingFacePipeline(pipeline=generator_pipeline)

# Step 6: Define a custom retriever
class CustomRetriever(BaseRetriever):
    def __init__(self, retrieve_function):
        super().__init__()
        self._retrieve_function = retrieve_function

    def get_relevant_documents(self, query: str) -> List[Document]:
        relevant_chunks = self._retrieve_function(query)
        return [Document(page_content=chunk) for chunk in relevant_chunks]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async version not implemented")

# Create the retriever
retriever = CustomRetriever(retrieve)

# Step 7: Build the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Step 8: Query the RAG system
query = "What is NLP?"  # Replace with your question
result = rag_chain.run(query)
print("Answer:", result)