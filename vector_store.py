import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Paths
PDF_FOLDER = "data"
FAISS_INDEX_PATH = "faiss_index"

# Step 1: Load PDF
def load_pdf_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
    return documents

# Step 2: Split into chunks
def split_documents(documents: List[Document]) -> List[Document]:
    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

# Step 3: Generate embeddings
def create_embeddings():
    print("🔍 Loading HuggingFace embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS
def store_faiss_index(chunks: List[Document], embeddings, index_path: str):
    print("💾 Storing embeddings in FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)
    print(f"✅ FAISS index saved at: {index_path}")

# ✅ Public method to call from app.py
def load_and_embed_pdfs():
    documents = load_pdf_documents(PDF_FOLDER)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    store_faiss_index(chunks, embeddings, FAISS_INDEX_PATH)

# Run this file directly
if __name__ == "__main__":
    load_and_embed_pdfs()
