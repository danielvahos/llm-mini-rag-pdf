# app/ingest.py

import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from tqdm import tqdm
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()

# === CONFIGURATION ===
PDF_PATH = "data/catecismo.pdf"
# EMBEDDING_TYPE = "openai"  # Change to 'hugging face' if I don't want Open AI
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
FAISS_INDEX_PATH = "data/faiss_index"

# === 1. Extract text from PDF ===
def load_pdf_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === 2. Divide in CHUNKS ===
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    return splitter.split_text(text)

# === 3. EMBEDDINGS ===
# === 3. EMBEDDINGS LOCALES ===
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# === 4. Build VECTORSTORE FAISS ===
def build_vector_store(chunks):
    embeddings = get_embedding_model()
    print("Generating vector of embeddings...")
    return FAISS.from_texts(chunks, embeddings)

# === 5. Save FAISS INDEX ===
def save_index(vector_store):
    print(f"Saving index in: {FAISS_INDEX_PATH}")
    vector_store.save_local(FAISS_INDEX_PATH)

# === MAIN ===
if __name__ == "__main__":
    print("📖 Loading PDF...")
    raw_text = load_pdf_text(PDF_PATH)

    print("✂️ Splitting text...")
    chunks = split_text(raw_text)

    print(f"🧠 {len(chunks)} fragments generated. Building index FAISS...")
    vector_store = build_vector_store(tqdm(chunks))

    save_index(vector_store)
    print("✅ Ready! the vector index has been created")
