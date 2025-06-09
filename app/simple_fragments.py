# app/query.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === CONFIG ===
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === LOAD VECTORSTORE ===
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# === INTERFACE OF QUESTIONS ===
def main():
    print("📚 Loading index of knowledge...")
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    while True:
        query = input("\n❓ Your question (or write 'exit' to finish): ").strip()
        if query.lower() in ["salir", "exit", "quit"]:
            break

        docs = retriever.get_relevant_documents(query)
        print("\n🔎 Most relevant fragments:")
        for i, doc in enumerate(docs, 1):
            print(f"\n🧩 Fragment #{i}:\n{doc.page_content.strip()[:500]}")  # Shows only first 500 characters

if __name__ == "__main__":
    main()
