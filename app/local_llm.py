# app/query.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
# from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === CONFIGURACIÓN ===
FAISS_INDEX_PATH = "data/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistral"  # modelo usado por ollama (ej: mistral, llama3, gemma, etc.)

# === PROMPT (personalizable) ===
PROMPT_TEMPLATE = """
Eres un asistente experto en el Catecismo de la Iglesia Católica. Responde de manera clara, respetuosa y con base en el contenido proporcionado.

Contexto:
{context}

Pregunta:
{question}
"""
Ollama= OllamaLLM
# === FUNCIONES ===

def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    llm = Ollama(model=LLM_MODEL_NAME, temperature=0.3)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

# === MAIN ===

def main():
    print("📚 Cargando índice vectorial...")
    vector_store = load_vector_store()

    print("🤖 Iniciando asistente con Mistral local...")
    qa_chain = build_qa_chain(vector_store)

    while True:
        question = input("\n❓ Tu pregunta (o 'salir'): ").strip()
        if question.lower() in ['salir', 'exit', 'quit']:
            break

        result = qa_chain({"query": question})
        print("\n💬 Respuesta:")
        print(result["result"])

        print("\n📚 Fragmentos usados:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n🧩 Fragmento #{i}:\n{doc.page_content.strip()[:500]}")

if __name__ == "__main__":
    main()
