# app/query.py

import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
EMBEDDING_TYPE = "openai"  # or "huggingface"
FAISS_INDEX_PATH = "data/faiss_index"
LLM_MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.3

# === PROMPT TEMPLATE  ===
PROMPT_TEMPLATE = """
You are an expert in the Catechism of the Catholic Church. Answer clearly, true to the text and with theological depth. 

Context:
{context}

Question:
{question}
"""

# === Loading EMBEDDINGS and FAISS ===
def load_vector_store():
    if EMBEDDING_TYPE == "openai":
        embedding_model = OpenAIEmbeddings()
    elif EMBEDDING_TYPE == "huggingface":
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Type of embeddings not supported")
    return FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# === CREAR CADENA DE QA ===
def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )
    
    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=TEMPERATURE)
    
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
    print("🔍 Loading vector index...")
    vector_store = load_vector_store()

    print("🤖 Intializing chain of QA...")
    qa_chain = build_qa_chain(vector_store)

    while True:
        user_question = input("\n❓  Ask (or write ‘exit’ to finish): ")
        if user_question.lower() in ["salir", "exit", "quit"]:
            break

        result = qa_chain({"query": user_question})
        print("\n💬 Answer:")
        print(result["result"])
        print("\n📚 Used fragments:")
        for doc in result["source_documents"]:
            print("- " + doc.page_content[:200].replace("\n", " ") + "...")
            print("---")

if __name__ == "__main__":
    main()
