import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import ChatOllama


# --- Load and split documents ---
def load_docs(uploaded_files):
    docs = []
    for file in uploaded_files:
        path = os.path.join("uploads", file.name)
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs


def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")
    vectorstore.persist()
    return vectorstore


def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the provided context to answer accurately.
    If the answer is not in the context, say "I don't know."

    Context: {context}
    Question: {input}
    """)

    # LCEL chain (lightweight, no tools or bind_tools)
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    print("✅ RAG Bot setup complete. You can now query your vectorstore using Ollama.")
