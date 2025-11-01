# app.py
import os
import streamlit as st
from rag_bot import load_docs, create_vectorstore, create_qa_chain

# Prepare directories
os.makedirs("uploads", exist_ok=True)

st.set_page_config(page_title="📚 Local RAG Chatbot", layout="wide")
st.title("💬 Chat with Your Local Documents (Ollama Powered)")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- Upload Section ---
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Process Documents") and uploaded_files:
    with st.spinner("Reading and embedding documents..."):
        for file in uploaded_files:
            with open(os.path.join("uploads", file.name), "wb") as f:
                f.write(file.getbuffer())
        docs = load_docs(uploaded_files)
        vectorstore = create_vectorstore(docs)
        st.session_state.qa_chain = create_qa_chain(vectorstore)
    st.success("✅ Documents processed and embedded successfully!")

st.divider()

# --- Chat Section ---
if st.session_state.qa_chain:
    query = st.text_input("Ask something based on your documents:")
    if st.button("Ask") and query:
        with st.spinner("Thinking locally..."):
            result = st.session_state.qa_chain.invoke(query)
            st.write("**Answer:**", result)
else:
    st.info("👆 Upload and process your documents first.")
