import streamlit as st
from vector_store import load_and_embed_pdfs
from rag_chain import run_rag
from llm_wrapper import load_llm
import os

# Streamlit UI config
st.set_page_config(page_title="🧠 Chat with Your PDF", layout="centered")
st.title("📄 Ask Your PDF — Chatbot Style 💬")

# Ensure "data" folder exists
os.makedirs("data", exist_ok=True)

# Load model once
if "model" not in st.session_state:
    with st.spinner("🧠 Loading model..."):
        st.session_state.model = load_llm()

# Setup session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Upload PDF section (if not already uploaded)
if not st.session_state.pdf_uploaded:
    uploaded_file = st.file_uploader("📁 Upload your PDF", type="pdf")
    if uploaded_file:
        pdf_path = os.path.join("data", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Uploaded: {uploaded_file.name}")

        with st.spinner("🔄 Embedding PDF..."):
            load_and_embed_pdfs()

        st.success("✅ PDF processed and ready to chat!")
        st.session_state.pdf_uploaded = True
        st.rerun()

# Chat UI after PDF is uploaded
if st.session_state.pdf_uploaded:
    st.subheader("💬 Chat with your document")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    # Chat input (sticky at bottom)
    question = st.chat_input("Ask something about the PDF...")

    if question:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate assistant response
        with st.spinner("🤖 Thinking..."):
            answer = run_rag(question, st.session_state.model)

        # Append assistant message
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
