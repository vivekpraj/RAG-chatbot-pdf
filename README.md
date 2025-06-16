# PDF Chatbot — Chat with Your Documents using RAG & LLMs

A simple and powerful chatbot that allows you to upload any PDF and chat with it in natural language using Retrieval-Augmented Generation (RAG) and Hugging Face language models.


## Features

- **Upload PDFs** and extract meaningful text
- **Ask questions** about the PDF content in a conversational style
- **RAG-powered**: Uses FAISS vector store + semantic search
- **Streamlit UI**: Interactive, chat-style interface
- **LLM-Ready**: Supports FLAN-T5, TinyLlama, Gemini Pro, and more
- **Offline-ready**: Use local Hugging Face models with cache
- **Deployable**: Works with Streamlit Cloud or Hugging Face Spaces

---

##  Folder Structure

rag_chatbot_pdf/
├── app.py # Main Streamlit app
├── vector_store.py # PDF loader, splitter, FAISS index
├── rag_chain.py # RAG logic (retrieval + prompt + answer)
├── llm_wrapper.py # Hugging Face / Gemini LLM loader
├── prompt_template.py # Custom prompt formatting
├── requirements.txt # Python dependencies
├── data/ # Uploaded PDFs

Model	Type:-	Notes
FLAN-T5:-	HuggingFace	Works offline, fast, accurate
TinyLlama:-	HuggingFace	Lightweight, local-friendly
Gemini Pro:-	Google API	Best quality, requires API key

Example Use Cases:-
Resume analysis chatbot,
Legal or contract Q&A,
Research paper summarizer,
Chat with financial reports or business docs.

Author
**Vivek Ashok Prajapati**
B.E. — University of Mumbai



