# 📚 RAG Document Q&A Chatbot

**Ask questions about your PDFs using Groq LLM + FAISS + Streamlit**

A simple yet powerful Retrieval-Augmented Generation (RAG) application that lets users chat with their PDF documents.

Live Demo: [![Streamlit App](https://img.shields.io/badge/🚀-Live%20Demo-blue?style=for-the-badge&logo=streamlit)](https://your-app-name.streamlit.app)

![Demo screenshot](https://via.placeholder.com/800x450.png?text=Chatbot+Screenshot+Here)  
*(Replace with actual screenshot – see instructions below)*

## ✨ Features

- 🔍 PDF document ingestion & chunking
- 🧠 Semantic search using sentence-transformers embeddings
- ⚡ Fast local vector store with **FAISS**
- 💬 Conversational Q&A interface powered by **Groq** (Llama 3.1 8B)
- 🌐 Beautiful web UI with Streamlit
- 🔒 Secure API key handling via Streamlit secrets
- Easy local & cloud deployment

## 🛠️ Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Frontend           | Streamlit                           |
| LLM                | Groq (llama-3.1-8b-instant)         |
| Embeddings         | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store       | FAISS                               |
| Document Loading   | PyPDF + LangChain loaders           |
| Orchestration      | LangChain                           |

## 🚀 Quick Start (Local)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
