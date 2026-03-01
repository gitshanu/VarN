# app.py
import streamlit as st
from rag_core import RAGChatbot

# ─── Session state ───────────────────────────────────────
if "chatbot" not in st.session_state:
    with st.spinner("Loading knowledge base... (only once)"):
        st.session_state.chatbot = RAGChatbot(data_dir="./data")
        st.session_state.chatbot.build_index(force_rebuild=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

chatbot = st.session_state.chatbot

# ─── UI ──────────────────────────────────────────────────
st.title("📄 RAG Document Chatbot")
st.caption("Ask questions about your PDFs")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask anything about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chatbot.answer(prompt)
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Optional buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("Rebuild index (takes time)"):
        with st.spinner("Rebuilding vector store..."):
            chatbot.build_index(force_rebuild=True)
        st.success("Index rebuilt!")
        st.rerun()