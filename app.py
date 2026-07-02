from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import AppConfig
from ingest import chunk_pages, extract_pdf_pages, save_uploaded_pdf
from qa import FALLBACK_ANSWER, LocalPdfQaService
from vector_store import ChromaVectorStore


st.set_page_config(page_title="Local PDF Chatbot", page_icon="📄", layout="wide")


def get_config() -> AppConfig:
    config = AppConfig()
    config.ensure_directories()
    return config


@st.cache_resource(show_spinner=False)
def get_vector_store(chroma_dir: str, embedding_model_name: str, collection_name: str) -> ChromaVectorStore:
    config = AppConfig(
        app_root=Path(chroma_dir).resolve().parent.parent,
        embedding_model_name=embedding_model_name,
        collection_name=collection_name,
    )
    return ChromaVectorStore(config)


def reset_chat_state() -> None:
    st.session_state["messages"] = []


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_pdf", "")
    st.session_state.setdefault("index_status", "No PDF indexed yet.")


def render_sidebar(config: AppConfig, vector_store: ChromaVectorStore) -> None:
    st.sidebar.title("Local PDF Chatbot")
    st.sidebar.caption("Upload one PDF, index it locally, and ask grounded questions.")
    st.sidebar.info(
        "Chunking now happens automatically using the default LangChain settings in the app config."
    )

    config.ollama_model_name = st.sidebar.text_input(
        "Ollama model",
        value=config.ollama_model_name,
        help="Example: llama3, mistral, llama3.1:8b",
    )
    config.embedding_model_name = st.sidebar.text_input(
        "Embedding model",
        value=config.embedding_model_name,
        disabled=True,
        help="The first version keeps the embedding model fixed for stability.",
    )
    config.retrieval_top_k = st.sidebar.slider("Top K", 2, 8, config.retrieval_top_k, 1)
    config.distance_threshold = st.sidebar.slider(
        "Distance threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(config.distance_threshold),
        step=0.05,
        help="Lower values are stricter.",
    )

    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        destination = config.uploads_dir / uploaded_file.name
        if st.sidebar.button("Index PDF", use_container_width=True):
            with st.spinner("Extracting, chunking, and indexing PDF..."):
                save_uploaded_pdf(uploaded_file.getvalue(), destination)
                pages = extract_pdf_pages(destination)
                chunks = chunk_pages(
                    pages,
                    source_pdf=uploaded_file.name,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                vector_store.reset_collection()
                indexed_count = vector_store.index_chunks(chunks)
            st.session_state["active_pdf"] = uploaded_file.name
            st.session_state["index_status"] = (
                f"Indexed {indexed_count} chunks from {uploaded_file.name}."
            )
            reset_chat_state()

    st.sidebar.success(st.session_state["index_status"])
    if st.session_state["active_pdf"]:
        st.sidebar.write(f"Active document: `{st.session_state['active_pdf']}`")


def main() -> None:
    initialize_session()
    config = get_config()
    vector_store = get_vector_store(
        str(config.chroma_dir),
        config.embedding_model_name,
        config.collection_name,
    )
    render_sidebar(config, vector_store)

    st.title("Chat with Your PDF")
    st.write(
        "Ask questions about the uploaded PDF. The assistant answers from the document and can also summarize it using the model's language understanding."
    )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded document")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    qa_service = LocalPdfQaService(config, vector_store)
    with st.chat_message("assistant"):
        with st.spinner("Thinking locally..."):
            try:
                result = qa_service.answer_question(question)
                answer = result.answer or FALLBACK_ANSWER
            except RuntimeError as exc:
                answer = str(exc)
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
