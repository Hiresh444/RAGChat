# Technical Report: Local PDF Chatbot Using LLM and Vector Database

## 1. Introduction

This project implements a local PDF chatbot that allows a user to upload a document and ask questions about its contents without relying on external APIs. The solution is designed for privacy, offline capability after setup, and explainable retrieval-based answering. The first version targets small PDFs of approximately five to six pages and focuses on strict grounded responses rather than open-ended generation.

The core idea is retrieval-augmented generation with local components only. The application extracts text from an uploaded PDF, divides it into manageable chunks, creates local embeddings, stores them in a vector database, retrieves the most relevant chunks for a question, and asks a local large language model to answer strictly from that retrieved context.

## 2. Objectives

The project goals are:

- Build a working PDF upload and chat experience using Streamlit
- Keep all inference and storage local to the machine
- Support semantic retrieval through vector search
- Use an Ollama-served local LLM for final answer generation
- Return a clear fallback when information is missing from the uploaded document

These objectives align with common academic and prototype requirements for document question-answering systems while avoiding reliance on cloud-hosted AI services.

## 3. System Architecture

The system consists of five main layers:

1. User Interface: A Streamlit web app handles file upload, indexing controls, model settings, and the chat interface.
2. Document Ingestion: PyMuPDF extracts raw page text from the PDF and a normalization step removes redundant whitespace.
3. Chunking and Embedding: LangChain text splitting creates overlapping chunks, and SentenceTransformers generates embeddings using `all-MiniLM-L6-v2`.
4. Vector Storage and Retrieval: ChromaDB stores the chunk embeddings and metadata locally and returns relevant chunks for each query.
5. Response Generation: Ollama serves a local LLM, such as Llama 3 or Mistral, which receives only the retrieved document context and the user question.

This design is modular and easy to extend. Each layer is separated into its own Python module so the application can evolve into a multi-document or citation-rich system later.

## 4. Workflow

The end-to-end workflow is:

1. The user uploads a PDF through the sidebar.
2. The application saves the file into a local project directory.
3. PyMuPDF reads text from each page.
4. The extracted text is normalized and chunked into segments of roughly 800 characters with overlap.
5. Sentence-transformer embeddings are generated locally for every chunk.
6. ChromaDB stores the embeddings with metadata such as source PDF name, page number, and chunk id.
7. When a user asks a question, the query is embedded with the same local embedding model.
8. ChromaDB retrieves the top matching chunks.
9. A confidence gate filters out weak matches using a distance threshold.
10. The retrieved context is sent to an Ollama model with a strict prompt.
11. The final answer is displayed in the chat interface.

If no retrieved chunks pass the threshold, the chatbot returns the required fallback message: `The information is not available in the uploaded document.`

## 5. Technology Choices

### Python

Python was selected because it has a strong ecosystem for NLP, vector databases, PDF processing, and quick UI development.

### Streamlit

Streamlit offers fast development for data and AI interfaces. It is well-suited to file upload, sidebar controls, and conversational layouts.

### PyMuPDF

PyMuPDF is lightweight, fast, and reliable for extracting text from PDFs, especially when documents are primarily text based.

### LangChain Text Splitters

LangChain text splitters provide robust chunking logic with overlap support. This reduces implementation complexity while keeping the stack local.

### SentenceTransformers

The `all-MiniLM-L6-v2` model provides a practical balance of quality and efficiency for semantic search on consumer hardware.

### ChromaDB

ChromaDB was selected over FAISS for the first version because it provides convenient local persistence and metadata management with minimal setup.

### Ollama

Ollama makes it straightforward to run local open-weight language models through a simple local interface. It allows the chatbot to remain fully offline after model download.

## 6. Implementation Details

The codebase is divided into focused modules:

- `app.py` handles the Streamlit user interface and session state
- `config.py` stores defaults for chunking, retrieval, paths, and model names
- `ingest.py` extracts text and builds chunk objects with metadata
- `vector_store.py` manages embedding generation and ChromaDB interactions
- `qa.py` performs retrieval, confidence filtering, prompt construction, and Ollama calls

The vector store is reset when a new PDF is indexed so the first version always works against one active document. This keeps the mental model simple and reduces accidental cross-document answers.

## 7. Testing Strategy

The system should be validated with both functional and behavioral checks:

- Upload a valid short PDF and confirm indexing completes
- Ask a direct question with an obvious answer in the text
- Ask a paraphrased question to verify semantic retrieval
- Ask an unsupported question and verify the exact fallback string
- Restart the app and confirm the ChromaDB directory persists
- Confirm a clear error message appears if Ollama is not running or the selected model is unavailable

Unit tests in the repository cover text normalization, chunk metadata preservation, and fallback behavior when retrieval relevance is too weak.

## 8. Limitations

This first version has several intentional limitations:

- It targets one uploaded PDF at a time
- It assumes text-based PDFs rather than scanned images with OCR needs
- It does not yet expose citations in the UI
- The retrieval confidence gate is heuristic rather than learned
- Performance and quality depend on the locally available embedding model and Ollama model

These limitations are acceptable for an initial academic or prototype deliverable and leave a clear roadmap for future improvements.

## 9. Future Enhancements

Potential extensions include:

- Multi-document support
- Citation display with page references in the UI
- OCR for scanned PDFs
- Adjustable prompt templates
- More advanced reranking for retrieval quality
- Exportable chat session history
- Benchmarking across multiple local embedding and LLM combinations

## 10. Conclusion

The local PDF chatbot demonstrates a practical retrieval-augmented generation pipeline using only local components. It satisfies the privacy and offline constraints while remaining understandable, modular, and suitable for demonstration. By combining Streamlit, PyMuPDF, SentenceTransformers, ChromaDB, and Ollama, the project delivers an end-to-end document QA workflow that is both educational and useful as a foundation for more advanced systems.
