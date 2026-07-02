# Local PDF Chatbot

Local-first PDF question answering app for indexing a single document, storing embeddings locally, and answering only from the uploaded content.

This project is built with `Streamlit`, `ChromaDB`, `SentenceTransformers`, `PyMuPDF`, and `Ollama`. A user uploads one PDF, the app extracts and chunks the text, stores embeddings in a persistent local vector database, and answers questions using only the uploaded document.

## Features

- Upload and index a PDF through a Streamlit web interface
- Extract text locally with `PyMuPDF`
- Chunk content automatically with LangChain text splitters using fixed app defaults
- Create embeddings locally with `all-MiniLM-L6-v2`
- Persist vectors locally with `ChromaDB`
- Generate answers locally with an `Ollama` model such as `llama3` or `mistral`
- Return the exact fallback message when the answer is not supported by the document
- Let the model synthesize summaries and answers from the retrieved document chunks

## Project Structure

```text
app.py
config.py
ingest.py
qa.py
vector_store.py
tests/
report/
presentation/
```

## Requirements

- Python `3.11` or `3.12`
- `uv` for environment and dependency management
- `Ollama` installed locally
- An Ollama model pulled locally, for example `llama3`

## Setup

Create a fresh environment with a supported Python version:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

Pull a local Ollama model if you do not already have one:

```bash
ollama pull llama3
```

## Run the App

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open the local Streamlit URL, upload a PDF, click `Index PDF`, and start asking questions.
Chunk size and overlap are handled automatically by the app.

## How It Works

1. The user uploads a PDF in the Streamlit sidebar.
2. `PyMuPDF` extracts text page by page.
3. LangChain smart chunking breaks each page on likely headings, paragraphs, bullets, and sentence boundaries before falling back to smaller splits.
4. `SentenceTransformers` creates embeddings for each chunk.
5. `ChromaDB` stores the embeddings and metadata locally.
6. The user question is embedded and matched against the indexed chunks.
7. Retrieved chunks are sent to `Ollama` with a synthesis prompt that can answer questions or summarize the document.
8. If support is not found, the app returns:

```text
The information is not available in the uploaded document.
```

## Notes

- The first version is optimized for one small PDF at a time.
- The vector collection is reset whenever a new PDF is indexed.
- The UI does not yet show citations, but page metadata is stored for future extension.

## Suggested Demo Flow

1. Upload a short PDF of 5-6 pages.
2. Ask one direct factual question from the document.
3. Ask a paraphrased version of the same question.
4. Ask an unsupported question to show the fallback behavior.
