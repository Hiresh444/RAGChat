# Final Presentation Outline: Local PDF Chatbot

## Slide 1: Title

- Local PDF Chatbot Using LLM and Vector Database
- Name, course or team details, and submission date

## Slide 2: Problem Statement

- Users need a way to ask questions about PDF content
- Many solutions depend on cloud APIs or external services
- This project keeps document processing and answering fully local

## Slide 3: Objective

- Upload a PDF
- Extract and chunk text
- Generate local embeddings
- Store vectors locally
- Answer questions using a local LLM only from the uploaded document

## Slide 4: Technology Stack

- Python
- Streamlit
- PyMuPDF
- LangChain text splitters
- SentenceTransformers
- ChromaDB
- Ollama with Llama 3 or Mistral

## Slide 5: System Architecture

- UI layer
- PDF extraction layer
- Chunking and embedding layer
- Vector database layer
- Retrieval and answer generation layer

Speaker note: include a simple block diagram showing the flow from PDF upload to final response.

## Slide 6: Workflow

- Upload PDF
- Extract text page by page
- Chunk text with overlap
- Create embeddings
- Store in ChromaDB
- Embed user query
- Retrieve relevant chunks
- Ask Ollama to answer from retrieved context

## Slide 7: Live Demo Flow

- Upload a 5-6 page PDF
- Index the document
- Ask a direct factual question
- Ask a paraphrased question
- Ask a question not covered in the document

Speaker note: highlight the fallback response for unsupported questions.

## Slide 8: Key Features

- Local-only processing
- No external APIs
- Persistent vector storage
- Strict grounded answering
- Simple and clean chat interface

## Slide 9: Testing and Results

- Successful indexing of short PDFs
- Correct retrieval on direct questions
- Semantic retrieval on paraphrased questions
- Fallback behavior for unsupported questions
- Graceful handling when Ollama is unavailable

## Slide 10: Limitations

- One PDF at a time
- No OCR for scanned PDFs
- No citations shown in the current UI
- Quality depends on the local model used

## Slide 11: Future Improvements

- Multiple document support
- Citations in responses
- OCR integration
- Better relevance scoring
- Export chat history

## Slide 12: Conclusion

- The project delivers a practical local RAG workflow
- It is privacy-friendly and suitable for offline use
- It creates a strong foundation for more advanced document QA systems
