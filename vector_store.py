from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from config import AppConfig
from ingest import DocumentChunk


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source_pdf: str
    page_number: int
    distance: float


class ChromaVectorStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.config.collection_name)
        except ValueError:
            pass
        self.collection = self._get_or_create_collection()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True).tolist()
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "chunk_id": chunk.chunk_id,
                "source_pdf": chunk.source_pdf,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata["chunk_id"]),
                    text=document,
                    source_pdf=str(metadata["source_pdf"]),
                    page_number=int(metadata["page_number"]),
                    distance=float(distance),
                )
            )
        return retrieved

    def has_embeddings(self) -> bool:
        return self.collection.count() > 0
