from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model_name: str = "llama3"
    collection_name: str = "pdf_chatbot"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4
    distance_threshold: float = 0.65
    app_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def data_dir(self) -> Path:
        return self.app_root / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_directories(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()
