from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_pdf: str
    page_number: int


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def build_smart_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # Prefer natural PDF structure first, then fall back to sentence and word boundaries.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True,
        separators=[
            r"\n(?=[A-Z][A-Z0-9\s]{2,}$)",  # all-caps section headings
            r"\n(?=\d+\.\s+[A-Z])",  # numbered headings
            r"\n(?=[\-\*\u2022]\s+)",  # bullet lists
            r"\n{2,}",  # paragraphs
            r"(?<=[.!?])\s+(?=[A-Z])",  # sentence boundaries
            r"(?<=[;:])\s+",
            r"\s+",
            r"",
        ],
    )


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            if raw_text.strip():
                pages.append((index, raw_text))
    return pages


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    *,
    source_pdf: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    splitter = build_smart_splitter(chunk_size, chunk_overlap)
    chunks: list[DocumentChunk] = []
    for page_number, page_text in pages:
        page_chunks = splitter.split_text(page_text)
        for position, chunk_text in enumerate(page_chunks, start=1):
            clean_text = normalize_text(chunk_text)
            if not clean_text:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{page_number}-{position}",
                    text=clean_text,
                    source_pdf=source_pdf,
                    page_number=page_number,
                )
            )
    return chunks


def save_uploaded_pdf(file_bytes: bytes, destination: Path) -> Path:
    destination.write_bytes(file_bytes)
    return destination
