from __future__ import annotations

from dataclasses import dataclass

from ollama import Client, ResponseError

from config import AppConfig
from vector_store import ChromaVectorStore, RetrievedChunk


FALLBACK_ANSWER = "The information is not available in the uploaded document."


@dataclass(slots=True)
class AnswerResult:
    answer: str
    used_fallback: bool
    retrieved_chunks: list[RetrievedChunk]


def _is_summary_request(question: str) -> bool:
    lowered = question.lower()
    return any(
        keyword in lowered
        for keyword in (
            "summary",
            "summarize",
            "summarise",
            "overview",
            "brief",
            "high level",
            "high-level",
            "tl;dr",
        )
    )


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    if _is_summary_request(question):
        instruction = (
            "You are summarizing a novel or story for a reader who has not read it yet. "
            "Write a fluid, human summary that focuses on the premise, major characters, "
            "the central conflict, important turning points, tone, and themes. "
            "Synthesize the story in your own words and avoid sounding like you are reporting on pages or extracted text. "
            "Do not mention pages, chunks, retrieval, or how the text was extracted."
        )
    else:
        instruction = (
            "Answer the question as a thoughtful reader who understands the book or document. "
            "Use the excerpts to reason about plot, characters, motives, themes, setting, events, and cause-and-effect relationships. "
            "Feel free to synthesize across multiple excerpts instead of repeating exact wording. "
            "Do not mention pages, chunks, retrieval, or how the text was extracted."
        )
    return (
        "You are a helpful local book and document assistant.\n"
        f"{instruction}\n"
        "If the context does not contain enough information to respond safely, reply exactly with:\n"
        f"{FALLBACK_ANSWER}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


class LocalPdfQaService:
    def __init__(
        self,
        config: AppConfig,
        vector_store: ChromaVectorStore,
        ollama_client: Client | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.ollama_client = ollama_client or Client()

    def answer_question(self, question: str) -> AnswerResult:
        query = question.strip()
        if not query or not self.vector_store.has_embeddings():
            return AnswerResult(
                answer=FALLBACK_ANSWER,
                used_fallback=True,
                retrieved_chunks=[],
            )

        chunks = self.vector_store.similarity_search(query, self.config.retrieval_top_k)
        if not chunks:
            return AnswerResult(
                answer=FALLBACK_ANSWER,
                used_fallback=True,
                retrieved_chunks=[],
            )

        relevant_chunks = chunks
        if _is_summary_request(query):
            summary_top_k = max(self.config.retrieval_top_k, 8)
            relevant_chunks = self.vector_store.similarity_search(query, summary_top_k)
        else:
            general_top_k = max(self.config.retrieval_top_k, 6)
            if general_top_k > self.config.retrieval_top_k:
                relevant_chunks = self.vector_store.similarity_search(query, general_top_k)

        prompt = _build_prompt(query, relevant_chunks)
        try:
            response = self.ollama_client.generate(
                model=self.config.ollama_model_name,
                prompt=prompt,
                options={"temperature": 0},
            )
        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama request failed for model '{self.config.ollama_model_name}': {exc.error}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure the Ollama app or server is running locally."
            ) from exc

        answer = response.get("response", "").strip()
        if not answer:
            answer = FALLBACK_ANSWER
        used_fallback = answer == FALLBACK_ANSWER
        return AnswerResult(
            answer=answer,
            used_fallback=used_fallback,
            retrieved_chunks=relevant_chunks,
        )
