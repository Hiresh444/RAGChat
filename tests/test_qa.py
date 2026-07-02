from config import AppConfig
from qa import FALLBACK_ANSWER, LocalPdfQaService
from vector_store import RetrievedChunk


class StubVectorStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[int] = []

    def has_embeddings(self) -> bool:
        return True

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(top_k)
        return self._chunks


class StubOllamaClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_prompt: str = ""

    def generate(self, model: str, prompt: str, options: dict[str, object]) -> dict[str, str]:
        self.last_prompt = prompt
        return {"response": self.response_text}


def test_answer_question_allows_llm_synthesis_from_retrieved_chunks() -> None:
    config = AppConfig()
    store = StubVectorStore(
        [
            RetrievedChunk(
                chunk_id="1-1",
                text="Chunk text",
                source_pdf="sample.pdf",
                page_number=1,
                distance=0.60,
            )
        ]
    )
    service = LocalPdfQaService(
        config,
        store,  # type: ignore[arg-type]
        ollama_client=StubOllamaClient("This is a synthesized answer."),
    )
    result = service.answer_question("What is this?")
    assert result.answer == "This is a synthesized answer."
    assert result.used_fallback is False
    assert store.calls == [4, 6]
    assert "thoughtful reader" in service.ollama_client.last_prompt
    assert "plot, characters, motives, themes, setting" in service.ollama_client.last_prompt
    assert "PDF" not in service.ollama_client.last_prompt


def test_summary_questions_use_summary_prompt_path() -> None:
    config = AppConfig()
    store = StubVectorStore(
        [
            RetrievedChunk(
                chunk_id="1-1",
                text="Section one.",
                source_pdf="sample.pdf",
                page_number=1,
                distance=0.10,
            ),
            RetrievedChunk(
                chunk_id="2-1",
                text="Section two.",
                source_pdf="sample.pdf",
                page_number=2,
                distance=0.20,
            ),
        ]
    )
    service = LocalPdfQaService(
        config,
        store,  # type: ignore[arg-type]
        ollama_client=StubOllamaClient("Summary answer."),
    )
    result = service.answer_question("Give me a summary of this book")
    assert result.answer == "Summary answer."
    assert result.used_fallback is False
    assert len(result.retrieved_chunks) == 2
    assert store.calls == [4, 8]
    assert "novel or story" in service.ollama_client.last_prompt
    assert "PDF" not in service.ollama_client.last_prompt
    assert "page " not in service.ollama_client.last_prompt.lower()
