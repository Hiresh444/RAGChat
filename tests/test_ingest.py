from ingest import build_smart_splitter, chunk_pages, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("Hello \n\n world\t test") == "Hello world test"


def test_chunk_pages_preserves_page_metadata() -> None:
    chunks = chunk_pages(
        [(2, "A" * 1200)],
        source_pdf="sample.pdf",
        chunk_size=800,
        chunk_overlap=120,
    )
    assert len(chunks) >= 2
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.source_pdf == "sample.pdf" for chunk in chunks)


def test_smart_splitter_prefers_sentence_boundaries() -> None:
    text = (
        "INTRODUCTION\n\n"
        "This is the first sentence. This is the second sentence. "
        "This is the third sentence.\n\n"
        "1. Methods\n"
        "This is the methods section."
    )
    splitter = build_smart_splitter(chunk_size=70, chunk_overlap=10)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert any("This is the first sentence." in chunk for chunk in chunks)
