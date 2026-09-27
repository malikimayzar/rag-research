from src.ingestion.chunker import Chunk, chunk_semantic
from src.ingestion.document_loader import Document


def make_document(text: str) -> Document:
    return Document(
        doc_id="quality_doc",
        text=text,
        source="tests/fixture.txt",
        tier=1,
        metadata={},
    )


def test_semantic_chunks_keep_source_offsets():
    text = (
        "A short title. "
        "This first body sentence provides sufficient context for retrieval. "
        "This second body sentence continues the same grounded explanation."
    )

    chunks = chunk_semantic(make_document(text), min_chars=60, max_chars=200)

    assert chunks
    for chunk in chunks:
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert text[chunk.start_char:chunk.end_char].split() == chunk.text.split()


def test_semantic_chunks_overlap_at_sentence_boundaries():
    sentences = [
        f"Sentence {i} contains enough descriptive words to be useful retrieval context."
        for i in range(1, 6)
    ]
    chunks = chunk_semantic(make_document(" ".join(sentences)), min_chars=80, max_chars=155)

    assert len(chunks) >= 2
    assert "Sentence 2" in chunks[0].text
    assert "Sentence 2" in chunks[1].text


def test_short_body_buffer_is_not_dropped_before_next_chunk():
    short_lead = "The lead sentence contains important context."
    long_followup = " ".join([
        "The following sentence expands the explanation with enough detail for retrieval."
    ] * 3)

    chunks = chunk_semantic(
        make_document(f"{short_lead} {long_followup}"), min_chars=80, max_chars=150
    )

    assert any(short_lead in chunk.text for chunk in chunks)


def test_reference_and_title_only_text_are_not_indexable_chunks():
    text = (
        "A Paper Title. "
        "Author Name. "
        "This body sentence contains the substantive explanation needed by a reader. "
        "and includes enough factual detail to support accurate grounded retrieval. "
        "References. Doe et al. doi:10.1000/example."
    )

    chunks = chunk_semantic(make_document(text), min_chars=60, max_chars=220)

    assert chunks
    assert all("doi:10.1000/example" not in chunk.text.lower() for chunk in chunks)
    assert all(len(chunk.text.split()) >= 20 for chunk in chunks)
