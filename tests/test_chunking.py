from docsqa.chunking import Block, chunk_blocks


def count_tokens(text: str) -> int:
    return len(text.split())


def test_chunks_never_exceed_size_and_overlap_is_64() -> None:
    words = [f"w{i}" for i in range(700)]
    chunks = chunk_blocks(
        [Block(" ".join(words), page=1, heading="Unit 1")],
        doc_id="doc",
        size=256,
        overlap=64,
        count_tokens=count_tokens,
    )

    assert all(count_tokens(chunk.text) <= 256 for chunk in chunks)
    assert chunks[0].text.split()[-64:] == chunks[1].text.split()[:64]


def test_chunk_spanning_pages_sets_page_start_lt_page_end() -> None:
    chunks = chunk_blocks(
        [Block("alpha beta", 1, "A"), Block("gamma delta", 2, None)],
        doc_id="doc",
        size=10,
        overlap=0,
        count_tokens=count_tokens,
    )

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_heading_boundary_not_crossed_when_section_fits() -> None:
    chunks = chunk_blocks(
        [Block("one two", 1, "A"), Block("three four", 2, "B")],
        doc_id="doc",
        size=10,
        overlap=0,
        count_tokens=count_tokens,
    )

    assert [chunk.text for chunk in chunks] == ["one two", "three four"]
    assert [chunk.section for chunk in chunks] == ["A", "B"]


def test_chunk_ids_stable_across_runs() -> None:
    blocks = [Block("one two three", 1, "A")]

    first = chunk_blocks(blocks, "doc", 10, 0, count_tokens)
    second = chunk_blocks(blocks, "doc", 10, 0, count_tokens)

    assert [chunk.chunk_id for chunk in first] == ["doc:0"]
    assert first == second
