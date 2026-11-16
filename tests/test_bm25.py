from __future__ import annotations

import unicodedata

from docsqa.bm25 import BM25Index, tokenize_vi
from docsqa.chunking import Chunk


def _chunk(ordinal: int, text: str, doc_id: str = "doc") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal}",
        doc_id=doc_id,
        text=text,
        page_start=1,
        page_end=2,
        section=None,
        ordinal=ordinal,
    )


def test_tokenize_vi_emits_unigrams_and_stripped_bigrams() -> None:
    # For "Câu hỏi": accented unigrams (câu, hỏi), accent-stripped unigrams
    # (cau, hoi) and accent-stripped neighbour bigrams (cau_hoi). NFC-folded
    # NFD input must tokenize identically.
    tokens = tokenize_vi("Câu hỏi")
    tokens_nfd = tokenize_vi(unicodedata.normalize("NFD", "Câu hỏi"))

    assert tokens == tokens_nfd
    for expected in ("câu", "hỏi", "cau", "hoi", "cau_hoi"):
        assert expected in tokens


def test_bm25_finds_rare_identifier_that_vector_misses() -> None:
    # A rare identifier ("TX-9000") appears verbatim in one chunk out of many
    # filler chunks; BM25 must rank that chunk first for the exact query.
    chunks = [
        _chunk(0, "general introduction to maintenance schedules and daily checks"),
        _chunk(1, "error code TX-9000 means the thermal sensor has failed"),
        _chunk(2, "general notes about cleaning and replacing filters"),
        _chunk(3, "another unrelated paragraph about warranty terms"),
        _chunk(4, "more filler text about routine operation of the machine"),
    ]
    index = BM25Index.build(chunks)

    hits = index.search("TX-9000", k=3)

    assert hits[0][0] == "doc:1"
    assert hits[0][1] > 0


def test_bm25_is_diacritic_insensitive_for_query() -> None:
    # Chunk text is accented NFC; the query is typed without diacritics.
    chunks = [_chunk(0, "Câu hỏi về GDP của Việt Nam"), _chunk(1, "irrelevant filler")]
    index = BM25Index.build(chunks)

    hits = index.search("cau hoi ve gdp", k=1)

    assert hits[0][0] == "doc:0"


def test_bm25_idf_downweights_common_syllables() -> None:
    # "và" appears in every chunk (a stopword-like syllable), "trả" in two.
    # The chunk-specific "lãi suất" must dominate scoring: the query
    # "trả lãi suất" ranks the two money chunks above both filler chunks.
    chunks = [
        _chunk(0, "ngân hàng trả lãi suất cho khách hàng và cung cấp dịch vụ"),
        _chunk(1, "lãi suất tiền gửi do ngân hàng công bố và thay đổi"),
        _chunk(2, "trẻ em học chữ và chơi ở sân trường"),
        _chunk(3, "mọi người đi làm và ăn trưa ở công ty"),
    ]
    index = BM25Index.build(chunks)

    hits = index.search("trả lãi suất", k=4)
    ranked = [chunk_id for chunk_id, _ in hits]

    # Both chunks containing the rare content syllables outrank both chunks
    # that only share the ubiquitous "và".
    assert set(ranked[:2]) == {"doc:0", "doc:1"}


def test_bm25_refreshes_store_snapshot_when_document_set_changes() -> None:
    from docsqa.chunking import Chunk
    from docsqa.bm25 import BM25Index
    from docsqa.store import DocumentInfo

    class Store:
        def __init__(self):
            self.chunks = [Chunk("a:0", "a", "initial needle", 1, 1, None, 0)]

        def all_chunks(self, doc_ids=None):
            return list(self.chunks)

        def list_documents(self):
            grouped = {}
            for chunk in self.chunks:
                grouped[chunk.doc_id] = grouped.get(chunk.doc_id, 0) + 1
            return [DocumentInfo(doc, f"{doc}.pdf", count) for doc, count in grouped.items()]

    store = Store()
    index = BM25Index.from_store(store)
    assert index.search("initial", 5)

    store.chunks.append(Chunk("b:0", "b", "newword needle", 1, 1, None, 0))
    index.refresh_from_store(store)

    assert set(index.chunk_ids) == {"a:0", "b:0"}
    assert len(index.tokenized_corpus) == 2
    assert "newword" in index.tokenized_corpus[1]
    assert index._store_signature == (("a", 1), ("b", 1))


def test_bm25_search_respects_k_and_returns_scores() -> None:
    chunks = [_chunk(i, f"document body number {i} with words") for i in range(6)]
    index = BM25Index.build(chunks)

    hits = index.search("document body", k=3)

    assert len(hits) == 3
    assert all(isinstance(score, float) for _, score in hits)
    # Scores are sorted descending.
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
