from __future__ import annotations

import unicodedata

import pytest

from docsqa.bm25 import BM25Index
from docsqa.chunking import Chunk
from docsqa.retrieval import HybridRetriever, RerankRetriever
from docsqa.store import Hit


def _chunk(ordinal: int, text: str, doc_id: str = "docA") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal}",
        doc_id=doc_id,
        text=text,
        page_start=ordinal + 1,
        page_end=ordinal + 2,
        section=None,
        ordinal=ordinal,
    )


class FakeVectorRetriever:
    """Returns a fixed ranking; records every call for assertions."""

    def __init__(self, hits_by_doc: dict[str | None, list[Hit]]) -> None:
        self.hits_by_doc = hits_by_doc
        self.calls: list[tuple[str, list[str] | None]] = []

    def retrieve(self, query: str, doc_ids: list[str] | None = None) -> list[Hit]:
        self.calls.append((query, doc_ids))
        hits = self.hits_by_doc.get(tuple(doc_ids) if doc_ids else None, [])
        if doc_ids is None:
            return hits
        allowed = set(doc_ids)
        return [hit for hit in hits if hit.chunk.doc_id in allowed]


class FakeStore:
    """Resolves BM25-only chunk_ids via all_chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.all_chunks_calls: list[list[str] | None] = []

    def all_chunks(self, doc_ids: list[str] | None = None) -> list[Chunk]:
        self.all_chunks_calls.append(doc_ids)
        if doc_ids is None:
            return self.chunks
        wanted = set(doc_ids)
        return [chunk for chunk in self.chunks if chunk.doc_id in wanted]

    def list_documents(self):
        from docsqa.store import DocumentInfo

        by_doc: dict[str, tuple[str, int]] = {}
        for chunk in self.chunks:
            source, count = by_doc.get(chunk.doc_id, (f"{chunk.doc_id}.pdf", 0))
            by_doc[chunk.doc_id] = (source, count + 1)
        return [
            DocumentInfo(doc_id, source, count)
            for doc_id, (source, count) in by_doc.items()
        ]


class FakeReranker:
    """Score map keyed by chunk text; records pairs."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[Hit], int]] = []

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        self.calls.append((query, list(hits), top_k))
        scored = sorted(hits, key=lambda hit: -self.scores[hit.chunk.text])
        return [
            Hit(chunk=hit.chunk, score=self.scores[hit.chunk.text])
            for hit in scored[:top_k]
        ]


def _hit(chunk: Chunk, score: float) -> Hit:
    return Hit(chunk=chunk, score=score)


def test_rerank_retriever_reranks_base_candidates() -> None:
    # v2a: base retriever returns `candidates` hits; the reranker reorders
    # them and RerankRetriever returns its top_k.
    base_chunks = [
        _chunk(0, "alpha filler"),
        _chunk(1, "beta relevant"),
        _chunk(2, "gamma middling"),
    ]
    base = FakeVectorRetriever(
        {None: [_hit(base_chunks[0], 0.9), _hit(base_chunks[1], 0.8), _hit(base_chunks[2], 0.7)]}
    )
    reranker = FakeReranker({"alpha filler": 0.1, "beta relevant": 0.99, "gamma middling": 0.5})

    retriever = RerankRetriever(base, reranker, candidates=3, top_k=2)

    hits = retriever.retrieve("some question")

    assert [hit.chunk.chunk_id for hit in hits] == ["docA:1", "docA:2"]
    assert [hit.score for hit in hits] == [0.99, 0.5]
    query, passed_hits, top_k = reranker.calls[0]
    assert query == "some question"
    assert [hit.chunk.chunk_id for hit in passed_hits] == ["docA:0", "docA:1", "docA:2"]
    assert top_k == 2
    # doc_ids=None passes through to the base retriever.
    assert base.calls == [("some question", None)]


def test_hybrid_without_reranker_returns_rrf_top_k() -> None:
    # v2b_norerank: RRF fuses the vector ranking with BM25 and returns the
    # top_k fused chunk_ids with RRF scores, no reranker involved.
    vec_chunks = [_chunk(0, "v0"), _chunk(1, "v1")]
    vector = FakeVectorRetriever(
        {None: [_hit(vec_chunks[0], 0.9), _hit(vec_chunks[1], 0.88)]}
    )
    bm25_chunks = [
        _chunk(2, "error code TX-9000 in the manual"),
        _chunk(0, "v0"),
    ]
    store = FakeStore(vec_chunks + bm25_chunks)
    index = BM25Index.from_store(store)

    hybrid = HybridRetriever(
        vector, index, store, reranker=None, candidates=10, top_k=3, rrf_k=60
    )

    hits = hybrid.retrieve("TX-9000")

    chunk_ids = [hit.chunk.chunk_id for hit in hits]
    # docA:0 is rank 1 in vector and rank 2 in BM25 (both lists) -> highest
    # RRF score; docA:2 and docA:1 each appear in one list.
    assert len(hits) == 3
    assert chunk_ids[0] == "docA:0"
    assert set(chunk_ids[1:]) == {"docA:2", "docA:1"}
    # Scores are the RRF fused scores, sorted descending.
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    expected_top = 1 / 61 + 1 / 62  # docA:0: vector rank 1, bm25 rank 2
    assert hits[0].score == pytest.approx(expected_top, abs=1e-12)


def test_page_filter_and_duplicate_chunks_handled() -> None:
    # doc_ids filter is passed through and enforced across vector/BM25; a
    # chunk present in both rankings must appear exactly once.
    doc_a_chunks = [_chunk(0, "alpha shared text about lending money")]
    doc_b_chunks = [
        Chunk(chunk_id="docB:0", doc_id="docB", text="alpha shared text about lending money",
              page_start=1, page_end=2, section=None, ordinal=0),
        _chunk(5, "beta other text"),
    ]
    vector = FakeVectorRetriever(
        {("docA",): [_hit(doc_a_chunks[0], 0.9), _hit(doc_b_chunks[0], 0.85)]}
    )
    store = FakeStore(doc_a_chunks + doc_b_chunks)
    index = BM25Index.from_store(store)

    hybrid = HybridRetriever(
        vector, index, store, reranker=None, candidates=10, top_k=5, rrf_k=60
    )

    hits = hybrid.retrieve("lending money", doc_ids=["docA"])

    # The vector retriever saw the doc_ids filter; HybridRetriever also
    # enforces it in case an adapter returns a leaked out-of-filter hit.
    assert vector.calls == [("lending money", ["docA"])]
    chunk_ids = [hit.chunk.chunk_id for hit in hits]
    assert "docB:0" not in chunk_ids
    assert chunk_ids[0] == "docA:0"
    # The duplicate (docA:0 in both rankings) is listed exactly once.
    assert chunk_ids.count("docA:0") == 1
    assert all(hit.chunk.doc_id == "docA" for hit in hits)
    assert store.all_chunks_calls == [None, ["docA"]]


def test_hybrid_rejects_out_of_filter_chunks_from_store_resolution() -> None:
    allowed = _chunk(0, "needle passage", doc_id="docA")
    leaked = _chunk(1, "needle passage", doc_id="docB")
    vector = FakeVectorRetriever({("docA",): []})
    class LeakyStore(FakeStore):
        def all_chunks(self, doc_ids=None):
            self.all_chunks_calls.append(doc_ids)
            return self.chunks  # Deliberately ignores doc_ids.

        def list_documents(self):
            return super().list_documents()

    store = LeakyStore([allowed, leaked])
    index = BM25Index.from_store(store)
    hybrid = HybridRetriever(vector, index, store, candidates=5, top_k=5)

    hits = hybrid.retrieve("needle", doc_ids=["docA"])

    assert [hit.chunk.chunk_id for hit in hits] == ["docA:0"]
    assert store.all_chunks_calls == [None, ["docA"]]


def test_rerank_retriever_filters_leaked_base_hits() -> None:
    allowed = _chunk(0, "allowed", doc_id="docA")
    leaked = _chunk(1, "leaked", doc_id="docB")

    class LeakyVector(FakeVectorRetriever):
        def retrieve(self, query, doc_ids=None):
            self.calls.append((query, doc_ids))
            return [_hit(allowed, 0.9), _hit(leaked, 0.8)]

    base = LeakyVector({})
    reranker = FakeReranker({"allowed": 0.9, "leaked": 1.0})
    retriever = RerankRetriever(base, reranker, candidates=5, top_k=5)

    hits = retriever.retrieve("q", doc_ids=["docA"])

    assert [hit.chunk.chunk_id for hit in hits] == ["docA:0"]
    assert [hit.chunk.chunk_id for hit in reranker.calls[0][1]] == ["docA:0"]


def test_hybrid_with_reranker_fuses_then_reranks() -> None:
    # v2b: RRF shortlists `candidates` fused chunks; the reranker re-scores
    # them and the top_k survive. Chunks that only BM25 found are resolved
    # via store.all_chunks.
    vec_chunks = [_chunk(0, "vector only passage"), _chunk(1, "shared passage")]
    vector = FakeVectorRetriever({None: [_hit(vec_chunks[0], 0.9), _hit(vec_chunks[1], 0.85)]})
    bm25_only = _chunk(7, "error code TX-9000 thermal sensor failed")
    store = FakeStore(vec_chunks + [bm25_only, _chunk(2, "unrelated noise")])
    index = BM25Index.from_store(store)
    reranker = FakeReranker(
        {
            "vector only passage": 0.2,
            "shared passage": 0.6,
            "error code TX-9000 thermal sensor failed": 0.97,
        }
    )

    hybrid = HybridRetriever(
        vector, index, store, reranker=reranker, candidates=3, top_k=2, rrf_k=60
    )

    hits = hybrid.retrieve("TX-9000")

    assert [hit.chunk.chunk_id for hit in hits] == ["docA:7", "docA:1"]
    assert [hit.score for hit in hits] == [0.97, 0.6]
    # The reranker received exactly the fused shortlist (3 chunks, no
    # duplicates) and was asked for top_k.
    query, passed_hits, top_k = reranker.calls[0]
    assert query == "TX-9000"
    assert top_k == 2
    passed_ids = [hit.chunk.chunk_id for hit in passed_hits]
    assert len(passed_ids) == len(set(passed_ids)) == 3
    # The BM25-only chunk was resolved through the store.
    assert store.all_chunks_calls == [None, None]


def test_rerank_retriever_passes_doc_ids_through() -> None:
    base = FakeVectorRetriever({("docB",): [_hit(_chunk(9, "doc b passage", doc_id="docA"), 0.5)]})
    reranker = FakeReranker({"doc b passage": 0.7})

    retriever = RerankRetriever(base, reranker, candidates=5, top_k=5)

    retriever.retrieve("q", doc_ids=["docB"])

    assert base.calls == [("q", ["docB"])]


def test_hybrid_nfd_query_behaves_like_nfc() -> None:
    nfc = "câu hỏi về lãi suất"
    nfd = unicodedata.normalize("NFD", nfc)
    chunks = [_chunk(0, "Câu hỏi về lãi suất của ngân hàng"), _chunk(1, "không liên quan")]
    vector = FakeVectorRetriever({None: [_hit(chunks[0], 0.9)]})
    store = FakeStore(chunks)
    index = BM25Index.from_store(store)
    hybrid = HybridRetriever(vector, index, store, reranker=None, candidates=5, top_k=5)

    nfc_hits = hybrid.retrieve(nfc)
    nfd_hits = hybrid.retrieve(nfd)

    assert [h.chunk.chunk_id for h in nfc_hits] == [h.chunk.chunk_id for h in nfd_hits]
    assert [h.score for h in nfc_hits] == [h.score for h in nfd_hits]
