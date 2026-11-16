from __future__ import annotations

from typing import Protocol

from .bm25 import BM25Index, reciprocal_rank_fusion
from .corpus import normalize_text, strip_accents
from .embedding import Embedder
from .store import Hit, VectorStore


def canonical_text(text: str) -> str:
    """Canonical embedding form shared by queries and chunk texts.

    NFC + whitespace-collapsed, then accent-stripped: an NFC query, its NFD
    variant, and the same question typed without diacritics all reduce to this
    one string, so they retrieve the same chunks from any embedder — the
    accent-stripping is applied at ingest time as well, so both sides of the
    similarity search see the same form.
    """
    return strip_accents(normalize_text(text))


class Retriever(Protocol):
    def retrieve(
        self, query: str, doc_ids: list[str] | None = None
    ) -> list[Hit]: ...


class VectorRetriever:
    """Dense retrieval over the pgvector store."""

    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self, query: str, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        query_vector = self.embedder.embed_query(canonical_text(query))
        return self.store.search_vector(query_vector, self.top_k, doc_ids)


class RerankRetriever:
    """V2a: dense top-``candidates`` -> cross-encoder rerank -> top-``top_k``.

    The base retriever is whatever produces an ordered candidate list (in
    practice ``VectorRetriever`` with ``top_k=candidates``); this stage
    re-scores those candidates with the reranker and keeps the best
    ``top_k``. ``doc_ids`` is forwarded to the base retriever unchanged so the
    page-filter behavior is preserved end to end.
    """

    def __init__(
        self,
        base: Retriever,
        reranker,
        candidates: int = 50,
        top_k: int = 5,
    ) -> None:
        if candidates <= 0 or top_k <= 0:
            raise ValueError("candidates and top_k must be positive")
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k

    def retrieve(
        self, query: str, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        hits = self.base.retrieve(query, doc_ids)
        if doc_ids is not None:
            allowed = set(doc_ids)
            hits = [hit for hit in hits if hit.chunk.doc_id in allowed]
        hits = hits[: self.candidates]
        if not hits:
            return []
        return self.reranker.rerank(query, hits, self.top_k)


class HybridRetriever:
    """V2b / v2b_norerank: RRF over (vector, BM25) then optional rerank.

    retrieve(query, doc_ids):
      1. vector.retrieve(query, doc_ids) -> ranking A (ordered chunk_ids).
      2. bm25.search_filtered(query, candidates, doc_ids) -> ranking B.
      3. rrf([A, B], k=rrf_k) -> fused (chunk_id, rrf_score) list.
      4. If reranker is None (v2b_norerank): return the top ``top_k`` fused
         chunk_ids as Hits carrying the RRF score.
      5. Else (v2b): shortlist the top ``candidates`` fused chunk_ids, resolve
         each to a Hit (reusing the vector Hits where available; BM25-only
         chunks resolved from a store.all_chunks prefetch filtered to
         doc_ids), rerank, return top_k.

    A chunk appearing in both rankings is fused once (RRF dedups within a
    ranking) and resolved once from the union map, so duplicate chunks are
    never double-listed.
    """

    def __init__(
        self,
        vector: Retriever,
        bm25: BM25Index,
        store,
        reranker=None,
        candidates: int = 50,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        if candidates <= 0 or top_k <= 0 or rrf_k <= 0:
            raise ValueError("candidates, top_k and rrf_k must be positive")
        self.vector = vector
        self.bm25 = bm25
        self.store = store
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        vector_hits = self.vector.retrieve(query, doc_ids)
        # Keep one global-IDF snapshot regardless of per-query doc filtering.
        self.bm25.refresh_from_store(self.store)
        ranking_a = [hit.chunk.chunk_id for hit in vector_hits]
        ranking_b = [
            chunk_id
            for chunk_id, _ in self.bm25.search_filtered(
                query, self.candidates, doc_ids
            )
        ]
        # Defensively enforce the document filter on vector results too; a
        # retriever implementation that ignores doc_ids must not leak chunks.
        if doc_ids is not None:
            allowed = set(doc_ids)
            vector_hits = [hit for hit in vector_hits if hit.chunk.doc_id in allowed]
            ranking_a = [hit.chunk.chunk_id for hit in vector_hits]
            allowed_chunk_ids = {
                chunk.chunk_id for chunk in self.bm25.chunks if chunk.doc_id in allowed
            }
            ranking_b = [
                chunk_id for chunk_id in ranking_b if chunk_id in allowed_chunk_ids
            ]

        fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=self.rrf_k)
        if not fused:
            return []

        if self.reranker is None:
            return self._rrf_top_k(fused, vector_hits, doc_ids)

        shortlist_hits = self._resolve_shortlist(fused[: self.candidates], vector_hits, doc_ids)
        if not shortlist_hits:
            return []
        return self.reranker.rerank(query, shortlist_hits, self.top_k)

    def _rrf_top_k(
        self,
        fused: list[tuple[str, float]],
        vector_hits: list[Hit],
        doc_ids: list[str] | None,
    ) -> list[Hit]:
        """Materialize the top ``top_k`` fused results as Hits with RRF scores."""
        by_id = {hit.chunk.chunk_id: hit for hit in vector_hits}
        top = fused[: self.top_k]
        missing_ids = [chunk_id for chunk_id, _ in top if chunk_id not in by_id]
        resolved: dict[str, object] = {}
        if missing_ids:
            prefetch = self.store.all_chunks(doc_ids)
            resolved = {chunk.chunk_id: chunk for chunk in prefetch}
        if doc_ids is not None:
            allowed = set(doc_ids)
            resolved = {
                chunk_id: chunk
                for chunk_id, chunk in resolved.items()
                if chunk.doc_id in allowed
            }
        out: list[Hit] = []
        for chunk_id, score in top:
            hit = by_id.get(chunk_id)
            if hit is not None:
                out.append(Hit(chunk=hit.chunk, score=score))
            else:
                chunk = resolved.get(chunk_id)
                if chunk is not None:
                    out.append(Hit(chunk=chunk, score=score))
        return out

    def _resolve_shortlist(
        self,
        shortlist: list[tuple[str, float]],
        vector_hits: list[Hit],
        doc_ids: list[str] | None,
    ) -> list[Hit]:
        """Build the candidate Hit list for the reranker, in fused order."""
        by_id = {hit.chunk.chunk_id: hit for hit in vector_hits}
        vector_ids = set(by_id)
        missing_ids = [
            chunk_id for chunk_id, _ in shortlist if chunk_id not in vector_ids
        ]
        resolved: dict[str, object] = {}
        if missing_ids:
            prefetch = self.store.all_chunks(doc_ids)
            resolved = {chunk.chunk_id: chunk for chunk in prefetch}
        if doc_ids is not None:
            allowed = set(doc_ids)
            resolved = {
                chunk_id: chunk
                for chunk_id, chunk in resolved.items()
                if chunk.doc_id in allowed
            }
        out: list[Hit] = []
        for chunk_id, _ in shortlist:
            hit = by_id.get(chunk_id)
            if hit is not None:
                out.append(hit)
            else:
                chunk = resolved.get(chunk_id)
                if chunk is not None:
                    # Reuse the chunk object; the reranker overwrites the
                    # score with the cross-encoder score.
                    out.append(Hit(chunk=chunk, score=0.0))
        return out
