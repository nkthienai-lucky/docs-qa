from __future__ import annotations

from typing import Protocol

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
