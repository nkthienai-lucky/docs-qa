from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .corpus import normalize_text, strip_accents


def tokenize_vi(text: str) -> list[str]:
    """Tokenize a (Vietnamese or English) string for lexical retrieval.

    Pipeline: NFC normalize -> lowercase -> whitespace-split -> for each
    whitespace token emit (a) the accented unigram, (b) its accent-stripped
    unigram, plus (c) pairwise bigrams over the accent-stripped token stream.

    No external word segmenter: the default is deliberately whitespace-based
    so a question typed with or without diacritics, or composed in NFC vs
    NFD, indexes identically. The bigrams give BM25 some local context for
    multi-syllable Vietnamese phrases that whitespace alone would scatter.

    This mirrors ``retrieval.canonical_text`` (NFC + accent-strip) but keeps
    the accented unigram too, so a diacritic-heavy corpus still rewards an
    exact diacritic match while remaining diacritically forgiving for queries
    typed without them.
    """
    nfc = unicodedata.normalize("NFC", text).lower()
    raw_tokens = nfc.split()
    if not raw_tokens:
        return []

    stripped = [strip_accents(tok) for tok in raw_tokens]

    tokens: list[str] = []
    for accented, bare in zip(raw_tokens, stripped):
        tokens.append(accented)
        tokens.append(bare)
    for left, right in zip(stripped, stripped[1:]):
        tokens.append(f"{left}_{right}")
    return tokens


@dataclass
class BM25Index:
    """Lexical index over a fixed chunk set, backed by ``rank_bm25``.

    The index is built once from ``store.all_chunks`` and held by a retriever
    for the lifetime of that document set; rebuilding on every query would
    re-tokenize the whole corpus. ``BM25Okapi`` provides IDF weighting, so a
    syllable appearing in every chunk contributes less than a rare one.
    """

    chunks: list[Chunk]
    chunk_ids: list[str]
    tokenized_corpus: list[list[str]]
    bm25: BM25Okapi
    _store_signature: tuple[tuple[str, int], ...] | None = None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        tokenized_corpus = [tokenize_vi(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(
            chunks=list(chunks),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            tokenized_corpus=tokenized_corpus,
            bm25=bm25,
        )

    @classmethod
    def from_store(
        cls, store, doc_ids: list[str] | None = None
    ) -> "BM25Index":
        """Build an index snapshot from a store and remember its doc signature."""
        chunks = store.all_chunks(doc_ids)
        index = cls.build(chunks)
        docs = store.list_documents()
        selected = set(doc_ids) if doc_ids is not None else None
        index._store_signature = tuple(
            sorted(
                (document.doc_id, document.chunk_count)
                for document in docs
                if selected is None or document.doc_id in selected
            )
        )
        return index

    def refresh_from_store(
        self, store, doc_ids: list[str] | None = None
    ) -> "BM25Index":
        """Rebuild and replace this index if the selected document set changed."""
        docs = store.list_documents()
        selected = set(doc_ids) if doc_ids is not None else None
        signature = tuple(
            sorted(
                (document.doc_id, document.chunk_count)
                for document in docs
                if selected is None or document.doc_id in selected
            )
        )
        if signature != self._store_signature:
            replacement = type(self).from_store(store, doc_ids)
            self.chunks = replacement.chunks
            self.chunk_ids = replacement.chunk_ids
            self.tokenized_corpus = replacement.tokenized_corpus
            self.bm25 = replacement.bm25
            self._store_signature = replacement._store_signature
        return self

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top-k ``(chunk_id, bm25_score)`` for the query."""
        if k <= 0 or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(tokenize_vi(query))
        # Pair each chunk_id with its score, then sort descending by score.
        ranked = sorted(
            zip(self.chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return ranked[:k]

    def search_filtered(
        self, query: str, k: int, doc_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """``search`` restricted to chunks whose ``doc_id`` is in ``doc_ids``.

        When ``doc_ids`` is None this is identical to ``search``. Filtering
        happens after scoring so the same query tokenizes once; the BM25 IDF
        statistics stay global (computed over the whole corpus at build time),
        which is the standard behavior and keeps rare-term IDF meaningful.
        """
        if k <= 0 or not self.chunk_ids:
            return []
        if doc_ids is None:
            return self.search(query, k)
        wanted = set(doc_ids)
        scores = self.bm25.get_scores(tokenize_vi(query))
        ranked = sorted(
            (
                (chunk_id, score)
                for chunk, chunk_id, score in zip(
                    self.chunks, self.chunk_ids, scores
                )
                if chunk.doc_id in wanted
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:k]


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse multiple rankings into one with the RRF formula.

    For each ranking, rank positions start at 1; a document's fused score is
    ``sum(1 / (k + rank))`` over the rankings that contain it. ``k`` defaults
    to 60 (the standard constant) and is configurable. Documents appearing in
    more rankings, or higher up, score higher. Within a single ranking, a
    duplicate later occurrence is ignored (the first occurrence fixes the
    rank). The output is sorted by descending fused score, ties broken by
    first-seen order across all rankings so the result is stable across runs.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_this_ranking: set[str] = set()
        for position, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_in_this_ranking:
                continue
            seen_in_this_ranking.add(doc_id)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    fused = sorted(
        scores.items(),
        key=lambda pair: (-pair[1], first_seen[pair[0]]),
    )
    return fused
