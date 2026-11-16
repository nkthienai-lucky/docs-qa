from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest

from docsqa.chunking import Chunk
from docsqa.store import EmbedderMeta, EmbedderMismatchError, VectorStore

TEST_DATABASE_URL = "postgresql://roadmap:roadmap@localhost:5432/docsqa_test"

pytestmark = pytest.mark.integration


def _chunk(doc_id: str, ordinal: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal}",
        doc_id=doc_id,
        text=text,
        page_start=ordinal + 1,
        page_end=ordinal + 1,
        section=None,
        ordinal=ordinal,
    )


def _drop_tables() -> None:
    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks CASCADE")
        conn.execute("DROP TABLE IF EXISTS documents CASCADE")
        conn.execute("DROP TABLE IF EXISTS index_meta CASCADE")


@pytest.fixture()
def store() -> Iterator[VectorStore]:
    _drop_tables()
    vector_store = VectorStore(
        TEST_DATABASE_URL,
        EmbedderMeta(model="test-embed", backend="test", dim=3),
    )
    vector_store.ensure_schema()
    yield vector_store
    vector_store.close()
    _drop_tables()


def test_upsert_is_idempotent_no_duplicate_chunks(store: VectorStore) -> None:
    chunks = [_chunk("doc1", i, f"nội dung {i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    first = store.upsert_document("doc1", "a.pdf", chunks, vectors)
    second = store.upsert_document("doc1", "a.pdf", chunks, vectors)

    assert first == second == 3
    stored = store.all_chunks(["doc1"])
    assert [chunk.chunk_id for chunk in stored] == [chunk.chunk_id for chunk in chunks]
    assert len({chunk.chunk_id for chunk in stored}) == 3


def test_reupload_same_bytes_new_filename_same_doc_id(store: VectorStore) -> None:
    chunks = [_chunk("deadbeef0000ffff", i, f"text {i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    store.upsert_document("deadbeef0000ffff", "old-name.pdf", chunks, vectors)
    store.upsert_document("deadbeef0000ffff", "renamed.pdf", chunks, vectors)

    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0].doc_id == "deadbeef0000ffff"
    assert documents[0].source == "renamed.pdf"
    assert documents[0].chunk_count == 3


def test_search_orders_by_cosine(store: VectorStore) -> None:
    chunks = [_chunk("docx", i, f"text {i}") for i in range(3)]
    vectors = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_document("docx", "s.pdf", chunks, vectors)

    hits = store.search_vector([1.0, 0.0, 0.0], k=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["docx:0", "docx:1", "docx:2"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 0.8, 0.0])
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )


def test_doc_filter_with_two_documents_still_returns_k_candidates(
    store: VectorStore,
) -> None:
    near_query = [_chunk("doc1", i, f"query side {i}") for i in range(3)]
    near_query_vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
    away = [_chunk("doc2", i, f"other side {i}") for i in range(3)]
    away_vectors = [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]
    store.upsert_document("doc1", "one.pdf", near_query, near_query_vectors)
    store.upsert_document("doc2", "two.pdf", away, away_vectors)

    hits = store.search_vector([1.0, 0.0, 0.0], k=5, doc_ids=["doc1", "doc2"])

    assert len(hits) == 5
    assert {hit.chunk.doc_id for hit in hits} <= {"doc1", "doc2"}
    assert [hit.score for hit in hits] == sorted(
        (hit.score for hit in hits), reverse=True
    )

    only_doc1 = store.search_vector([1.0, 0.0, 0.0], k=5, doc_ids=["doc1"])
    assert len(only_doc1) == 3
    assert {hit.chunk.doc_id for hit in only_doc1} == {"doc1"}


def test_embedder_mismatch_raises(store: VectorStore) -> None:
    store.ensure_schema()  # same metadata again is fine

    same_meta = VectorStore(TEST_DATABASE_URL, store.embedder_meta)
    same_meta.ensure_schema()
    same_meta.close()

    other_model = VectorStore(
        TEST_DATABASE_URL, EmbedderMeta(model="other-embed", backend="test", dim=3)
    )
    with pytest.raises(EmbedderMismatchError):
        other_model.ensure_schema()

    other_dim = VectorStore(
        TEST_DATABASE_URL, EmbedderMeta(model="test-embed", backend="test", dim=4)
    )
    with pytest.raises(EmbedderMismatchError):
        other_dim.ensure_schema()
