from __future__ import annotations

from pathlib import Path

import pytest

from docsqa.corpus import doc_id_from_bytes
from docsqa.ingest import IngestResult, ingest_pdf
from docsqa.parse import IngestError
from docsqa.store import DocumentInfo


def _write_minimal_pdf(path: Path, pages: int = 3) -> None:
    """Same programmatic minimal-PDF pattern as tests/test_parse.py (Task 3)."""
    objects = ["1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"]
    kids = " ".join(f"{i + 3} 0 R" for i in range(pages))
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {pages} >>\nendobj\n")
    for i in range(pages):
        objects.append(
            f"{i + 3} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            f"/Contents {pages + 3 + i} 0 R /Resources << /Font << /F1 {2 * pages + 3} 0 R >> >> >>\nendobj\n"
        )
    font_id = 2 * pages + 3
    objects.append(f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    for i in range(pages):
        stream = f"BT /F1 18 Tf 40 200 Td (Page {i + 1}) Tj ET"
        objects.append(f"{pages + 3 + i} 0 obj\n<< /Length {len(stream)} >>\nstream\n{stream}\nendstream\nendobj\n")
    body = "%PDF-1.4\n" + "".join(objects) + "%%EOF\n"
    path.write_bytes(body.encode("ascii"))


class FakeIngestEmbedder:
    model = "fake-embed"
    backend = "fake"
    dim = 3
    query_instruction = "retrieve passages"

    def __init__(self) -> None:
        self.document_texts: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, q: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeIngestStore:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, int]] = {}
        self.upserts: list[tuple[str, int, int]] = []  # (doc_id, n_chunks, n_vectors)

    def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(doc_id, source, chunk_count)
            for doc_id, (source, chunk_count) in self.documents.items()
        ]

    def upsert_document(self, doc_id, source, chunks, vectors) -> int:
        self.upserts.append((doc_id, len(chunks), len(vectors)))
        self.documents[doc_id] = (source, len(chunks))
        return len(chunks)


def test_ingest_twice_skips_second_time_and_adds_no_chunks(tmp_path: Path) -> None:
    pdf = tmp_path / "tiny.pdf"
    _write_minimal_pdf(pdf, pages=3)
    store = FakeIngestStore()
    embedder = FakeIngestEmbedder()

    first = ingest_pdf(pdf, store, embedder, size=16, overlap=4)
    second = ingest_pdf(pdf, store, embedder, size=16, overlap=4)

    expected_doc_id = doc_id_from_bytes(pdf.read_bytes())
    assert isinstance(first, IngestResult)
    assert first.doc_id == expected_doc_id
    assert first.skipped is False
    assert first.n_chunks >= 1

    assert second == IngestResult(doc_id=expected_doc_id, n_chunks=0, skipped=True)
    # Exactly one upsert and one embed batch: the second run did no work.
    assert store.upserts == [(expected_doc_id, first.n_chunks, first.n_chunks)]
    assert len(embedder.document_texts) == 1
    # The document row uses the file name as its source.
    assert store.documents[expected_doc_id][0] == "tiny.pdf"


def test_unreadable_pdf_indexes_zero_chunks_and_raises(tmp_path: Path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"these bytes are definitely not a pdf document")
    store = FakeIngestStore()
    embedder = FakeIngestEmbedder()

    with pytest.raises(IngestError):
        ingest_pdf(pdf, store, embedder, size=16, overlap=4)

    assert store.upserts == []
    assert store.documents == {}
    assert embedder.document_texts == []
