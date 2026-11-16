from __future__ import annotations

import pytest

from docsqa.uploads import (
    DEFAULT_MAX_BYTES,
    MAX_CHUNKS_PER_DOCUMENT,
    UploadGuardStore,
    UploadRejected,
    sanitize_filename,
    validate_upload,
)


def _pdf_bytes(size: int) -> bytes:
    """A payload of exactly ``size`` bytes that starts with the PDF magic."""
    return b"%PDF-1.4\n" + b"%" * (size - 9)


def test_small_valid_pdf_passes() -> None:
    assert validate_upload(_pdf_bytes(64)) is None


def test_non_pdf_magic_rejected_with_400() -> None:
    with pytest.raises(UploadRejected, match="PDF") as excinfo:
        validate_upload(b"PK\x03\x04 definitely not a pdf")

    assert excinfo.value.status_code == 400


def test_payload_exactly_at_cap_passes() -> None:
    assert validate_upload(_pdf_bytes(1_000), max_bytes=1_000) is None


def test_payload_one_byte_over_cap_rejected_with_413() -> None:
    with pytest.raises(UploadRejected, match="exceeds") as excinfo:
        validate_upload(_pdf_bytes(1_001), max_bytes=1_000)

    assert excinfo.value.status_code == 413


def test_default_cap_is_30mb() -> None:
    assert DEFAULT_MAX_BYTES == 30_000_000
    # Boundary behaviour is covered above at a small cap; the production cap
    # value itself is pinned here without allocating two 30MB payloads.
    assert validate_upload.__defaults__ == (DEFAULT_MAX_BYTES,)


def test_sanitize_filename_strips_every_directory_component() -> None:
    assert sanitize_filename("plain.pdf", "fallback.pdf") == "plain.pdf"
    assert sanitize_filename("../../x.pdf", "fallback.pdf") == "x.pdf"
    # Windows separators are not separators on POSIX, so they are normalized
    # explicitly.
    assert sanitize_filename("..\\..\\x.pdf", "fallback.pdf") == "x.pdf"
    assert sanitize_filename("/etc/passwd", "fallback.pdf") == "passwd"


def test_sanitize_filename_falls_back_on_empty_or_dangerous_names() -> None:
    assert sanitize_filename(None, "fallback.pdf") == "fallback.pdf"
    assert sanitize_filename("", "fallback.pdf") == "fallback.pdf"
    assert sanitize_filename("../../", "fallback.pdf") == "fallback.pdf"
    assert sanitize_filename("..", "fallback.pdf") == "fallback.pdf"
    # Null bytes are stripped, not forwarded into display metadata.
    assert sanitize_filename("a\x00b.pdf", "fallback.pdf") == "ab.pdf"


class _InnerStore:
    def __init__(self) -> None:
        self.documents: list[tuple[str, str, int]] = []
        self.listed = 0

    def list_documents(self) -> list:
        self.listed += 1
        return []

    def upsert_document(self, doc_id, source, chunks, vectors) -> int:
        self.documents.append((doc_id, source, len(chunks)))
        return len(chunks)


def test_guard_store_records_client_source_not_ingest_path_name() -> None:
    inner = _InnerStore()
    guard = UploadGuardStore(inner, source="client name.pdf")

    guard.upsert_document(
        "doc", "3f0d2a9b8c1d4e7f.pdf", ["chunk"] * 3, [[0.0]] * 3
    )

    # The ingest-provided source is the stored path name; the wrapper replaces
    # it with the sanitized client filename for display.
    assert inner.documents == [("doc", "client name.pdf", 3)]


def test_guard_store_blocks_upsert_over_chunk_cap_and_never_reaches_inner() -> None:
    inner = _InnerStore()
    guard = UploadGuardStore(inner, source="x.pdf", max_chunks=2)

    guard.upsert_document("doc", "doc.pdf", ["c"] * 2, [[0.0]] * 2)
    with pytest.raises(UploadRejected, match="chunk") as excinfo:
        guard.upsert_document("doc2", "doc2.pdf", ["c"] * 3, [[0.0]] * 3)

    assert excinfo.value.status_code == 422
    assert inner.documents == [("doc", "x.pdf", 2)]


def test_guard_store_default_cap_is_5000_and_delegates_listing() -> None:
    inner = _InnerStore()
    guard = UploadGuardStore(inner, source="x.pdf")

    assert guard.max_chunks == MAX_CHUNKS_PER_DOCUMENT == 5000
    assert guard.list_documents() == []
    assert inner.listed == 1
