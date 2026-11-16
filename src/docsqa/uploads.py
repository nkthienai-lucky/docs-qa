from __future__ import annotations

PDF_MAGIC = b"%PDF-"
DEFAULT_MAX_BYTES = 30_000_000
MAX_CHUNKS_PER_DOCUMENT = 5000


class UploadRejected(ValueError):
    """An upload failed validation. ``status_code`` is the HTTP response."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_upload(data: bytes, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    """Reject non-PDF payloads (magic bytes) and payloads over the size cap."""
    if not data.startswith(PDF_MAGIC):
        raise UploadRejected("uploaded file is not a PDF (missing %PDF- header)")
    if len(data) > max_bytes:
        raise UploadRejected(
            f"upload of {len(data)} bytes exceeds the {max_bytes}-byte limit",
            status_code=413,
        )


def sanitize_filename(name: str | None, fallback: str) -> str:
    """Reduce a client-supplied filename to a bare, display-safe basename.

    The result is never used for storage — uploads are stored as
    ``{doc_id}.pdf`` — only as the ``source`` shown in document listings.
    Both POSIX and Windows separators are treated as directory separators,
    and null bytes are dropped, so the value cannot smuggle a path.
    """
    if name:
        basename = name.replace("\\", "/").rsplit("/", 1)[-1]
        basename = basename.replace("\x00", "")
        if basename not in ("", ".", ".."):
            return basename
    return fallback


class UploadGuardStore:
    """One-upload wrapper around the store, owned by the API layer.

    ``ingest_pdf`` records the *stored* file name as the document source
    (``{doc_id}.pdf``); the wrapper substitutes the sanitized client
    filename so ``GET /documents`` shows what the user uploaded. It also
    enforces the per-document chunk cap at the upsert boundary — before the
    inner store is touched — so an oversized parse leaves nothing indexed.
    """

    def __init__(self, store, source: str, max_chunks: int = MAX_CHUNKS_PER_DOCUMENT) -> None:
        self._store = store
        self.source = source
        self.max_chunks = max_chunks

    def list_documents(self):
        return self._store.list_documents()

    def upsert_document(self, doc_id: str, source: str, chunks, vectors) -> int:
        if len(chunks) > self.max_chunks:
            raise UploadRejected(
                f"document produced {len(chunks)} chunks, over the cap of "
                f"{self.max_chunks}",
                status_code=422,
            )
        return self._store.upsert_document(doc_id, self.source, chunks, vectors)
