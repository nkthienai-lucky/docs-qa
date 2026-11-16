from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_blocks
from .corpus import doc_id_from_bytes
from .embedding import Embedder
from .parse import IngestError, parse_pdf
from .retrieval import canonical_text
from .store import VectorStore


def _count_whitespace_tokens(text: str) -> int:
    """Whitespace token counting, as validated for this corpus (docs/corpus.md)."""
    return len(text.split())


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    n_chunks: int
    skipped: bool


def ingest_pdf(
    path: Path,
    store: VectorStore,
    embedder: Embedder,
    size: int,
    overlap: int = 64,
    timeout_s: int = 300,
) -> IngestResult:
    """Parse, chunk, embed and index one PDF.

    Order matters: the fingerprint check runs before the expensive parse, so a
    re-upload of the same bytes is a no-op, and a parse failure leaves the
    index untouched (nothing was written yet). ``timeout_s`` is the parser
    budget: large native-text PDFs need well over the default (the approved
    265-page baseline parses in ~10+ minutes CPU→GPU mixed).
    """
    # Bytes are read before parse_pdf so the fingerprint check can skip an
    # already-indexed document without paying for a parse; an unreadable file
    # therefore surfaces here and is normalized to IngestError like every
    # other ingest failure.
    try:
        data = path.read_bytes()
    except OSError as error:
        raise IngestError(f"cannot read {path.name}: {error}") from error
    doc_id = doc_id_from_bytes(data)
    if any(document.doc_id == doc_id for document in store.list_documents()):
        return IngestResult(doc_id=doc_id, n_chunks=0, skipped=True)

    blocks = parse_pdf(path, timeout_s=timeout_s)
    chunks = chunk_blocks(
        blocks,
        doc_id=doc_id,
        size=size,
        overlap=overlap,
        count_tokens=_count_whitespace_tokens,
    )
    texts = [canonical_text(chunk.text) for chunk in chunks]
    vectors = embedder.embed_documents(texts) if texts else []
    store.upsert_document(doc_id, path.name, chunks, vectors)
    return IngestResult(doc_id=doc_id, n_chunks=len(chunks), skipped=False)
