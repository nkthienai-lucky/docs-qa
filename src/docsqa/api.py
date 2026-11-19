from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .corpus import doc_id_from_bytes
from .ingest import IngestError, ingest_pdf
from .pipeline import AnswerResponse, Pipeline
from .store import EmbedderMeta, VectorStore
from .uploads import (
    DEFAULT_MAX_BYTES,
    UploadGuardStore,
    UploadRejected,
    sanitize_filename,
    validate_upload,
)

HOST = "127.0.0.1"
MAX_UPLOAD_BYTES = DEFAULT_MAX_BYTES
# Frozen V1 ingestion settings (docs/corpus.md): 512-token chunks with 64
# overlap. The 900s parse budget matches the approved 265-page baseline —
# parse_pdf's 300s default would reject it during cold model startup. The
# 300-page cap is enforced by parse_pdf's own default (ingest_pdf does not
# forward max_pages).
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
PARSE_TIMEOUT_S = 900


class AskRequest(BaseModel):
    question: str
    config: str = "v2a"
    doc_ids: list[str] | None = None


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    chunk_count: int
    skipped: bool


def create_app(pipeline: Pipeline, store: VectorStore, data_dir: Path) -> FastAPI:
    app = FastAPI(title="docs-qa")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/documents")
    def documents() -> list[dict[str, Any]]:
        return [
            {"doc_id": d.doc_id, "source": d.source, "chunk_count": d.chunk_count}
            for d in store.list_documents()
        ]

    @app.post("/documents", status_code=201, response_model=DocumentOut)
    def upload(file: Annotated[UploadFile, File()]) -> DocumentOut:
        # Read at most cap+1 bytes: one over the cap is enough to reject,
        # without pulling an arbitrarily large body into memory.
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        try:
            validate_upload(data, max_bytes=MAX_UPLOAD_BYTES)
        except UploadRejected as error:
            raise HTTPException(error.status_code, str(error)) from error

        # The client filename never touches the filesystem: bytes are stored
        # under the content hash, and the sanitized name is display metadata.
        doc_id = doc_id_from_bytes(data)
        display_name = sanitize_filename(file.filename, f"{doc_id}.pdf")
        display_name = re.sub(r"[\x00-\x1f\x7f<>:\"|?*]", "_", display_name)[:255]
        if not display_name.lower().endswith(".pdf"):
            display_name += ".pdf"
        data_dir.mkdir(parents=True, exist_ok=True)
        stored = data_dir / f"{doc_id}.pdf"
        stored.write_bytes(data)
        # The ingest embedder must be the one behind the index; every retriever
        # of this deployment shares it, so take it from the first.
        embedder = next(iter(pipeline.retrievers.values())).embedder
        guard = UploadGuardStore(store, source=display_name)
        ingest_succeeded = False
        try:
            try:
                result = ingest_pdf(
                    stored,
                    guard,
                    embedder,
                    size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    timeout_s=PARSE_TIMEOUT_S,
                )
            except UploadRejected as error:
                raise HTTPException(error.status_code, str(error)) from error
            except IngestError as error:
                raise HTTPException(422, str(error)) from error
            ingest_succeeded = True
        finally:
            # Keep the file only after a completed ingest/duplicate check;
            # unexpected embedder or database failures must not orphan uploads.
            if not ingest_succeeded:
                stored.unlink(missing_ok=True)
        # On a skipped re-upload the store has the real count and source; keep
        # those canonical metadata values even if the client chose a new name.
        info = {d.doc_id: d for d in store.list_documents()}.get(doc_id)
        return DocumentOut(
            doc_id=result.doc_id,
            source=info.source if result.skipped and info else display_name,
            chunk_count=info.chunk_count if info else result.n_chunks,
            skipped=result.skipped,
        )

    @app.post("/ask")
    def ask(request: AskRequest) -> AnswerResponse:
        if request.config not in pipeline.retrievers:
            raise HTTPException(
                400, f"unknown retrieval configuration: {request.config}"
            )
        return pipeline.ask(
            request.question, config=request.config, doc_ids=request.doc_ids
        )

    return app


def serve() -> None:
    """Build the deployment stack from settings and serve on loopback only."""
    import uvicorn

    from .bm25 import BM25Index
    from .embedding import OllamaEmbedder
    from .generation import OllamaGenerator
    from .rerank import Reranker
    from .retrieval import HybridRetriever, RerankRetriever, VectorRetriever
    from .settings import get_settings

    settings = get_settings()
    embedder = OllamaEmbedder(
        host=settings.ollama_host,
        model=settings.embed_model,
        query_instruction="Given a web search query, retrieve relevant passages",
    )
    # dim is 0 until the first embed call — probe before building EmbedderMeta.
    embedder.embed_documents(["probe text"])
    store = VectorStore(
        settings.database_url,
        EmbedderMeta(model=embedder.model, backend=embedder.backend, dim=embedder.dim),
    )
    store.ensure_schema()
    vector = VectorRetriever(store, embedder, top_k=50)
    bm25 = BM25Index.from_store(store)
    reranker = Reranker(
        settings.reranker_model,
        device=settings.reranker_device,
        batch_size=settings.reranker_batch_size,
    )
    retrievers = {
        "v1": VectorRetriever(store, embedder, top_k=5),
        "v2a": RerankRetriever(vector, reranker, candidates=50, top_k=5),
        "v2b": HybridRetriever(
            vector, bm25, store, reranker=reranker, candidates=50, top_k=5
        ),
        "v2b_norerank": HybridRetriever(
            vector, bm25, store, reranker=None, candidates=50, top_k=5
        ),
    }
    pipeline = Pipeline(
        retrievers,
        OllamaGenerator(
            host=settings.ollama_host,
            model=settings.generator_model,
            num_ctx=settings.num_ctx,
            num_predict=settings.generator_num_predict,
        ),
        final_k=5,
        generation_k=settings.generation_k,
    )
    uvicorn.run(create_app(pipeline, store, Path("data/uploads")), host=HOST, port=8000)
