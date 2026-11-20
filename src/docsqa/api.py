from __future__ import annotations

import re
from contextlib import asynccontextmanager
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
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if getattr(app.state, "owns_store", False):
                store.close()

    app = FastAPI(title="docs-qa", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.store = store
    app.state.owns_store = False

    from fastapi.responses import JSONResponse

    from .api_rerank import RerankerError
    from .embedding import EmbeddingError
    from .generation import GeneratorError

    async def provider_error(request, error):
        return JSONResponse(
            status_code=502, content={"detail": "provider request failed"}
        )

    for error_type in (GeneratorError, EmbeddingError, RerankerError):
        app.add_exception_handler(error_type, provider_error)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        index_validated = bool(getattr(app.state, "runtime_index_validated", False))
        initialized = (
            bool(getattr(app.state, "runtime_initialized", False)) and index_validated
        )
        database_ready = False
        if initialized:
            try:
                database_ready = store.check_ready()
            except Exception:  # noqa: BLE001 — readiness must not leak DB diagnostics
                database_ready = False
        state = {
            "config": "validated" if initialized else "unverified",
            "index": "validated" if index_validated else "unverified",
            "database": "ready" if database_ready else "unavailable",
            "providers": "configured" if initialized else "unverified",
        }
        return JSONResponse(
            status_code=200 if initialized and database_ready else 503, content=state
        )

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


def build_runtime_app(settings) -> FastAPI:
    from . import runtime
    from .bm25 import BM25Index
    from .pipeline import Pipeline
    from .retrieval import HybridRetriever, RerankRetriever, VectorRetriever

    store = None
    app = None
    try:
        components = runtime.build_runtime_components(settings)
        embedder = components.embedder
        embedder.embed_documents(["runtime dimension probe"])
        identity = runtime.runtime_index_identity(settings, embedder)
        store = VectorStore(
            settings.database_url,
            EmbedderMeta(embedder.model, embedder.backend, embedder.dim, identity),
            connect_timeout=3,
            statement_timeout_ms=2000,
        )
        store.ensure_schema()
        vector = VectorRetriever(store, embedder, top_k=50)
        bm25 = BM25Index.from_store(store)
        retrievers = {
            "v1": VectorRetriever(store, embedder, top_k=5),
            "v2a": RerankRetriever(vector, components.reranker, candidates=50, top_k=5),
            "v2b": HybridRetriever(
                vector,
                bm25,
                store,
                reranker=components.reranker,
                candidates=50,
                top_k=5,
            ),
            "v2b_norerank": HybridRetriever(
                vector, bm25, store, reranker=None, candidates=50, top_k=5
            ),
        }
        pipeline = Pipeline(
            retrievers,
            components.generator,
            final_k=5,
            generation_k=settings.generation_k,
        )
        app = create_app(pipeline, store, Path(settings.runtime_upload_dir))
        app.state.owns_store = True
        app.state.runtime_initialized = True
        # Only set after probe + identity validation + ensure_schema succeeded.
        app.state.runtime_index_validated = True
    except Exception:  # noqa: BLE001 — startup failures may contain credentials
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001, S110 — cleanup errors must not leak credentials
                pass
    if app is None:
        raise runtime.RuntimeConfigurationError(
            "runtime initialization failed; verify configuration and fresh index"
        )
    return app


def serve() -> None:
    from .serve import main

    main()
