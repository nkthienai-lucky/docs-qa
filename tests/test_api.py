from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import docsqa.api as api_module
import docsqa.ingest as ingest_module
from docsqa.api import HOST, create_app
from docsqa.chunking import Block
from docsqa.corpus import doc_id_from_bytes
from docsqa.ingest import IngestResult
from docsqa.parse import IngestError
from docsqa.pipeline import AnswerResponse, Citation, RetrievalInfo
from docsqa.store import DocumentInfo
from docsqa.uploads import MAX_CHUNKS_PER_DOCUMENT

PDF_BYTES = b"%PDF-1.4\nminimal magic-validated payload for the api tests"


class FakeStore:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, int]] = {}

    def list_documents(self) -> list[DocumentInfo]:
        return [
            DocumentInfo(doc_id, source, chunk_count)
            for doc_id, (source, chunk_count) in self.documents.items()
        ]

    def upsert_document(self, doc_id, source, chunks, vectors) -> int:
        self.documents[doc_id] = (source, len(chunks))
        return len(chunks)


class FakeEmbedder:
    model = "fake-embed"
    backend = "fake"
    dim = 4
    query_instruction = "retrieve passages"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, q):
        return [1.0, 0.0, 0.0, 0.0]


class FakeRetriever:
    """Exposes ``embedder`` like VectorRetriever, which create_app relies on."""

    def __init__(self, embedder: FakeEmbedder) -> None:
        self.embedder = embedder

    def retrieve(self, query, doc_ids=None):
        return []


class FakePipeline:
    def __init__(self) -> None:
        self.embedder = FakeEmbedder()
        self.retrievers = {
            "v1": FakeRetriever(self.embedder),
            "v2a": FakeRetriever(self.embedder),
        }
        self.calls: list[tuple[str, str, list[str] | None]] = []

    def ask(self, question, config="v2a", doc_ids=None):
        if config not in self.retrievers:
            raise KeyError(f"unknown retrieval configuration: {config}")
        self.calls.append((question, config, doc_ids))
        return AnswerResponse(
            answer="Borrow means to take [1].",
            abstained=False,
            citations=[
                Citation(
                    source="3a357be041df101c",
                    page_start=3,
                    page_end=4,
                    section="Unit 3",
                    chunk_id="3a357be041df101c:0",
                    excerpt="borrow: to take something temporarily",
                )
            ],
            retrieval=RetrievalInfo(config, ["3a357be041df101c:0"], [0.9]),
        )


def fake_parse_pdf(path, max_pages=300, timeout_s=300):
    """Stand-in for the docling parse: small real blocks, no subprocess."""
    return [
        Block("borrow and lend vocabulary unit content block one", 1),
        Block("grammar content block two with more words here", 2),
        Block("final content block three about phrasal verbs", 3),
    ]


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(ingest_module, "parse_pdf", fake_parse_pdf)
    store = FakeStore()
    pipeline = FakePipeline()
    data_dir = tmp_path / "uploads"
    client = TestClient(create_app(pipeline, store, data_dir))
    return SimpleNamespace(
        client=client,
        store=store,
        pipeline=pipeline,
        data_dir=data_dir,
        tmp_path=tmp_path,
    )


def test_health_returns_ok(env) -> None:
    response = env.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_serializes_answer_response_and_forwards_body(env) -> None:
    response = env.client.post(
        "/ask",
        json={
            "question": "what does borrow mean?",
            "config": "v1",
            "doc_ids": ["3a357be041df101c"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Borrow means to take [1]."
    assert body["abstained"] is False
    citation = body["citations"][0]
    assert citation["source"] == "3a357be041df101c"
    assert citation["page_start"] == 3
    assert citation["page_end"] == 4
    assert citation["section"] == "Unit 3"
    assert citation["chunk_id"] == "3a357be041df101c:0"
    assert body["retrieval"] == {
        "configuration": "v1",
        "candidates": ["3a357be041df101c:0"],
        "scores": [0.9],
    }
    assert env.pipeline.calls == [
        ("what does borrow mean?", "v1", ["3a357be041df101c"])
    ]


def test_ask_applies_v2a_default_and_preserves_explicit_v1(env) -> None:
    default_response = env.client.post("/ask", json={"question": "job vs work?"})
    explicit_response = env.client.post(
        "/ask", json={"question": "borrow?", "config": "v1"}
    )

    assert default_response.status_code == 200
    assert explicit_response.status_code == 200
    assert env.pipeline.calls == [
        ("job vs work?", "v2a", None),
        ("borrow?", "v1", None),
    ]

    rejected = env.client.post("/ask", json={"question": "q", "config": "v9"})

    assert rejected.status_code == 400
    assert "v9" in rejected.json()["detail"]


def test_non_pdf_and_oversize_rejected_with_4xx(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("ingest must not run for a rejected upload")

    monkeypatch.setattr(api_module, "ingest_pdf", fail_if_called)

    not_pdf = env.client.post(
        "/documents",
        files={
            "file": ("a.pdf", b"plain text with no magic header", "application/pdf")
        },
    )
    assert not_pdf.status_code == 400
    assert "PDF" in not_pdf.json()["detail"]

    monkeypatch.setattr(api_module, "MAX_UPLOAD_BYTES", 1000)
    oversize = env.client.post(
        "/documents",
        files={"file": ("big.pdf", b"%PDF-1.4" + b"x" * 993, "application/pdf")},
    )
    assert oversize.status_code == 413

    # Rejected before any write: no directory, no document, no ingest call.
    assert env.store.documents == {}
    assert not env.data_dir.exists()


def test_client_filename_with_path_traversal_never_used_for_storage(env) -> None:
    response = env.client.post(
        "/documents",
        files={"file": ("../../escape.pdf", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    doc_id = doc_id_from_bytes(PDF_BYTES)
    # Exactly one file exists anywhere under the test root: the doc_id-named
    # copy inside the data directory. Nothing escaped it.
    stored_files = sorted(
        p.relative_to(env.tmp_path).as_posix()
        for p in env.tmp_path.rglob("*")
        if p.is_file()
    )
    assert stored_files == [f"uploads/{doc_id}.pdf"]
    body = response.json()
    assert body == {
        "doc_id": doc_id,
        "source": "escape.pdf",  # sanitized client name, display only
        "chunk_count": 1,
        "skipped": False,
    }
    assert env.store.documents[doc_id][0] == "escape.pdf"


def test_upload_display_filename_is_safe_and_has_pdf_extension(env) -> None:
    response = env.client.post(
        "/documents",
        files={"file": ("<script>\nnotes.txt", PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 201
    source = response.json()["source"]
    assert "<" not in source and ">" not in source and "\n" not in source
    assert source.endswith(".pdf")
    assert env.store.documents[doc_id_from_bytes(PDF_BYTES)][0] == source


def test_upload_same_bytes_twice_lists_one_document(env) -> None:
    first = env.client.post(
        "/documents", files={"file": ("book.pdf", PDF_BYTES, "application/pdf")}
    )
    second = env.client.post(
        "/documents", files={"file": ("book.pdf", PDF_BYTES, "application/pdf")}
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["skipped"] is False
    assert second.json()["skipped"] is True
    # The duplicate's count and source come from the existing indexed record.
    assert second.json()["chunk_count"] == first.json()["chunk_count"] == 1
    assert second.json()["source"] == "book.pdf"

    # A retry with different display metadata must not mutate the indexed one.
    renamed = env.client.post(
        "/documents", files={"file": ("renamed.pdf", PDF_BYTES, "application/pdf")}
    )
    assert renamed.status_code == 201
    assert renamed.json()["source"] == "book.pdf"
    assert renamed.json()["skipped"] is True
    assert env.store.documents[doc_id_from_bytes(PDF_BYTES)][0] == "book.pdf"
    assert len(env.store.documents) == 1
    assert len(list(env.data_dir.iterdir())) == 1
    assert (env.data_dir / f"{doc_id_from_bytes(PDF_BYTES)}.pdf").exists()
    assert not (env.data_dir / "renamed.pdf").exists()

    listed = env.client.get("/documents")
    assert listed.status_code == 200
    documents = listed.json()
    assert len(documents) == 1
    assert documents[0] == {
        "doc_id": doc_id_from_bytes(PDF_BYTES),
        "source": "book.pdf",
        "chunk_count": 1,
    }


def test_chunk_cap_per_document_rejects_huge_parse(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_ingest(path, store, embedder, size, overlap=64, timeout_s=300):
        # The tail of a real ingest whose parse produced a huge document:
        # one whole-document upsert carrying over-cap chunks. The ``store``
        # received here is the API's guard store.
        chunks = [f"chunk-{i}" for i in range(MAX_CHUNKS_PER_DOCUMENT + 1)]
        store.upsert_document("overcap", path.name, chunks, [[0.0]] * len(chunks))
        return IngestResult(doc_id="overcap", n_chunks=len(chunks), skipped=False)

    monkeypatch.setattr(api_module, "ingest_pdf", fake_ingest)

    response = env.client.post(
        "/documents", files={"file": ("huge.pdf", PDF_BYTES, "application/pdf")}
    )

    assert response.status_code == 422
    assert "chunk" in response.json()["detail"]
    # The guard raised before the inner store saw anything, and the written
    # upload file was discarded.
    assert env.store.documents == {}
    assert not any(env.data_dir.iterdir())


def test_ingest_error_maps_to_422_and_discards_file(
    env, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_parse(path, max_pages=300, timeout_s=300):
        raise IngestError("PDF contains no usable text")

    monkeypatch.setattr(ingest_module, "parse_pdf", failing_parse)

    response = env.client.post(
        "/documents", files={"file": ("empty.pdf", PDF_BYTES, "application/pdf")}
    )

    assert response.status_code == 422
    assert "usable text" in response.json()["detail"]
    assert env.store.documents == {}
    assert not any(env.data_dir.iterdir())


def test_serve_binds_loopback_only() -> None:
    # Unit-level: the host constant serve() passes to uvicorn is loopback.
    assert HOST == "127.0.0.1"


@pytest.mark.parametrize("error_kind", ["generation", "embedding", "rerank"])
def test_provider_errors_return_generic_502(env, error_kind, monkeypatch):
    from docsqa.api_rerank import RerankerError
    from docsqa.embedding import EmbeddingError
    from docsqa.generation import GeneratorError

    def fail(*args, **kwargs):
        raise {
            "generation": GeneratorError,
            "embedding": EmbeddingError,
            "rerank": RerankerError,
        }[error_kind]("sentinel-private")

    monkeypatch.setattr(env.pipeline, "ask", fail)
    response = env.client.post("/ask", json={"question": "q"})
    assert response.status_code == 502
    assert response.json() == {"detail": "provider request failed"}


def runtime_app_fakes(monkeypatch, tmp_path, fail=False):
    from docsqa import bm25
    from docsqa.runtime import RuntimeComponents
    from docsqa.settings import Settings

    class Store(FakeStore):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args, self.kwargs, self.closed = args, kwargs, False

        def ensure_schema(self):
            if fail:
                raise RuntimeError("sentinel-db-url")

        def close(self):
            self.closed = True

        def check_ready(self):
            return True

    store = Store()
    monkeypatch.setattr(api_module, "VectorStore", lambda *args, **kwargs: store)
    embedder = FakeEmbedder()
    probes = []
    original = embedder.embed_documents

    def embed(texts):
        probes.extend(texts)
        return original(texts)

    embedder.embed_documents = embed
    components = RuntimeComponents(
        SimpleNamespace(generate=lambda messages: "answer"),
        embedder,
        SimpleNamespace(rerank=lambda query, hits, top_k: hits[:top_k]),
    )
    monkeypatch.setattr(
        "docsqa.runtime.build_runtime_components", lambda settings: components
    )
    monkeypatch.setattr(
        "docsqa.runtime.runtime_index_identity", lambda *args: "runtime-v1:hash"
    )
    monkeypatch.setattr(bm25.BM25Index, "from_store", lambda store: SimpleNamespace())
    return Settings(_env_file=None, runtime_upload_dir=str(tmp_path)), store, probes


def test_runtime_composition_and_shutdown(monkeypatch, tmp_path):
    settings, store, probes = runtime_app_fakes(monkeypatch, tmp_path)
    app = api_module.build_runtime_app(settings)
    assert probes == ["runtime dimension probe"]
    pipeline = app.state.pipeline
    assert set(pipeline.retrievers) == {"v1", "v2a", "v2b", "v2b_norerank"}
    assert pipeline.final_k == 5 and pipeline.generation_k == 1
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert not store.closed
    assert store.closed


def test_runtime_startup_failure_closes_and_hides_context(monkeypatch, tmp_path):
    import traceback

    from docsqa.runtime import RuntimeConfigurationError

    settings, store, _ = runtime_app_fakes(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeConfigurationError) as caught:
        api_module.build_runtime_app(settings)
    assert store.closed
    assert caught.value.__context__ is None
    assert "sentinel" not in "".join(traceback.format_exception(caught.value))


def test_readiness_is_not_provider_liveness(monkeypatch, tmp_path):
    settings, store, _ = runtime_app_fakes(monkeypatch, tmp_path)
    app = api_module.build_runtime_app(settings)
    monkeypatch.setattr(
        "docsqa.provider_http.post_json",
        lambda *args: pytest.fail("readiness must not call providers"),
    )
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["providers"] == "configured"
        monkeypatch.setattr(store, "check_ready", lambda: False)
        response = client.get("/ready")
        assert response.status_code == 503
        assert client.get("/health").status_code == 200


def test_runtime_captures_store_options_and_generation_k(monkeypatch, tmp_path):
    captured = {}
    settings, store, _probes = runtime_app_fakes(monkeypatch, tmp_path)

    def spy(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return store

    monkeypatch.setattr(api_module, "VectorStore", spy)
    settings.generation_k = 3
    app = api_module.build_runtime_app(settings)
    assert captured["kwargs"]["connect_timeout"] == 3
    assert captured["kwargs"]["statement_timeout_ms"] == 2000
    assert app.state.pipeline.generation_k == 3
    assert app.state.runtime_index_validated is True
