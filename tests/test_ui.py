from __future__ import annotations

import ast
from pathlib import Path

import gradio as gr
import pytest

from docsqa.pipeline import AnswerResponse, Citation, RetrievalInfo
from docsqa.ui import (
    _ApiClient,
    _make_ask_handler,
    _make_citation_selection_handler,
    answer_display_text,
    build_ui,
    launch_ui,
    render_citation,
    serve_ui,
)


class FakeApi:
    def __init__(self, response: AnswerResponse | None = None) -> None:
        self.response = response or AnswerResponse(
            answer="A grounded answer.",
            abstained=False,
            citations=[],
            retrieval=RetrievalInfo("v1", [], []),
        )
        self.asked: list[tuple[str, str, list[str] | None]] = []
        self.list_calls = 0

    def list_documents(self):
        self.list_calls += 1
        return []

    def upload_document(self, filepath):
        return {"doc_id": "doc1", "source": Path(filepath).name}

    def ask(self, question: str, config: str, doc_ids: list[str] | None):
        self.asked.append((question, config, doc_ids))
        return self.response


def test_render_citation_escapes_html_and_markdown_images() -> None:
    citation = Citation(
        source="<doc>.pdf",
        page_start=2,
        page_end=4,
        section="<script>section</script>",
        chunk_id="c<&1",
        excerpt="<script>alert(1)</script> ![x](http://evil)",
    )

    rendered = render_citation(citation)

    assert "<script>alert(1)</script>" in rendered
    assert "![x](http://evil)" in rendered
    assert "<doc>.pdf" in rendered
    assert "<script>section</script>" in rendered
    assert "c<&1" in rendered


def test_render_citation_handles_missing_section() -> None:
    citation = Citation(
        source="book.pdf",
        page_start=5,
        page_end=5,
        section=None,
        chunk_id="book:2",
        excerpt="Text",
    )

    rendered = render_citation(citation)

    assert "5" in rendered
    assert "book:2" in rendered
    assert "None" not in rendered


def test_answer_links_and_images_not_auto_rendered() -> None:
    answer = '<a href="https://evil">click</a> ![x](http://evil)'

    rendered = answer_display_text(answer)

    assert rendered == '<a href="https://evil">click</a> ![x](http://evil)'


def test_api_client_maps_existing_routes_and_multipart_upload(tmp_path: Path) -> None:
    calls: list[tuple[str, str, dict]] = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Transport:
        def get(self, url):
            calls.append(("GET", url, {}))
            return Response([])

        def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            return Response({"doc_id": "d", "source": "sample.pdf"})

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    api = _ApiClient("http://api.test/", transport=Transport())

    assert api.list_documents() == []
    assert api.upload_document(pdf)["doc_id"] == "d"
    assert [call[:2] for call in calls] == [
        ("GET", "http://api.test/documents"),
        ("POST", "http://api.test/documents"),
    ]
    upload = calls[1][2]
    assert upload["files"]["file"][0] == "sample.pdf"
    assert upload["files"]["file"][2] == "application/pdf"


def test_default_api_client_uses_timeout_longer_than_upload_parse_budget() -> None:
    api = _ApiClient("http://api.test")

    assert api._transport.timeout == 960


def test_oversized_upload_is_rejected_before_transport_post(tmp_path: Path) -> None:
    class Transport:
        def __init__(self):
            self.posts = 0

        def get(self, url):
            raise AssertionError("GET is not used for upload preflight")

        def post(self, url, **kwargs):
            self.posts += 1
            raise AssertionError("oversized upload must not reach transport")

    transport = Transport()
    api = _ApiClient("http://api.test", transport=transport)
    path = tmp_path / "oversized.pdf"
    path.write_bytes(b"x" * (30_000_001))

    with pytest.raises(ValueError, match="30,000,000-byte limit"):
        api.upload_document(path)

    assert transport.posts == 0


def test_ui_handles_abstention_without_citations() -> None:
    response = AnswerResponse(
        answer="I do not have enough information to answer.",
        abstained=True,
        citations=[],
        retrieval=RetrievalInfo("v1", [], []),
    )
    api = FakeApi(response)
    handler = _make_ask_handler(api)

    answer, citations, status = handler("What is unknown?", "v1", [])

    assert answer == "I do not have enough information to answer."
    assert citations in ("", "No citations available.")
    assert "abstain" in status.lower()
    assert api.asked == [("What is unknown?", "v1", None)]


def test_ui_builds_with_fake_api() -> None:
    api = FakeApi()

    app = build_ui("http://localhost:8000", client=api)

    assert isinstance(app, gr.Blocks)
    ui_path = Path(__file__).parents[1] / "src" / "docsqa" / "ui.py"
    module = ast.parse(ui_path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(name == "docsqa.eval" or name.startswith("docsqa.eval.") for name in imports)


def test_ui_loads_documents_and_separates_status_fields() -> None:
    app = build_ui("http://localhost:8000", client=FakeApi())

    components = {component.get("id"): component for component in app.config["components"]}
    labels = {
        component.get("props", {}).get("label")
        for component in components.values()
    }
    assert "Document list status" in labels
    assert "Upload status" in labels

    load_dependencies = [
        dependency
        for dependency in app.config["dependencies"]
        if any(target[1] == "load" for target in dependency.get("targets", []))
    ]
    assert len(load_dependencies) == 1
    assert len(load_dependencies[0]["outputs"]) == 2


def test_retrieval_options_explain_modes_and_keep_api_values() -> None:
    app = build_ui("http://localhost:8000", client=FakeApi())

    dropdown = next(
        component
        for component in app.config["components"]
        if component.get("props", {}).get("label") == "Retrieval strategy"
    )

    assert dropdown["props"]["value"] == "v2a"
    assert [tuple(choice) for choice in dropdown["props"]["choices"]] == [
        ("V1 · Dense vector search", "v1"),
        ("V2a · Dense search + reranking", "v2a"),
        ("V2b · Hybrid search + reranking", "v2b"),
        ("V2b · Hybrid search, no reranking", "v2b_norerank"),
    ]


def test_citation_selection_reveals_one_source_detail() -> None:
    citations = [
        Citation(
            source="book.pdf",
            page_start=2,
            page_end=3,
            section="Borrow and lend",
            chunk_id="book:1",
            excerpt="Borrow means take temporarily.",
        ),
        Citation(
            source="book.pdf",
            page_start=8,
            page_end=8,
            section="Present perfect",
            chunk_id="book:2",
            excerpt="The present perfect connects past and present.",
        ),
    ]

    handler = _make_citation_selection_handler(citations)

    labels, detail = handler(1)

    assert labels == ["[1] book.pdf · pages 2–3 · Borrow and lend", "[2] book.pdf · page 8 · Present perfect"]
    assert "Borrow and lend" in detail
    assert "book:1" in detail
    assert "Present perfect connects" not in detail


def test_serve_ui_uses_environment_api_url_and_loopback(monkeypatch) -> None:
    captured = {}

    def fake_build_ui(api_base):
        captured["api_base"] = api_base
        return "app"

    def fake_launch_ui(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs
        return "launched"

    monkeypatch.setenv("DOCSQA_API_BASE", "http://127.0.0.1:8123/")
    monkeypatch.setattr("docsqa.ui.build_ui", fake_build_ui)
    monkeypatch.setattr("docsqa.ui.launch_ui", fake_launch_ui)

    assert serve_ui() == "launched"
    assert captured == {
        "api_base": "http://127.0.0.1:8123/",
        "app": "app",
        "kwargs": {"port": 7860},
    }

def test_ui_launch_uses_loopback_default() -> None:
    class LaunchCapture:
        def __init__(self) -> None:
            self.kwargs = None

        def launch(self, **kwargs):
            self.kwargs = kwargs
            return kwargs

    app = LaunchCapture()

    assert launch_ui(app, port=7860, inbrowser=False, prevent_thread_lock=True) == {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "inbrowser": False,
        "prevent_thread_lock": True,
    }
    assert app.kwargs == {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "inbrowser": False,
        "prevent_thread_lock": True,
    }
