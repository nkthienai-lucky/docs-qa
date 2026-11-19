from __future__ import annotations

import http.client
import json
import os
import secrets
from html import escape
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import gradio as gr

from .pipeline import AnswerResponse, Citation, RetrievalInfo
from .uploads import DEFAULT_MAX_BYTES

_RETRIEVAL_CONFIGS = [
    ("V1 · Dense vector search", "v1"),
    ("V2a · Dense search + reranking", "v2a"),
    ("V2b · Hybrid search + reranking", "v2b"),
    ("V2b · Hybrid search, no reranking", "v2b_norerank"),
]
_UPLOAD_TIMEOUT_S = 960


def answer_display_text(answer: str) -> str:
    """Return answer text unchanged for a plain-text Gradio component."""
    return answer


def render_answer(answer: str) -> str:
    """Return the answer as plain text without parsing Markdown."""
    return answer_display_text(answer)


def render_citation(citation: Citation) -> str:
    """Render citation metadata and excerpt as plain text."""
    section = citation.section if citation.section else "(no section)"
    return (
        f"Source: {citation.source}\n"
        f"Pages: {citation.page_start}-{citation.page_end}\n"
        f"Section: {section}\n"
        f"Chunk ID: {citation.chunk_id}\n"
        f"Excerpt: {citation.excerpt}"
    )


def _citation_label(index: int, citation: Citation) -> str:
    page_label = (
        f"page {citation.page_start}"
        if citation.page_start == citation.page_end
        else f"pages {citation.page_start}–{citation.page_end}"
    )
    section = citation.section or "(no section)"
    return (
        f"[{index}] {citation.source} · {page_label} · {section}"
    )


def _citation_labels(citations: list[Citation]) -> list[str]:
    return [_citation_label(index, citation) for index, citation in enumerate(citations, 1)]


def _selected_citation_detail(
    citations: list[Citation], selected: str | int | None
) -> str:
    if not citations or selected in (None, ""):
        return "Select a source to inspect its excerpt."

    if isinstance(selected, int):
        index = selected - 1
    else:
        labels = _citation_labels(citations)
        try:
            index = labels.index(selected)
        except ValueError:
            return "Select a source to inspect its excerpt."
    if index < 0 or index >= len(citations):
        return "Select a source to inspect its excerpt."
    return render_citation(citations[index])


def _make_citation_selection_handler(citations: list[Citation]):
    """Return a deterministic handler for selecting one citation."""
    labels = _citation_labels(citations)

    def handle_selection(selected: str | int | None) -> tuple[list[str], str]:
        return labels, _selected_citation_detail(citations, selected)

    return handle_selection


def _make_ask_ui_handler(client: object):
    """Ask handler that also populates the selectable citation inspector."""
    def handle_ask(
        question: str, config: str, doc_ids: list[str] | None
    ) -> tuple[str, Any, str, list[Citation], str]:
        try:
            selected = [value for value in (doc_ids or []) if value]
            response = client.ask(question, config, selected or None)
            labels = _citation_labels(response.citations)
            selected_label = labels[0] if labels else None
            detail = _selected_citation_detail(response.citations, selected_label)
            return (
                answer_display_text(response.answer),
                gr.update(choices=labels, value=selected_label),
                detail,
                response.citations,
                (
                    "Answer abstained; no supported answer was found."
                    if response.abstained
                    else "Answer ready."
                ),
            )
        except (ValueError, RuntimeError, OSError, http.client.HTTPException, urllib_error.URLError) as error:
            return (
                "",
                gr.update(choices=[], value=None),
                "Select a source to inspect its excerpt.",
                [],
                f"Unable to get an answer: {_safe_error(error)}",
            )

    return handle_ask


def _answer_response(data: dict[str, Any]) -> AnswerResponse:
    citations = [Citation(**item) for item in data.get("citations", [])]
    retrieval_data = data.get("retrieval") or {}
    retrieval = RetrievalInfo(
        configuration=retrieval_data.get("configuration", "v1"),
        candidates=retrieval_data.get("candidates", []),
        scores=retrieval_data.get("scores", []),
    )
    return AnswerResponse(
        answer=str(data.get("answer", "")),
        abstained=bool(data.get("abstained", False)),
        citations=citations,
        retrieval=retrieval,
    )


class _Response(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class _Transport(Protocol):
    def get(self, url: str) -> _Response: ...

    def post(self, url: str, **kwargs: Any) -> _Response: ...


class _UrllibResponse:
    def __init__(self, response: Any) -> None:
        self._response = response

    def raise_for_status(self) -> None:
        status = getattr(self._response, "status", 200)
        if status >= 400:
            detail = self._response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {detail}")

    def json(self) -> Any:
        return json.loads(self._response.read().decode("utf-8"))


class _UrllibTransport:
    """Standard-library transport that streams bounded multipart uploads."""

    def __init__(self, timeout: float = _UPLOAD_TIMEOUT_S) -> None:
        self.timeout = timeout

    def _send(self, request: urllib_request.Request) -> _UrllibResponse:
        try:
            response = urllib_request.urlopen(request, timeout=self.timeout)
        except urllib_error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
        return _UrllibResponse(response)

    def get(self, url: str) -> _Response:
        return self._send(urllib_request.Request(url))

    def _post_multipart(
        self, url: str, filename: str, file_obj: Any, content_type: str
    ) -> _Response:
        parsed = urllib_parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("API base must be an HTTP or HTTPS URL")
        boundary = "docsqa-" + secrets.token_hex(16)
        prefix = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{Path(filename).name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
        file_size = file_obj.seek(0, 2)
        file_obj.seek(0)
        content_length = len(prefix) + file_size + len(suffix)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(parsed.hostname, parsed.port, timeout=self.timeout)
        target = urllib_parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            connection.putrequest("POST", target)
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(content_length))
            connection.endheaders()
            connection.send(prefix)
            while chunk := file_obj.read(64 * 1024):
                connection.send(chunk)
            connection.send(suffix)
            return _UrllibResponse(connection.getresponse())
        except Exception:
            connection.close()
            raise

    def post(self, url: str, **kwargs: Any) -> _Response:
        if "json" in kwargs:
            body = json.dumps(kwargs["json"]).encode("utf-8")
            request = urllib_request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            return self._send(request)

        filename, file_obj, content_type = kwargs["files"]["file"]
        return self._post_multipart(url, filename, file_obj, content_type)


class _ApiClient:
    """Small injectable client boundary for the existing FastAPI routes."""

    def __init__(self, api_base: str, transport: _Transport | None = None) -> None:
        self.api_base = api_base.rstrip("/")
        self._transport = transport or _UrllibTransport()

    def list_documents(self) -> list[dict[str, Any]]:
        response = self._transport.get(f"{self.api_base}/documents")
        response.raise_for_status()
        return response.json()

    def upload_document(self, filepath: str | Path) -> dict[str, Any]:
        path = Path(filepath)
        size = path.stat().st_size
        if size > DEFAULT_MAX_BYTES:
            raise ValueError(
                f"upload of {size} bytes exceeds the {DEFAULT_MAX_BYTES:,}-byte limit"
            )
        with path.open("rb") as file:
            response = self._transport.post(
                f"{self.api_base}/documents",
                files={"file": (path.name, file, "application/pdf")},
            )
        response.raise_for_status()
        return response.json()

    def ask(
        self, question: str, config: str, doc_ids: list[str] | None
    ) -> AnswerResponse:
        response = self._transport.post(
            f"{self.api_base}/ask",
            json={"question": question, "config": config, "doc_ids": doc_ids},
        )
        response.raise_for_status()
        return _answer_response(response.json())


def _safe_error(error: Exception) -> str:
    return escape(str(error), quote=True)


def _make_ask_handler(client: object):
    def handle_ask(
        question: str, config: str, doc_ids: list[str] | None
    ) -> tuple[str, str, str]:
        try:
            selected = [value for value in (doc_ids or []) if value]
            response = client.ask(question, config, selected or None)
            citation_text = "\n\n".join(
                render_citation(citation) for citation in response.citations
            )
            return (
                answer_display_text(response.answer),
                citation_text or "No citations available.",
                (
                    "Answer abstained; no supported answer was found."
                    if response.abstained
                    else "Answer ready."
                ),
            )
        except (ValueError, RuntimeError, OSError, http.client.HTTPException, urllib_error.URLError) as error:
            return "", "No citations available.", f"Unable to get an answer: {_safe_error(error)}"

    return handle_ask


def build_ui(api_base: str, client: object | None = None) -> gr.Blocks:
    """Build a Gradio client for document upload, retrieval, and QA."""
    api = client if client is not None else _ApiClient(api_base)
    css = """
    :root { --body-text-size: 16px; }
    body, textarea, input, button { font-family: "Atkinson Hyperlegible", system-ui, sans-serif; }
    textarea, input { font-size: var(--body-text-size) !important; }
    button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 3px solid #1e3a5f !important;
        outline-offset: 2px;
    }
    """

    with gr.Blocks(title="Docs QA", css=css) as app:
        gr.Markdown("# Docs QA")
        gr.Markdown(
            "Upload a PDF, ask a question, and inspect the exact source used for the answer."
        )

        gr.Markdown("## 1. Add documents")
        with gr.Row():
            upload = gr.File(
                label="PDF file",
                file_types=[".pdf"],
                type="filepath",
            )
            upload_button = gr.Button("Upload and index", variant="secondary")
        upload_status = gr.Textbox(label="Upload status", interactive=False)

        documents = gr.CheckboxGroup(
            label="Limit search to these documents (optional)",
            choices=[],
            info="Leave all unchecked to search every indexed document.",
        )
        with gr.Row():
            refresh_button = gr.Button("Refresh document list", variant="secondary")
            document_status = gr.Textbox(
                label="Document list status", interactive=False
            )

        gr.Markdown("## 2. Ask a question")
        question = gr.Textbox(
            label="Question",
            placeholder="Ask a question about the indexed documents",
            lines=3,
        )
        config = gr.Dropdown(
            label="Retrieval strategy",
            choices=_RETRIEVAL_CONFIGS,
            value="v2a",
            info=(
                "Dense search finds semantically similar passages. Hybrid adds keyword search; "
                "reranking reorders candidates for relevance."
            ),
        )
        ask_button = gr.Button("Ask", variant="primary")
        ask_status = gr.Textbox(label="Answer status", interactive=False)

        gr.Markdown("## 3. Answer and sources")
        answer = gr.Textbox(label="Answer", lines=6, interactive=False)
        citation_state = gr.State([])
        citation_picker = gr.Radio(
            label="Sources (select one to inspect)",
            choices=[],
            type="value",
            interactive=True,
        )
        citation_detail = gr.Textbox(
            label="Selected source excerpt",
            value="Select a source to inspect its excerpt.",
            lines=12,
            interactive=False,
        )
        with gr.Accordion("About retrieval strategies", open=False):
            gr.Markdown(
                "- **V1 · Dense vector search:** semantic matching; fastest baseline.\n"
                "- **V2a · Dense + reranking:** reranks semantic candidates.\n"
                "- **V2b · Hybrid + reranking:** combines semantic and keyword search, then reranks.\n"
                "- **V2b · Hybrid, no reranking:** combines semantic and keyword search without the reranking step."
            )

        def refresh_documents() -> tuple[Any, str]:
            try:
                found = api.list_documents()
                choices = [
                    (
                        f"{item.get('source', '')} ({item.get('doc_id', '')})",
                        item.get("doc_id", ""),
                    )
                    for item in found
                ]
                return gr.update(choices=choices, value=[]), (
                    f"{len(found)} document(s) ready."
                    if found
                    else "No indexed documents yet. Upload a PDF to get started."
                )
            except (ValueError, RuntimeError, OSError, http.client.HTTPException, urllib_error.URLError) as error:
                return gr.update(choices=[], value=[]), (
                    f"Could not load documents: {_safe_error(error)}"
                )

        def upload_document(filepath: str | None) -> tuple[str, Any, str, str]:
            if not filepath:
                return "", gr.update(), "", "Choose a PDF to upload."
            try:
                result = api.upload_document(filepath)
                group, list_status = refresh_documents()
                source = escape(
                    str(result.get("source", Path(filepath).name)), quote=True
                )
                if list_status.startswith("Could not load documents:"):
                    return (
                        "",
                        group,
                        list_status,
                        f"Uploaded {source}, but the document list did not refresh.",
                    )
                return "", group, list_status, f"Uploaded and indexed {source}."
            except (ValueError, RuntimeError, OSError, http.client.HTTPException, urllib_error.URLError) as error:
                return "", gr.update(), "", f"Upload failed: {_safe_error(error)}"

        def show_selected_citation(
            selected: str | None, citations: list[Citation]
        ) -> str:
            return _selected_citation_detail(citations, selected)

        refresh_button.click(
            refresh_documents,
            outputs=[documents, document_status],
            show_progress="minimal",
        )
        app.load(
            refresh_documents,
            outputs=[documents, document_status],
            show_progress="minimal",
        )
        upload_button.click(
            upload_document,
            inputs=[upload],
            outputs=[upload, documents, document_status, upload_status],
            show_progress="full",
        )
        ask_button.click(
            _make_ask_ui_handler(api),
            inputs=[question, config, documents],
            outputs=[
                answer,
                citation_picker,
                citation_detail,
                citation_state,
                ask_status,
            ],
            show_progress="full",
        )
        citation_picker.change(
            show_selected_citation,
            inputs=[citation_picker, citation_state],
            outputs=[citation_detail],
        )

    return app


def serve_ui() -> Any:
    """Launch the UI using the configured local API base URL."""
    api_base = os.environ.get("DOCSQA_API_BASE", "http://127.0.0.1:8000")
    app = build_ui(api_base)
    return launch_ui(app, port=7860)


def launch_ui(
    app: gr.Blocks,
    *,
    port: int = 7860,
    inbrowser: bool = False,
    prevent_thread_lock: bool = False,
) -> Any:
    """Launch a UI on loopback unless the caller explicitly changes the port."""
    return app.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=inbrowser,
        prevent_thread_lock=prevent_thread_lock,
    )
