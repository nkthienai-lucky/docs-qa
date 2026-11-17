# Gradio UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin Gradio UI for document upload, indexed-document listing, question answering, and safe citation inspection without changing the FastAPI or pipeline contracts.

**Architecture:** `src/docsqa/ui.py` will be a client-only presentation layer that talks to the existing FastAPI endpoints through a small injectable HTTP client boundary. It will keep answer and citation content as plain text, escape untrusted values for display, and expose a `gr.Blocks` app from `build_ui(api_base)`. The UI will not import `docsqa.eval`; tests will use a fake API client/HTTP transport and will not require PostgreSQL, Ollama, or a browser.

**Tech Stack:** Python 3.12, Gradio 5.x, standard-library HTML escaping and HTTP client support, pytest.

**Spec:** `AI-Engineer-Roadmap/docs/superpowers/specs/2026-11-13-docs-qa-stage-3-design.md` (UI/citation contract and security requirements); Stage 3 plan Task 8.

## Global Constraints

- No LangChain or LlamaIndex imports anywhere in `src/docsqa`.
- The UI must support upload/index, indexed-document listing, question entry, grounded answer display, and revealable source excerpts.
- Citation excerpts and model answers must render as inert plain text; raw HTML, Markdown links, and Markdown images must not be auto-rendered.
- The UI must not import `docsqa.eval`.
- API base URL and request transport remain configurable/injectable; no credentials or document content is hard-coded.
- The app launch path must bind to `127.0.0.1`.

## Review Focus

- Citation text containing `<script>` or Markdown image syntax must remain inert text; `test_render_citation_escapes_html_and_markdown_images` pins this in `tests/test_ui.py`.
- An answer containing HTML/Markdown must not become rendered markup; `test_answer_links_and_images_not_auto_rendered` pins this in `tests/test_ui.py`.
- A citation with a missing/empty optional section must still render usable page and chunk metadata; `test_render_citation_handles_missing_section` pins this in `tests/test_ui.py`.
- API responses with no citations or an abstention must not crash the event handlers; `test_ui_handles_abstention_without_citations` pins this in `tests/test_ui.py`.
- UI construction must remain independent of eval modules and external services; `test_ui_builds_with_fake_api` pins this in `tests/test_ui.py`.

---

### Task 1: Safe citation rendering and API client boundary

**Files:**
- Create: `src/docsqa/ui.py`
- Test: `tests/test_ui.py`

**Interfaces:**
- Consumes: `Citation`, `AnswerResponse`, and the existing FastAPI routes (`GET /documents`, `POST /documents`, `POST /ask`).
- Produces: `render_citation(c: Citation) -> str`, `build_ui(api_base: str, client: object | None = None) -> gr.Blocks`, and a small internal client boundary whose methods map to the existing API routes.

- [ ] **Step 1: Write failing tests**
  - `test_render_citation_escapes_html_and_markdown_images`: build a `Citation` whose excerpt contains `<script>alert(1)</script>` and `![x](http://evil)`, then assert the result contains no executable/raw tag and preserves the text as escaped/inert content.
  - `test_render_citation_handles_missing_section`: render a citation with `section=None` and assert page range and `chunk_id` are present without the literal string `None`.
  - `test_answer_links_and_images_not_auto_rendered`: pass an answer containing `<a href=...>` and Markdown image syntax through the answer-display helper, then assert the result is plain text/inert.
  - `test_ui_handles_abstention_without_citations`: use a fake API returning `abstained=True`, an abstention answer, and an empty citation list; invoke the ask handler and assert it returns the answer plus an empty/neutral citation state without raising.
  - `test_ui_builds_with_fake_api`: construct the UI with a fake client, assert it is a `gr.Blocks`, and verify the module source/import graph contains no `docsqa.eval` import.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest -q tests/test_ui.py`

Expected: FAIL because `docsqa.ui` and its public helpers do not exist yet.

- [ ] **Step 3: Implement safe presentation helpers in `src/docsqa/ui.py`**
  - Define `render_citation(c: Citation) -> str` using HTML escaping for every untrusted field and a readable plain-text layout containing source, page range, optional section, chunk ID, and excerpt.
  - Define an answer-display helper that escapes HTML-sensitive characters and does not return a Markdown-rendered component value.
  - Keep the UI output components configured for text/plain behavior rather than Markdown/HTML rendering.
  - Use an injectable client object for deterministic tests; the default client sends requests to `api_base` using the existing route shapes and does not expose tokens or settings.

- [ ] **Step 4: Implement `build_ui(api_base: str, client: object | None = None) -> gr.Blocks`**
  - Add upload control and an upload action that posts the selected PDF to `/documents`.
  - Add an indexed-document list populated from `GET /documents`.
  - Add question input, retrieval-config selection (`v1`, `v2a`, `v2b`, `v2b_norerank`), optional document selection, and an ask action that posts `{question, config, doc_ids}` to `/ask`.
  - Display the answer as plain text and expose citations as a selectable plain-text list/details area showing page range, section, chunk ID, source, and excerpt.
  - Ensure handlers tolerate empty documents, empty citations, abstentions, and API errors by returning user-readable plain-text status rather than raising through the Gradio event loop.

- [ ] **Step 5: Run focused tests to verify they pass**

Run: `.venv/bin/pytest -q tests/test_ui.py`

Expected: all UI tests pass without PostgreSQL or Ollama.

- [ ] **Step 6: Run the full suite and static checks**

Run: `.venv/bin/pytest -q` and `git diff --check`.

Expected: existing tests remain green, UI tests pass, and `git diff --check` is clean. A live browser/API smoke test is separate and may remain unavailable while PostgreSQL/Ollama are stopped.

- [ ] **Step 7: Commit the independently testable UI change**

```bash
git add src/docsqa/ui.py tests/test_ui.py docs/superpowers/plans/2026-11-17-docs-qa-gradio-ui.md
git commit -m "feat: add Gradio UI with citation inspection"
```

The commit message must end with:

```text
Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
```

## UI usage

Install the project into the active environment, start the FastAPI service separately, then run `docsqa-ui` (or `uv run docsqa-ui`). The UI reads its API base URL from `DOCSQA_API_BASE`, defaulting to `http://127.0.0.1:8000`, and binds only to `127.0.0.1:7860`.
