# Runtime API Providers and Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support local hybrid inference and cloud/Render API-only inference for generation, embeddings, and reranking, with fail-closed startup and tested deployment readiness.

**Architecture:** Keep Ollama adapters for local use, add provider-neutral runtime protocols and HTTP adapters for OpenAI-compatible chat/embeddings plus a documented reranking API contract. An explicit `local|cloud` profile factory selects providers; cloud refuses Ollama and incomplete API configuration. Preserve the existing pgvector metadata guard and add explicit embedding identity/rebuild safety before deployment.

**Tech Stack:** Python 3.12, urllib HTTP adapters, FastAPI, Pydantic Settings, PostgreSQL/pgvector, sentence-transformers local reranker, pytest, Ruff, Docker/Render.

**Spec:** `docs/superpowers/specs/2026-11-19-runtime-api-provider-design.md`

## Global Constraints

- Local mode may independently choose Ollama or OpenAI-compatible providers for generation and embeddings; reranking may be local or API.
- Cloud/Render mode must use external APIs for generation, embeddings, and reranking; it must never instantiate or fall back to Ollama.
- Runtime provider credentials/settings remain separate from `API_JUDGE_*` and existing eval Ollama settings.
- Unit tests never call external services; use mocked HTTP transport.
- Integration smoke may call the already-provided API credentials with non-sensitive test input only; never print keys, headers, URLs containing credentials, or corpus excerpts.
- Changing embedding provider/model/dimension requires an explicit fresh index or rebuild; stale vectors must not be silently reused.
- Local defaults remain loopback-only; cloud uses `0.0.0.0` and `$PORT`.
- Do not rerun held-out benchmark or publish a Render service in this implementation pass.

## Review Focus

- Cloud with Ollama variables present must still construct no Ollama adapter; test in Task 1.
- API embedding responses out of order, missing indices, duplicate indices, wrong dimensions, non-finite values, zero vectors, and empty input; test in Task 3.
- API URLs containing `/v1` or endpoint suffixes must not produce duplicate paths; test in Tasks 2–3.
- Malformed API success/error payloads and provider failures must become sanitized application errors; test in Tasks 2–4.
- Existing Ollama eval settings and query canonicalization must remain unchanged; test in Tasks 1 and 3.

---

### Task 1: Runtime profiles and settings

**Files:**
- Modify: `src/docsqa/settings.py`
- Create: `src/docsqa/runtime.py`
- Modify: `tests/test_settings.py`
- Create: `tests/test_runtime.py`
- Modify: `.env.example`

**Interfaces:**
- Produces `RuntimeProfile` with `local` and `cloud` values, runtime provider settings, and `validate_runtime_profile(settings)`.
- Produces factories that later tasks consume: `build_runtime_components(settings)` returning generator, embedder, reranker adapters.

- [ ] Write failing tests for profile defaults, independent local provider selection, cloud fail-closed validation, and no `API_JUDGE_*` fallback.
- [ ] Run focused tests and verify expected missing-symbol/validation failures.
- [ ] Implement settings/profile validation without changing existing eval settings.
- [ ] Run focused tests and full settings/runtime tests.
- [ ] Commit `feat: add local and cloud runtime profiles`.

### Task 2: OpenAI-compatible generation adapter

**Files:**
- Modify: `src/docsqa/generation.py`
- Modify: `tests/test_generation.py`

**Interfaces:**
- Produces `OpenAICompatibleGenerator(host, model, token, timeout, max_tokens)` implementing `generate(messages) -> str`.
- Base URL accepts root or `/v1`, strips endpoint suffixes, and sends `Authorization: Bearer`.

- [ ] Write failing mocked-HTTP tests for normalized URL, request fields, response parsing, malformed responses, HTTP errors, and secret redaction.
- [ ] Run focused tests red.
- [ ] Implement adapter and generic sanitized `GeneratorError` behavior.
- [ ] Run generation tests green and existing generation tests.
- [ ] Commit `feat: add API generation adapter`.

### Task 3: OpenAI-compatible embedding adapter

**Files:**
- Modify: `src/docsqa/embedding.py`
- Modify: `tests/test_embedding.py`

**Interfaces:**
- Produces `OpenAICompatibleEmbedder(host, model, token, query_instruction, timeout)` implementing the existing `Embedder` protocol.
- `embed_documents` and `embed_query` canonicalize inputs as current callers expect; response data is sorted by validated `index`.

- [ ] Write failing mocked-HTTP tests for URL/auth/request, ordering, empty input, dimension discovery, malformed/index/dimension/non-finite/zero-vector rejection, and error redaction.
- [ ] Run focused tests red.
- [ ] Implement adapter using existing normalization and query instruction contract.
- [ ] Run embedding tests green and retrieval/embedding regression tests.
- [ ] Commit `feat: add API embedding adapter`.

### Task 4: External reranker adapter

**Files:**
- Modify: `src/docsqa/rerank.py`
- Modify: `tests/test_rerank.py`

**Interfaces:**
- Produces `ApiReranker` with `rerank(query, hits, top_k) -> list[Hit]`.
- Validates returned candidate indices/scores and preserves chunk identity.

- [ ] Write failing mocked-HTTP tests for request shape, ranking/order, malformed/missing/duplicate candidate responses, timeout and redaction.
- [ ] Run focused tests red.
- [ ] Implement the documented external rerank contract and generic sanitized errors.
- [ ] Run reranker tests green.
- [ ] Commit `feat: add API reranker adapter`.

### Task 5: Runtime factory and index identity safety

**Files:**
- Modify: `src/docsqa/runtime.py`
- Modify: `src/docsqa/store.py`
- Modify: `src/docsqa/api.py`
- Modify: `tests/test_runtime.py`
- Modify: `tests/test_store.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- `build_runtime_components(settings)` constructs local mixed adapters or cloud API-only adapters, including reranker required by V2a.
- `runtime_index_identity(settings, embedder)` returns a non-secret provider/endpoint/model/dimension identity.

- [ ] Write failing tests for cloud no-Ollama construction, required three APIs, index identity mismatch, missing metadata with stale chunks, and local defaults.
- [ ] Run focused tests red.
- [ ] Implement factory and fail-closed index guard; require explicit fresh/rebuild path rather than automatic destructive reset.
- [ ] Run runtime/store/API tests green.
- [ ] Commit `feat: enforce cloud API-only runtime and index identity`.

### Task 6: Render entrypoint, health/readiness and storage boundary

**Files:**
- Modify: `src/docsqa/api.py`
- Create: `src/docsqa/serve.py`
- Create: `Dockerfile`
- Create: `render.yaml`
- Modify: `tests/test_api.py`
- Create: `tests/test_deployment.py`

**Interfaces:**
- Local serving remains `127.0.0.1:8000`; cloud serving uses `HOST`/`PORT` with Render defaults `0.0.0.0`/`$PORT`.
- Exposes ASGI factory/entrypoint compatible with Render.
- Readiness distinguishes config/DB/provider status without sending corpus text.
- First deployment uses configured persistent disk path for uploads; object storage remains a separate follow-up.

- [ ] Write failing tests for host/port parsing, ASGI creation, health/readiness semantics, Docker command and Render configuration.
- [ ] Run deployment tests red.
- [ ] Implement entrypoint/configuration and persistent upload path validation.
- [ ] Run deployment/API tests green and smoke-import API/UI.
- [ ] Commit `feat: add Render deployment readiness`.

### Task 7: Documentation, controlled real-API smoke and final verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/corpus.md`
- Modify: `docs/superpowers/specs/2026-11-19-runtime-api-provider-design.md`
- Modify: `.superpowers/sdd/.../progress.md`
- Create: `scripts/runtime_api_smoke.py`
- Modify: `tests/test_security_sweep.py`

**Interfaces:**
- Smoke script reads runtime API settings only from environment, uses a non-sensitive test string, reports pass/fail counts without secrets, and refuses to run if cloud profile is incomplete.

- [ ] Write failing documentation/security tests for API-only cloud instructions, no `API_JUDGE_*` runtime fallback, and smoke secret handling.
- [ ] Run focused tests red.
- [ ] Implement documentation and opt-in smoke script; run it only when required runtime API variables are present, otherwise report skipped configuration without contacting providers.
- [ ] Run full `uv run pytest -q`, `uv run ruff check src tests eval`, `git diff --check`, and secret sweep.
- [ ] Commit `docs: document API-only cloud deployment`.
- [ ] Perform final whole-branch review before any deployment decision.
