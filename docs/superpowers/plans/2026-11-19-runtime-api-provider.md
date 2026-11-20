# Runtime API Providers and Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. No subagent execution tools are available in the current harness.

**Goal:** Implement local mixed-provider and cloud three-API inference, preserving V2a and preparing API-only Render infrastructure without publishing a service.

**Architecture:** Settings validation is network-free and precedes component construction. Three adapters share a bounded, redirect-denying JSON transport. A runtime factory wires existing retrieval/pipeline code behind a versioned index identity guard; ASGI lifecycle and serving configuration enforce storage/public-bind gates.

**Tech Stack:** Python 3.12, urllib, Pydantic Settings, FastAPI/Uvicorn, PostgreSQL/pgvector, pytest, Ruff, Docker/Render.

**Spec:** `docs/superpowers/specs/2026-11-19-runtime-api-provider-design.md` (revised after scope approval).

**Audit:** `docs/superpowers/plans/2026-11-19-runtime-api-provider-audit.md` records findings against the previous draft; this revision addresses them, not the original audit verdict retrospectively.

## Global Constraints

- Preserve handwritten retrieval, query canonicalization, citations, abstention and all four retrieval configurations. Keep V2a default and existing `generation_k` behavior.
- Runtime credentials/settings never fall back to `API_JUDGE_*`. Existing eval settings and held-out artifacts remain unchanged.
- No external services in unit tests or ordinary smoke tests; use injected/mocked HTTP transport. Do not run held-out evaluation in this pass.
- No real provider calls in this implementation session without separate explicit endpoint/privacy/cost approval. Credentials present in an environment are not approval.
- No public deployment, production corpus upload or destructive reset in this pass. Deployment artifacts are preparation, not certification that public operation is safe.
- No credentials, request/response bodies, document excerpts or provider URLs in errors, tracebacks or logs. Non-secret model identifiers may appear in identity metadata, not raw settings dumps.
- Local serving defaults to `127.0.0.1:8000`; cloud defaults to `0.0.0.0` and requires a valid assigned `PORT`.
- No provider SDKs; use stdlib HTTP and existing Python/FastAPI/Pydantic/pgvector stack.
- No commits or branches without a separate explicit user instruction. Record task completion in the plan ledger instead of executing the superseded plan's commit steps.
- Preserve user-owned untracked files. DB integration uses `docsqa_test` only, never production `docsqa`.

## Review Focus

- Redirects and exception chains leak bearer tokens/provider text despite sanitized outer errors: transport and adapter tests in Tasks 2–4.
- Inherited Ollama variables or partial cloud configuration trigger local inference/model loading: validation/construction spies in Tasks 1 and 5.
- Shuffled/incomplete/nonfinite embedding and rerank results silently misassociate chunks or mutate dimension: adversarial response tests in Tasks 3–4.
- Missing/legacy metadata over stale chunks or concurrent startup adopts an incompatible index: pre-mutation and transaction tests in Task 5.
- A successful `/health` or writable ephemeral directory is mistaken for public deployment readiness: explicit state/interlock/storage tests in Task 6 and docs tests in Task 7.

## File responsibilities

- `settings.py`: runtime configuration; `runtime.py`: profile validation, protocols, component factory and identity; no serving side effects.
- `provider_http.py` (new): URL policy, redirect denial, bounded POST JSON, sanitized transport errors.
- `generation.py`, `embedding.py`: existing local adapters plus API implementations; `api_rerank.py` (new): external reranker without local ML imports; `rerank.py`: local cross-encoder with lazy import.
- `store.py`: metadata/schema safety and bounded runtime connection/check behavior; `api.py`: composition, routes and lifecycle.
- `serve.py` (new): bind/deployment/storage interlocks, CLI and ASGI factory; deployment files: Docker/Render packaging only.
- `scripts/runtime_api_smoke.py` (new): separately approved, opt-in synthetic provider checks; docs explain unimplemented public-security gates.

## Execution setup and evidence

Baseline at `e51325d`: 166 tests passed, Ruff passed. Recheck working tree, read revised spec and audit, and create this plan's ignored `.superpowers/sdd/2026-11-19-runtime-api-provider/progress.md` ledger before implementation. Verify current workspace through `using-git-worktrees`; do not create a branch without permission. Use `test-driven-development` and `verification-before-completion`. Ledger each cross-task interface/ruling and each RED → GREEN command/result; never mark a task complete solely because code exists.

Every task below has its own failing-test gate and green regression gate. Tests use `Settings(_env_file=None, ...)` plus cleared relevant environment variables. Runtime/HTTP tests must not load models, call providers or require DB. Do not import API judge adapters into runtime.

---

### Task 1: Runtime settings, validation and protocols

**Files:** Modify `src/docsqa/settings.py`, `.env.example`, `tests/test_settings.py`; create `src/docsqa/runtime.py`, `tests/test_runtime.py`.

**Interfaces:**
- `RuntimeProfile(str, Enum)` with `LOCAL="local"`, `CLOUD="cloud"`.
- `validate_runtime_profile(settings: Settings) -> None`; fixed `RuntimeConfigurationError` diagnostics.
- Protocols `Generator.generate(messages: list[dict[str,str]]) -> str`, `RuntimeReranker.rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]`; dataclass `RuntimeComponents(generator: Generator, embedder: Embedder, reranker: RuntimeReranker)`.
- No component factory until Task 5. Use `TYPE_CHECKING` where appropriate to avoid importing local ML providers.

- [ ] Write `test_runtime_defaults_are_local`, `test_local_provider_choices_are_independent`, `test_cloud_requires_three_explicit_api_providers`, `test_cloud_rejects_each_incomplete_api`, `test_runtime_never_uses_judge_credentials`, `test_cloud_ignores_unused_ollama_variables`, `test_runtime_errors_hide_input_values`. Assert exact settings/defaults from spec, reject blank/control-character model/token values, HTTPS violations, relative cloud upload path, and timeout values 0/NaN/Infinity/121. Existing Ollama/eval settings must remain unchanged. Validating settings must perform no HTTP or provider construction.
- [ ] Run `uv run pytest tests/test_runtime.py tests/test_settings.py -q`. Expected: new missing-symbol/validation assertions fail, not external-service errors.
- [ ] Implement spec settings table and protocols; `SecretStr` token/base URLs; enum/Literal selectors; hide input values in rendered validation errors. Add selected-provider URL policy validation locally in Task 1, then share canonical policy with Task 2 without relaxing it.
- [ ] Run the same command. Expected: all pass. Record commands/results and Task 1 completion in ledger.

### Task 2: Safe HTTP transport and generation

**Files:** Create `src/docsqa/provider_http.py`, `tests/test_provider_http.py`; modify `src/docsqa/generation.py`, `src/docsqa/runtime.py`, `tests/test_generation.py`.

**Interfaces:**
- `provider_endpoint(host: str, endpoint: str) -> str` where endpoint is `chat/completions`, `embeddings`, or `rerank`; implements exact spec root/custom-prefix/suffix policy.
- `post_json(url: str, token: str, payload: dict, timeout: float) -> dict`; fixed `ProviderHTTPError`, no raw exception chain. Module-level `open_provider_request(request, timeout)` is injectable, production implementation uses a redirect-denying urllib opener.
- `OpenAICompatibleGenerator(host: str, model: str, token: str, timeout: float = 30.0, max_tokens: int = 256)`; existing `GeneratorError`.

- [ ] Write transport tests: root→`/v1`, existing `/v1`, `/gateway/v1`, trailing slash, recognized same/cross endpoint suffixes, rejected userinfo/query/fragment/control characters, denied 301/302/307/308, non-object/malformed/NaN JSON, oversize 8 MiB+1 body, HTTP/timeout errors. Use a fake opener; inspect actual Request headers/body/method and bounded read size. Capture formatted traceback/stdout/stderr to assert sentinel token/URL/body absent and sanitized exception has no cause/context.
- [ ] Write generation tests asserting exact model/messages/temperature=0/stream=false/max_tokens payload, valid content/citation preservation, blank content and malformed envelopes rejected, fixed sanitized `GeneratorError` for HTTP and parse failures.
- [ ] Run `uv run pytest tests/test_provider_http.py tests/test_generation.py -q`. Expected: new transport/adapter tests fail before implementation.
- [ ] Implement transport and generator, reusing shared URL validation in runtime. Never log/re-raise raw provider exceptions. Retain existing Ollama behavior.
- [ ] Run `uv run pytest tests/test_provider_http.py tests/test_generation.py tests/test_runtime.py tests/test_settings.py -q`. Expected: pass. Record evidence and Task 2 completion.

### Task 3: API embeddings

**Files:** Modify `src/docsqa/embedding.py`, `tests/test_embedding.py`; use existing `tests/test_retrievers.py`, `tests/test_ingest.py` for regressions.

**Interfaces:** `OpenAICompatibleEmbedder(host: str, model: str, token: str, query_instruction: str, timeout: float = 30.0)` implements existing `Embedder`; new `EmbeddingError`.

- [ ] Write tests for exact request/auth, out-of-order `[1,0]` restored to request order, unit vectors, instruction template and empty instruction, empty input without HTTP, and no dimension mutation on failed response. Parametrize missing/duplicate/boolean/out-of-range indices, wrong count, mixed/drifting/empty dimension, boolean/string elements, NaN/Infinity and zero vectors. Test very large/small finite values with stable normalization, and fixed errors/secret-free tracebacks. Existing caller canonicalization must remain unchanged; no extra canonicalization inside adapter.
- [ ] Run `uv run pytest tests/test_embedding.py -q`. Expected: new API-adapter tests fail.
- [ ] Implement using Task 2 transport; validate entire response before setting `dim`; normalize robustly without changing Ollama/eval contract.
- [ ] Run `uv run pytest tests/test_embedding.py tests/test_retrievers.py tests/test_ingest.py -q`. Expected: pass. Record evidence and Task 3 completion.

### Task 4: API reranking and local loading boundary

**Files:** Create `src/docsqa/api_rerank.py`, `tests/test_api_rerank.py`; modify `src/docsqa/rerank.py`, `tests/test_rerank.py`.

**Interfaces:** `ApiReranker(host: str, model: str, token: str, timeout: float = 30.0)` implements Task 1 protocol; `RerankerError`. Existing local `Reranker(...)` remains compatible; CrossEncoder import moves to local construction.

- [ ] Write tests for spec Cohere-style request (`top_n=len(hits)`) and `results/index/relevance_score` response. Assert descending score order, ties by original index, exact original Chunk identity, top_k clamp, and empty/nonpositive no-HTTP behavior. Reject partial/duplicate/missing/out-of-range/boolean indices and nonfinite/boolean scores. Assert sanitized timeout/HTTP/malformed errors and traceback redaction. Add isolated import test proving API reranker/runtime protocol imports do not import sentence-transformers/CrossEncoder; adapt existing local tests to mock the lazy import boundary.
- [ ] Run `uv run pytest tests/test_api_rerank.py tests/test_rerank.py -q`. Expected: new API/import-boundary tests fail.
- [ ] Implement strict full-list rerank contract via Task 2 transport and lazy local CrossEncoder construction. No arbitrary-provider compatibility claim.
- [ ] Run the same command. Expected: pass. Record evidence and Task 4 completion.

### Task 5: Runtime composition and fail-closed index identity

**Files:** Modify `src/docsqa/runtime.py`, `src/docsqa/store.py`, `src/docsqa/api.py`, `tests/test_runtime.py`, `tests/test_store.py`, `tests/test_api.py`; create `tests/test_index_identity.py` (DB-free).

**Interfaces:**
- `build_runtime_components(settings: Settings) -> RuntimeComponents` (no probes/DB).
- `runtime_index_identity(settings: Settings, embedder: Embedder) -> str`: spec canonical JSON SHA-256; requires discovered dim 1..2000.
- `EmbedderMeta(model, backend, dim, identity: str | None = None)`; legacy callers remain supported against legacy indexes only.
- `build_runtime_app(settings: Settings) -> FastAPI`: compose existing four retrievers, `generation_k`, Pipeline and injected `create_app`; close store on failure/shutdown.
- Add `VectorStore.check_ready() -> bool`, connect/statement timeout keyword options with runtime values 3s/2s and unchanged defaults for legacy callers.

- [ ] Write constructor-spy tests: invalid cloud config constructs nothing; valid cloud constructs exactly three API adapters, never Ollama/CrossEncoder; local mixed choices work. Building components performs no HTTP. Identity is stable across URL suffix/trailing slash equivalents, changes for provider/endpoint/model/dim/instruction, and stores only `runtime-v1:<hash>`, never raw URL/token.
- [ ] Write DB-free guard/order tests with a fake connection: stale chunks + missing metadata, populated legacy runtime index, empty legacy runtime index, mismatches and invalid dimensions all fail before metadata/DDL/chunk mutation. Legacy caller cannot open identity-bearing index. Fresh runtime creation is atomic with transaction-scoped advisory lock; failure rolls back. Retain valid legacy model/backend/dim behavior. Integration tests verify fresh/mismatch/legacy cases using `docsqa_test` and verify serialized initialization under two connections.
- [ ] Write API composition tests with fake providers/store: probe is exactly `runtime dimension probe`, all four retrieval modes exist, V2a/default generation limits preserved, failed init closes DB, shutdown closes DB, provider request failures return generic 502. Assert startup diagnostics/tracebacks never include sentinel DB URL/token/provider body.
- [ ] Run `uv run pytest tests/test_runtime.py tests/test_index_identity.py tests/test_store.py tests/test_api.py -q`. Expected: new factory/identity/lifecycle assertions fail.
- [ ] Implement factory, identity/schema guard and API composition; inspection/lock precedes writes. Do not reset or backfill populated legacy indexes. Keep dependency-injected create_app compatible; do not automatically start serving here.
- [ ] Run `uv run pytest tests/test_runtime.py tests/test_index_identity.py tests/test_store.py tests/test_api.py tests/test_pipeline.py tests/test_retrievers.py tests/test_import_hygiene.py -q`. Expected: pass, including test DB integrations. Record evidence and strict legacy rebuild policy in ledger; mark Task 5 complete.

### Task 6: ASGI serving, readiness and deployment artifacts

**Files:** Create `src/docsqa/serve.py`, `Dockerfile`, `.dockerignore`, `render.yaml`, `scripts/container_start.sh`, `tests/test_deployment.py`; modify `src/docsqa/api.py`, `pyproject.toml`, `uv.lock`, `tests/test_api.py`.

**Interfaces:**
- `resolve_bind(settings: Settings, environ: Mapping[str,str]) -> tuple[str,int]`.
- `validate_upload_dir(settings: Settings) -> Path`: spec local/cloud rules and sentinel write/cleanup.
- `create_runtime_app() -> FastAPI`: settings/storage validation before Task 5 composition; `main() -> None` serves without duplicate initialization. Preserve `api.serve()` wrapper; console script `docsqa-api`.
- GET `/health`: 200 process-only; GET `/ready`: config/index/DB state + provider `configured`, 200/503 without provider HTTP.

- [ ] Write tests for local bind defaults, cloud assigned PORT and explicit deployment gate, bad/missing/boolean-like/out-of-range PORT, local non-loopback rejection. Validate cloud missing/relative/symlink/unwritable paths fail before any provider/DB calls; temporary sentinel cleanup on success/failure. Test readiness DB failure→503, health→200, no provider calls, no false live-provider claim, bounded DB checks and lifecycle cleanup.
- [ ] Write static packaging tests: direct Uvicorn dependency, console script and ASGI factory import, Docker startup command via `scripts/container_start.sh`, no secret build args, `.dockerignore` excludes env/PDF/data/eval/Git, Render single-instance disk `/var/data`, uploads `/var/data/uploads`, correct assigned port and deployment approval false. Startup script creates upload subdirectory only under configured mount; cloud app must reject absent mount/config rather than silently using ephemeral local uploads. Add Python subprocess offline import smoke for API/UI.
- [ ] Run `uv run pytest tests/test_deployment.py tests/test_api.py -q`. Expected: new serving/readiness/artifact tests fail.
- [ ] Implement entrypoint/readiness/storage and artifacts; add compatible stable direct Uvicorn dependency through uv. Use reproducible lock-based Docker install, no build-time model downloads/secrets. Render template is not auto-deployed; document parser resource/network limits. Validate mount presence in container script (not simply directory existence) before creating cloud upload subdirectory.
- [ ] Run `uv run pytest tests/test_deployment.py tests/test_api.py tests/test_runtime.py -q`; `uv run python -c "import docsqa.api, docsqa.ui, docsqa.serve"`. Expected: pass, no provider calls. If Docker available, build with bounded timeout and no provider credentials; report build separately, never substitute static tests for build evidence. Record Task 6 completion and any platform validation not run.

### Task 7: Documentation, opt-in smoke and final review

**Files:** Modify `README.md`, `.env.example`, `docs/corpus.md`, `tests/test_security_sweep.py`; create `scripts/runtime_api_smoke.py`, `tests/test_runtime_smoke.py`. Update this plan's ignored ledger.

**Interfaces:** `scripts/runtime_api_smoke.py:main(argv: list[str] | None = None) -> int`; uses process env only (`Settings(_env_file=None)`), requires `--allow-network`, and reports synthetic-test pass/fail counts without secrets.

- [ ] Write tests: no flag→generic skipped, exit 0, zero construction/HTTP; flag + incomplete cloud→nonzero, zero HTTP; valid mocked config→generation/embedding/rerank synthetic checks; failed synthetic semantics count as failure, not merely completed HTTP; sanitized output with no raw error, URL, model response, token or corpus text. Run under mocks only. Security/docs tests pin runtime/judge separation, V2a three-provider cloud, local UI boundary, legacy rebuild requirement and unimplemented public-security gates.
- [ ] Run `uv run pytest tests/test_runtime_smoke.py tests/test_security_sweep.py -q`. Expected: new smoke/docs tests fail.
- [ ] Implement smoke interlock and docs: exact env examples/CLI, full-list rerank contract, network/cost/retention implications, separate approved provider smoke, fresh DB/reindex guidance without running it, backup/mount/resource/parser requirements, API-only serving and UI connection. Preserve benchmark numbers/provenance; do not resolve predeclared-vs-held-out wording without dated evidence. Update status to distinguish implemented infrastructure from deployment authorization.
- [ ] Run `uv run pytest -q`, `uv run ruff check src tests eval scripts`, `git --no-pager diff --check`, `uv run pytest tests/test_security_sweep.py tests/test_import_hygiene.py -q`. Expected: pass. Run smoke without flag only; expected skip with zero network calls. Secret-scan new/untracked implementation files too; report paths/counts only, never matching secret text.
- [ ] Perform whole-change self-review against revised spec, audit and Review Focus; no fresh reviewer tool is available, so explicitly record weaker self-review. Fix important findings with RED → GREEN regression evidence; ledger minor deferrals. Update completed checkboxes only for verified tasks. Do not deploy, invoke network-enabled smoke, commit or delete another plan's workspace.

## Self-review of revised plan

- Scope conflicts resolved: cloud requires three APIs/V2a; hosted UI excluded; real smoke approval remains separate.
- Interface chain: Task 1 types → Tasks 2–4 adapters → Task 5 factory/index/API → Task 6 serving → Task 7 opt-in smoke/docs. No factory depends on an adapter before it exists.
- Audit coverage: transport/redirect/traceback Task 2; vector semantics Task 3; rerank wire contract Task 4; lazy local load Tasks 4–5; legacy index/atomicity Task 5; lifecycle/readiness/storage/dependencies/interlock Task 6; privacy/docs/secret sweep Task 7.
- Not claimed: production auth/rate limiting implementation, live-provider availability, lightweight image, parser asset provisioning, actual Docker/Render validation unless run, or benchmark revalidation.
- Builder decisions requiring review: Cohere-style full-list rerank response; strict runtime legacy rebuild policy; hashed versioned index identity; 8 MiB/30s HTTP limits; public-bind interlock.

## Handoff

Scope approved; revised plan is ready for user review before implementation. Recommend native inline execution because interfaces are sequential and no subagent tool is available. No new runtime code has been written by this planning revision.
