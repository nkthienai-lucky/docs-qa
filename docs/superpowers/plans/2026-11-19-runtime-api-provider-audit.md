# Runtime API provider plan audit — 2026-11-19

## Scope and evidence

Reviewed HEAD `e51325d` on branch `stage3-eval`, the runtime implementation plan and design, settings, API startup, embedding/reranking adapters, store metadata guard, UI entrypoint, and current tests.

- Current local Stage 3 baseline: `uv run pytest -q` → **166 passed, 4 deprecation warnings**, 65.25s. An earlier attempt timed out at 120s; the successful retry is the current evidence.
- `uv run pytest -q -m 'not integration'` → **161 passed, 5 deselected**.
- `uv run ruff check src tests eval` → passed.
- `git diff --check` → passed before this documentation update.
- Existing README records 80/80 held-out rerun results and 72/72 in-scope API scores. This audit did not rerun or independently rescore the benchmark.
- Runtime provider implementation: **0/7 tasks complete**. `runtime.py`, API adapters, deployment entrypoint, Dockerfile, and Render blueprint are not present. API startup still constructs Ollama generation/embedding and a local cross-encoder directly.

## Verdict

**Local baseline verified; new plan needs clarification before implementation.** The seven-task direction is viable, but the draft spec contradicts itself and several task interfaces lack executable acceptance criteria. This is an audit, not design approval or deployment authorization.

## Findings and required plan amendments

### Important — contradictory scope and approval

Design status is `Draft for approval`. Goals and adapter section require external reranking in cloud, while non-goal 34 excludes it and open decision 127 still permits local CPU reranking or `v2b_norerank`. Non-goal 29 and the approval section prohibit implementation-time provider calls, but plan Task 7 permits a real-API smoke based on environment configuration.

Required: explicitly approve API-only cloud with all three providers and V2a retained, or revise the plan to a narrower scope. Remove stale contradictory statements after approval. Runtime credentials being present are not evidence of permission to send data or incur charges. Keep real-API smoke opt-in and require explicit endpoint/privacy/cost approval; skip it otherwise.

### Important — Task 1 / Task 5 factory dependency

Task 1 advertises a factory returning adapters that Tasks 2–4 have not implemented; Task 5 implements that factory again. Cloud no-Ollama *construction* cannot be fully verified by settings validation alone.

Required: Task 1 owns provider settings, profile validation and component/protocol types only. Task 5 owns actual construction and constructor-spy tests, including proving no local cross-encoder construction in cloud. Validate all cloud configuration before constructing any provider or probing the network. Preserve existing eval settings.

### Important — reranking HTTP contract is not documented

Task 4 promises a documented contract but neither document specifies path, request fields, response envelope, score field, full-list versus top-N semantics, or tie behavior. These differ across providers.

Required: freeze one supported wire contract with example mocked payloads before Task 4. Define index coverage, unique integer indices (reject booleans), finite numeric scores, tie/order semantics, empty hits/top_k behavior and bounds. Do not claim arbitrary reranker compatibility. Provider selection can remain a deployment gate if the implemented contract is named explicitly.

### Important — endpoint/security/error contract

URL normalization is requested without specifying custom prefixes, cross-endpoint suffixes or redirect handling. A bearer token must not be forwarded to an unexpected redirect target. Sanitized outer exception messages alone do not guarantee that chained HTTP exceptions, response bodies or request URLs stay out of logs.

Required: define accepted root, `/v1`, custom-prefix and endpoint-suffixed URL behavior once, shared across adapters. Reject URL credentials/query/fragment and require HTTPS in cloud (mock transport in tests); specify a no-cross-origin-redirect policy. Bound timeouts and response sizes, and test malformed JSON/envelopes, status failures and sanitization of logs as well as visible messages. Keep DB/provider secrets out of startup diagnostics.

### Important — index identity and legacy behavior

Current `EmbedderMeta` contains only model/backend/dim. `ensure_schema()` inserts metadata when row 1 is missing without first rejecting existing chunks. Task 5 identifies this problem but does not define migration of existing local indexes, canonical endpoint identity or inclusion of query-instruction semantics.

Required: define a versioned, non-secret identity including provider, canonical endpoint identity, model, dimension and query instruction/preprocessing semantics. Never persist a credential-bearing URL. Reject populated legacy/missing/mismatched metadata before mutation; document explicit fresh-index/rebuild steps and require approval before destructive reset. Specify whether unchanged local legacy indexes require migration or rebuild. Add mocked DB guard tests plus integration tests on `docsqa_test` only. Guard positive dimensions and supported pgvector/HNSW dimensions before DDL; changing model names alone does not establish vector compatibility.

### Important — API-only inference versus deployment footprint

`rerank.py` imports `sentence_transformers.CrossEncoder` at module import. Docling and local ML packages remain ordinary dependencies. API-only inference therefore does not by itself mean a lightweight cloud image or no local model downloads during parsing.

Required: distinguish API-only generation/embedding/reranking from PDF parsing requirements. Make local reranker loading lazy where needed, test that cloud construction never loads a cross-encoder, and explicitly document/package Docling model assets and resource requirements. Task 6 must include `pyproject.toml`/lock changes if adding a direct production server dependency or console script; do not rely implicitly on transitive Uvicorn availability.

### Important — readiness, storage, lifecycle and public deployment gates

Task 6 lists readiness and persistent storage but lacks status semantics, retry/time budgets, cleanup and enforcement details. An absolute directory path alone does not prove durable storage. Auth/rate-limit/CORS/quotas/cost controls are mandatory in the spec but have no implementation task.

Required: define `/health` as process liveness and readiness as a bounded, sanitized dependency/config/index check with 503 on failure; clarify whether providers are actually probed or merely configured. No paid provider calls on every health request. Close DB resources on shutdown and failed startup. Require a writable configured upload path and a Render persistent mount, document backup/single-instance limits, and test missing/unwritable paths. Keep deployment disabled pending explicit security controls, provider approval and infrastructure checks; a blueprint is not deployment readiness certification.

### Clarification — UI deployment boundary

The current UI is a separate loopback Gradio client using `DOCSQA_API_BASE`. Task 6 supplies a FastAPI entrypoint, not a hosted UI.

Required: state that this pass deploys **API infrastructure only**, retaining local UI access, unless a separate UI design is approved. Do not describe API smoke-import as an end-to-end deployed UI smoke.

### Minor — benchmark provenance wording

`README.md` says V2a was predeclared from dev tuning, whereas `eval/protocol-v2/README.md` says it was selected from the held-out rerun. Resolve against the dated decision evidence before changing either statement. This audit does not infer the selection history from benchmark scores.

## Suggested execution order after approval

1. Reconcile spec and approve the three-provider cloud contract, rerank wire contract and legacy-index policy.
2. Task 1: settings/profile validation and protocols only; test defaults, independent local selection, invalid cloud config, judge isolation and secret-safe diagnostics.
3. Tasks 2–4: mocked HTTP adapters with shared URL/transport safety policy, RED → GREEN per adapter.
4. Task 5: component factory, legacy-safe metadata guard and API wiring; preserve all four retrieval modes and V2a default.
5. Task 6: API-only ASGI lifecycle, readiness, persistent storage and deployment artifacts; no public publication.
6. Task 7: docs and offline verification; real API smoke only with separate explicit opt-in approval. No held-out rerun.

Do not create commits or branches merely because the original plan includes commit steps; obtain the user's explicit instruction first. Existing untracked user files are out of scope.
