# Runtime API Provider Design

Date: 2026-11-19
Status: Scope approved in conversation after pre-implementation audit; revised implementation details below are builder decisions pending plan review.

## Intent and approved scope

Prepare Docs-QA's FastAPI infrastructure for Render without Ollama inference. Cloud generation, embeddings and reranking use external APIs; V2a remains the default. The Gradio UI remains a separate local client. Local development retains independently selectable Ollama/API generation and embeddings and local/API reranking.

The user approved this scope after reviewing `docs/superpowers/plans/2026-11-19-runtime-api-provider-audit.md`. Approval does not authorize a public deployment, real provider calls, corpus egress, destructive reindexing, commits or new branches.

## Approaches considered

- **Three API adapters behind a runtime factory (selected):** retains the V2a pipeline in cloud and allows mixed local development; requires independent provider configuration and index safety.
- **Hosted chat/embeddings with a local cross-encoder:** retains local model startup and resource requirements in cloud; does not satisfy approved API-only inference.
- **Hosted chat/embeddings without reranking:** simpler, but silently replacing V2a with `v2b_norerank` violates the approved retrieval contract.

## Constraints

- Preserve handwritten retrieval, query canonicalization, citations, abstention and all four retrieval configurations. Keep V2a default and existing `generation_k` behavior.
- Runtime credentials/settings never fall back to `API_JUDGE_*`. Existing eval settings and held-out artifacts remain unchanged.
- No external services in unit tests or ordinary smoke tests; use injected/mocked HTTP transport. Do not run held-out evaluation in this pass.
- No real provider calls in this implementation session without separate explicit endpoint/privacy/cost approval. Credentials present in an environment are not approval.
- No public deployment, production corpus upload or destructive reset in this pass. Deployment artifacts are preparation, not certification that public operation is safe.
- No credentials, request/response bodies, document excerpts or provider URLs in errors, tracebacks or logs. Non-secret model identifiers may appear in identity metadata, not raw settings dumps.
- Local serving defaults to `127.0.0.1:8000`; cloud defaults to `0.0.0.0` and requires a valid assigned `PORT`.
- No provider SDKs; use stdlib HTTP and existing Python/FastAPI/Pydantic/pgvector stack.

## Current baseline

At `e51325d`, local Stage 3 tests pass (166 tests); API startup directly constructs Ollama generation/embedding and a local cross-encoder. `Embedder` is a protocol; generation/reranking are duck-typed. `index_meta` guards model/backend/dimension, but missing metadata can be inserted over existing chunks. UI is a loopback Gradio client configured through `DOCSQA_API_BASE`. Uploaded PDFs live under `data/uploads`.

## Settings and profiles — builder decisions

Use enum `RuntimeProfile(local, cloud)` and explicit provider settings:

| Setting | Default / accepted values |
|---|---|
| `runtime_profile` | `local` / `cloud` |
| `runtime_generation_provider` | `ollama` / `openai_compatible` |
| `runtime_embedding_provider` | `ollama` / `openai_compatible` |
| `runtime_rerank_provider` | `local` / `api` |
| `runtime_chat_base_url`, `runtime_chat_token`, `runtime_chat_model` | unset |
| `runtime_embedding_base_url`, `runtime_embedding_token`, `runtime_embedding_model` | unset |
| `runtime_rerank_base_url`, `runtime_rerank_token`, `runtime_rerank_model` | unset |
| `runtime_embedding_query_instruction` | `Given a web search query, retrieve relevant passages` |
| `runtime_api_timeout_s` | 30; finite, greater than 0 and at most 120 |
| `runtime_upload_dir` | `data/uploads` local; explicit absolute path required in cloud |
| `runtime_deployment_approved` | false; explicit public-security gate, not automatic approval |

Use `SecretStr` for tokens and base URLs; reject blank tokens/models for selected API providers. Validate models as nonblank trimmed strings without control characters. Field errors contain field names and generic messages, never input values; hide Pydantic input values in rendered validation errors. Do not alter existing Ollama/eval settings or their defaults.

`validate_runtime_profile(settings) -> None` validates all selected provider configuration before any provider construction or network request. Cloud requires all three explicit API provider selectors and complete credentials; reject local selectors. Merely having unused `OLLAMA_*` variables in the environment is harmless and must not construct Ollama. Cloud additionally requires HTTPS provider URLs and configured absolute upload path. Settings validation remains network-free. Public bind authorization is checked by the serving entrypoint, not inferred from valid provider settings.

Task 1 defines profile validation and component protocols, not the component factory. Task 5 constructs adapters after Tasks 2–4 exist. Local CrossEncoder import/construction is lazy; cloud adapter construction must neither import nor load CrossEncoder. PDF parsing remains local Docling work and is explicitly outside API-only *inference* claims.

## Shared HTTP contract — builder decisions

Add `provider_http.py` for endpoint normalization and bounded JSON POST. Adapters use separate endpoint credentials even if providers share a host.

- Accept HTTP(S) absolute URLs with hostname; reject userinfo, query, fragment, whitespace/control characters and absent host. Cloud settings require HTTPS; HTTP is available for local development only.
- A root URL becomes `/v1/<endpoint>`; a non-root custom prefix is preserved (`/gateway/v1` stays `/gateway/v1`). Strip trailing slashes and one recognized terminal suffix (`chat/completions`, `embeddings`, `rerank`) before appending the requested suffix. Do not duplicate `/v1`. Do not lowercase case-sensitive paths.
- POST JSON with `Content-Type: application/json` and `Authorization: Bearer <token>`. Disable **all** redirects: no destination receives credentials through a redirect. No automatic retries or fallback providers in this pass.
- Finite socket timeout from settings; read at most 8 MiB plus one byte and reject oversize responses before JSON parsing. This is a socket timeout, not a hard overall request deadline; document that limitation.
- Reject malformed JSON, non-object envelopes, non-2xx statuses and non-standard NaN/Infinity JSON constants. Transport errors become fixed application errors. Never incorporate provider bodies, URLs or headers. Raise sanitized exceptions outside the caught-exception handler so neither `__context__` nor `__cause__` retains raw provider exceptions. Do not log provider exceptions.
- Transport uses an injectable opener boundary for deterministic tests. Tests prove redirect rejection and captured output/tracebacks do not contain sentinel secrets, URLs or request text.

## Generation contract

`OpenAICompatibleGenerator(host, model, token, timeout=30.0, max_tokens=256)` implements `generate(messages: list[dict[str, str]]) -> str`.

POST `/chat/completions` with model, messages, `temperature: 0`, `stream: false`, `max_tokens`. Require nonempty choices and a string `choices[0].message.content`; blank content is a malformed provider response. Return content without modifying citation markers. All API failures raise existing `GeneratorError` with fixed `generation provider request failed`. Keep Ollama behavior and error contract unchanged.

## Embedding contract

`OpenAICompatibleEmbedder(host, model, token, query_instruction, timeout=30.0)` implements existing `Embedder`, with `backend = "openai_compatible"`, `dim = 0` until successful discovery.

POST `/embeddings` with model and input list. `embed_documents([])` returns `[]` without HTTP. Embed documents as supplied: ingest/retrieval callers already own NFC/whitespace/accent canonicalization; do not add a circular import or change that boundary. `embed_query(q)` applies exactly `Instruct: {query_instruction}\nQuery: {q}` when instruction is nonempty; an explicitly empty instruction sends the query unchanged for models that do not use instructions.

Require `data` to cover every requested index exactly once. Indices are integers, not booleans, in `[0, n)`. Restore request order. Vectors are nonempty, finite numeric (not boolean), same-dimension lists with nonzero norm; normalize using a numerically stable norm. Lock dimension only after validating the entire successful response and reject later dimension drift. Failed responses do not mutate `dim`. Raise fixed `EmbeddingError("embedding provider request failed")` without raw exception chains.

## Reranking contract — one explicitly supported wire shape

`ApiReranker(host, model, token, timeout=30.0)` implements `rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]`.

Use a Cohere-style **v1 rerank JSON shape** at the configured root/custom prefix, not a claim of compatibility with every provider:

```json
{"model":"configured-model","query":"question","documents":["passage A","passage B"],"top_n":2}
```

```json
{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}
```

Request all candidates (`top_n = len(hits)`), then select locally. Require exactly one result per input candidate, unique in-range non-boolean integer indices and finite non-boolean numeric `relevance_score`. Do not require a particular score range. Sort descending by score, ties by original candidate index; preserve original Chunk objects and replace only Hit scores. Empty hits or `top_k <= 0` returns `[]` without HTTP; clamp large top_k to available candidates. Reject incomplete top-N responses under this full-list contract. Raise fixed `RerankerError("reranking provider request failed")` without raw exception chains. Production provider must be verified against this contract before deployment.

## Index identity and legacy policy — builder decisions

Add optional `identity: str | None = None` to `EmbedderMeta` for compatibility with existing callers and eval. Runtime always supplies a versioned identity:

`runtime-v1:` + SHA-256 of canonical JSON (sorted keys, compact separators) containing provider, canonical embedding endpoint (or normalized Ollama origin + `/api/embed`), model, discovered dimension, query instruction, and preprocessing version `canonical-text-v1`.

Only the hash and prefix are persisted as identity; never store a raw API endpoint. Reject credentials/query/fragment in URLs before hashing. `runtime_index_identity(settings, embedder) -> str` is called only after a successful non-sensitive dimension probe. Runtime requires `1 <= dim <= 2000` for the existing `vector` HNSW index; this is an index capability limit, not an embedding API limit.

- **Existing legacy callers without identity:** retain current model/backend/dim checks against legacy metadata; they cannot open a runtime-identified index with unknown identity.
- **Runtime over populated legacy metadata, missing metadata, mismatched identity, or incompatible schema:** fail closed with generic rebuild-required diagnostics before inserting metadata, altering schema or modifying chunks.
- **Unchanged populated local legacy indexes also require explicit migration/rebuild for new runtime startup.** No automatic backfill: old metadata cannot prove preprocessing/endpoint identity. Legacy eval callers remain compatible with old indexes.
- **Empty existing legacy index:** reject with explicit fresh-index instruction rather than implicitly adopting it; a genuinely fresh schema can create runtime metadata/DDL atomically.
- Keep missing-metadata stale-chunk guard for all callers, including legacy callers. Check table presence and populated chunks before inserting any metadata. Guard concurrent initialization with a transaction-scoped advisory lock and atomic DDL/metadata creation. Close/rollback on failure.

No reset command is executed in this pass. Document separate fresh database/reindex steps, backups and approval before deletion. Use only `docsqa_test` for DB integration verification; mock ordering/no-mutation tests must also run without Postgres.

## Runtime composition and API lifecycle

Define `Generator` and `RuntimeReranker` protocols plus `RuntimeComponents(generator, embedder, reranker)`. `build_runtime_components(settings)` validates first, then creates selected adapters; it does not probe providers or open a DB. Local factory loads CrossEncoder only if local reranking was selected.

`build_runtime_app(settings) -> FastAPI` builds the actual stack, probes embeddings with `"runtime dimension probe"`, enforces index identity, constructs all four existing retrievers and creates Pipeline with unchanged generation/final-k semantics. Failures close any created store and emit sanitized startup diagnostics, not a raw DB URL or provider traceback. API request provider failures return a generic 502 with no upstream detail. App shutdown closes its store. Keep `create_app(pipeline, store, data_dir)` dependency-injection usage working for existing tests/UI.

## Serving, readiness, storage and cloud boundary

- `docsqa.serve:create_runtime_app()` is an ASGI factory; `docsqa.serve:main()` is the production CLI. Keep `api.serve()` as a backwards-compatible wrapper. Add a direct Uvicorn dependency and console script `docsqa-api`.
- `resolve_bind(settings, environ)` returns local default `127.0.0.1:8000` or cloud default `0.0.0.0:<PORT>`. Validate integer PORT in `[1,65535]`. Local HOST overrides are restricted to loopback. Cloud startup refuses to bind unless `runtime_deployment_approved` is explicitly true; Render template leaves it false.
- `/health`: process liveness only, returns 200 `{"status":"ok"}` and does no network work.
- `/ready`: no provider HTTP requests. Return sanitized state for config, initialized/identity-validated index, and bounded database `SELECT 1`; 200 if all pass, 503 otherwise. Label provider state `configured`, **not** live/healthy. Use DB connection timeout 3s and server statement timeout 2s on runtime DB connections. Provider startup proof is embedding-only; generator/reranker live availability remains unverified until separately approved smoke. Authentication errors and provider outages during requests do not silently downgrade retrieval.
- Before dimension probing/DB work, validate upload path: local may use `data/uploads`; cloud requires explicit existing absolute writable directory. Verify writeability with a temporary sentinel and cleanup. Reject cloud symlink paths/ancestors to avoid misleading mount resolution; existence/writeability do not prove persistence.
- First deployment template uses a single-instance persistent disk mounted at `/var/data` and uploads `/var/data/uploads`; create upload subdirectory in the container startup script under the mounted parent before app startup. Object storage and multi-instance shared uploads are follow-ups. Document disk/DB backup and restore pairing.
- Docker excludes `.env*`, `.git`, `.venv`, eval artifacts, local uploads and user PDFs; retain only a sanitized example if needed. No build-time provider secrets or model downloads. Keep existing Docling/ML dependencies initially; this is not a lightweight image claim. Local PDF parsing may need model assets/network and substantial CPU/RAM; asset provisioning/resource smoke is a separate pre-deployment gate.
- Render preparation is API-only. UI remains loopback and points at the API through `DOCSQA_API_BASE`. No hosted UI claim.
- Before public deployment: approved auth, rate limits, CORS policy, quotas, provider budget/data-retention terms, pgvector availability, persistent mount/backups and parser assets/resources. The boolean deployment gate is an operator acknowledgement, **not an implementation** of these controls.

## Verification and rollout

1. RED → GREEN tests for settings, transport, each adapter, factory, index guard, API lifecycle and deployment gates. No external calls. Keep 166-test baseline regressions green; preserve eval import isolation.
2. Full pytest, Ruff including new scripts, diff check and tracked-source secret sweep. Docker/Render artifacts get static tests; report container build/platform validation separately if not actually executed.
3. Provide `scripts/runtime_api_smoke.py`: requires `--allow-network`, complete cloud settings read from process environment only (`_env_file=None`), and prior explicit approval. Without flag return a generic skipped message and make zero HTTP calls; with flag but invalid settings exit nonzero before construction. Use synthetic generation/embedding/rerank inputs only, fixed pass/fail counters, no provider diagnostics. Flag is a technical interlock, not evidence of privacy approval. Do not run the network-enabled path in this pass.
4. Later, after approval, verify selected provider wire contracts, provision a fresh index and approved corpus, and validate ingestion/QA separately. Never rerun held-out here.
5. Only after public security/infrastructure gates are met, stage API deployment and verify health/readiness/citations. On failure stop or roll back; never silently select Ollama or disable reranking in cloud.

## Outstanding operator decisions

Production endpoints/models/dimensions, provider retention/cost approval, production auth/rate-limit policy, persistent disk capacity/backups, pgvector service compatibility, parser assets and Render resource sizing remain deployment gates. They do not prevent implementing the explicitly documented mocked contracts above.

## Approval boundary

The three-provider/V2a/API-only scope was approved. Rerank wire shape, bounded transport, strict legacy-index rebuild policy, identity encoding and deployment interlock are explicit builder decisions introduced by this revision and are highlighted for plan review before implementation.
