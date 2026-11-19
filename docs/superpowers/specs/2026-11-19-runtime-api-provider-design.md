# Runtime API Provider Design

Date: 2026-11-19
Status: Draft for approval

## Intent

Make the Docs-QA runtime deployable on Render without Ollama by replacing the runtime-only Ollama dependency with configurable OpenAI-compatible HTTP adapters for both generation and embeddings. The application must not enter a partially configured public deployment where generation works but embeddings do not, or where an old Ollama-built index is silently reused with a different embedding model.

The existing eval API judge configuration remains separate. Runtime provider credentials and endpoints must not reuse `API_JUDGE_*` settings, because eval judges may receive corpus excerpts under a different privacy policy.

## Goals

- Support runtime chat generation through a configurable OpenAI-compatible chat endpoint.
- Support runtime document/query embeddings through a configurable OpenAI-compatible embeddings endpoint.
- Support runtime reranking through a configurable external reranking API so cloud can retain the V2a retrieval contract.
- Keep Ollama and API providers available for local development.
- Define an explicit `local` profile where generation and embedding can independently use Ollama or OpenAI-compatible APIs, and reranking can use local CPU/CUDA or an API.
- Define an explicit `cloud` profile where generation, embedding, and reranking must use external APIs; cloud startup must reject Ollama configuration or any incomplete API side.
- Validate that generation and embedding configuration is complete before starting a Render deployment.
- Reindex the corpus when the runtime embedding identity or vector dimension changes; never silently mix indexes.
- Preserve the handwritten retrieval, citation, abstention, V1/V2a/V2b behavior, and eval-only API judge boundary.
- Add Render-compatible startup behavior using `0.0.0.0` and the assigned `$PORT`, while retaining loopback-only local defaults.
- Document required environment variables, provider capabilities, data-egress implications, and deployment steps.

## Non-goals

- No provider-specific SDK integration beyond the OpenAI-compatible HTTP contract in this phase.
- No automatic account creation, API-key provisioning, or external API calls during implementation/tests.
- No public deployment or corpus upload in this phase.
- No claim that an arbitrary provider supports embeddings merely because it supports chat.
- No reuse of the eval judge adapter as the runtime provider.
- No change to held-out benchmark scores or rerun of the held-out evaluation.
- No replacement of the local cross-encoder reranker with a hosted reranking API in this phase.

## Anti-goals and safety constraints

- Do not log API keys, authorization headers, full provider URLs when they may contain credentials, or document excerpts sent to providers.
- Do not start a public deployment unless both runtime generation and runtime embeddings are configured and their model/dimension identity is validated.
- In `cloud` profile, do not allow either runtime side to resolve to Ollama, even if Ollama variables are present; fail closed instead.
- Do not use an embedding API with an existing Ollama-built pgvector index without an explicit reindex.
- Do not make `0.0.0.0` the local default.
- Do not call external providers from unit tests or ordinary smoke tests.

## Current state

- `OllamaGenerator` calls Ollama `/api/chat` directly.
- `OllamaEmbedder` calls Ollama `/api/embed` directly.
- `api.serve()` constructs both Ollama classes directly.
- `Embedder` is already a protocol, so an API embedder can implement the existing retrieval/store contract.
- Generation currently has a concrete class rather than a shared protocol.
- `Settings` has Ollama and eval-judge settings but no runtime provider settings.
- The pgvector schema stores embedder model/backend/dimension in `index_meta` and rejects mismatches, which is the correct safety boundary to preserve.
- The API currently binds to loopback and port 8000; Render requires `0.0.0.0` and its `$PORT`.
- Uploads are stored under `data/uploads`, so production storage must be explicitly configured as a persistent disk or object storage in a later deployment step.

## Chosen approach: OpenAI-compatible runtime adapters

Add provider-neutral runtime protocols/factories and two HTTP adapters:

1. `OpenAICompatibleGenerator`
   - POST chat messages to `<chat_base_url>/chat/completions`.
   - Send model, messages, deterministic temperature/options, and bounded output settings.
   - Parse the standard `choices[0].message.content` response.
   - Sanitize errors and raise the existing generator-facing error type without exposing credentials.

2. `OpenAICompatibleEmbedder`
   - POST input texts to `<embedding_base_url>/embeddings`.
   - Send model and input list.
   - Parse `data[*].embedding`, preserve request order, normalize vectors using the existing canonical path, and discover/validate dimension.
   - Expose a stable backend/model identity for `index_meta`.

3. `OpenAICompatibleReranker` or a provider-specific API adapter
   - POST the query and candidate passages to a documented reranking endpoint.
   - Parse ranked candidate indices/scores, preserve chunk identity, and validate that every returned item maps to an input candidate.
   - Expose a stable provider/model/endpoint identity for deployment metadata.
   - The cloud profile requires this adapter so V2a remains the cloud default; it must not silently downgrade to `v2b_norerank`.


- deployment profile (`local` or `cloud`),
- local generation provider (`ollama` or `openai_compatible`),
- local embedding provider (`ollama` or `openai_compatible`),
- cloud runtime chat base URL, token, and model (required),
- cloud runtime embedding base URL, token, and model (required),
- optional provider organization/project headers only if required and never logged,
- public bind host and port with local-safe defaults.

The provider factory must enforce the profile contract: local may mix providers independently; cloud may use only API adapters for both runtime sides. A single provider may use the same base URL for both endpoints, but the configuration supports separate URLs and credentials so generation and embeddings can be hosted independently. The application startup factory must reject an incomplete API configuration before constructing the service.

## Render deployment shape

- Add a production entrypoint or command that invokes the FastAPI app with `--host 0.0.0.0 --port $PORT`.
- Keep local `api.serve()` defaults at `127.0.0.1:8000`.
- Add Docker/Render configuration only after runtime provider factories and health checks are tested.
- Render Postgres must provide pgvector. On first startup, schema creation must verify the extension and embedding metadata.
- Uploaded PDFs must use a configured persistent storage location or object storage; ephemeral filesystem storage is not acceptable as the sole production copy.
- The service must expose a lightweight health endpoint and a readiness path that can distinguish app process readiness from database/provider readiness without sending corpus data.
- Production auth, rate limits, CORS policy, upload quotas, and provider cost controls are deployment gates, not optional polish.

## Tests and verification

- Unit tests for API generator request/response parsing, error sanitization, and missing configuration.
- Unit tests for API embedder parsing, order preservation, normalization, dimension discovery, and error sanitization.
- Factory tests proving local mode still creates Ollama adapters and API mode creates only API adapters.
- Tests proving runtime settings never read `API_JUDGE_*` as a fallback.
- Tests proving an embedder identity/dimension mismatch fails before use.
- API startup tests for local loopback defaults and production host/port overrides.
- No tests call external providers; use mocked HTTP transport at the adapter boundary.
- Existing full suite, Ruff, diff check, and secret sweep must remain clean.
- Before deployment, run a provider-specific integration smoke only with an explicitly approved API endpoint/key and a non-sensitive test string, then reindex the approved corpus and rerun the relevant evaluation separately.

## Rollout and rollback

1. Implement adapters and settings behind explicit provider mode; keep Ollama defaults unchanged.
2. Validate all tests and local smoke behavior without external calls.
3. Configure a staging provider with a non-sensitive test corpus only after privacy/retention approval.
4. Reindex the production corpus into a fresh database or after a deliberate index reset; do not mutate the old Ollama index in place without backup and approval.
5. Deploy Render staging with API provider mode, Postgres/pgvector, persistent storage, and production host/port.
6. Verify health, upload, retrieval, generation, citations, and error redaction.
7. If any provider or storage check fails, roll back to local mode or stop deployment; do not fall back silently to a broken hybrid configuration.

## Open builder decisions

- Exact OpenAI-compatible endpoint/provider to use for production.
- Whether the chosen provider offers both chat and embeddings, or whether two providers are needed.
- Embedding model and vector dimension, which determine the reindex.
- Whether CPU reranking latency is acceptable on Render or whether the initial demo uses `v2b_norerank`.
- Persistent disk versus object storage for uploads.
- Authentication and rate-limit policy for the public service.

## Approval requested

Approve this design to proceed to the implementation plan. The implementation will not deploy publicly or call an external provider; it will first add tested, configurable adapters and Render readiness locally.
