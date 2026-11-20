# Docs-QA

A local, handwritten document-question-answering project with page-aware citations. It compares dense retrieval (V1), dense + reranking (V2a), hybrid BM25/vector + reranking (V2b), and hybrid without reranking. The web app does not import the eval-only judge adapters.

## Status

- Stage 3 implementation, evaluation, security sweep, and UI are complete for the current local deployment scope.
- Runtime/API/UI default retrieval is **V2a (dense candidates + cross-encoder reranking)**, predeclared from dev tuning with the primary judge before held-out execution. Explicit `config="v1"`, `v2b`, or `v2b_norerank` remains supported for comparison and compatibility.
- The final evaluation report is [eval/protocol-v2/report-final-rerun-gpu-api/](eval/protocol-v2/report-final-rerun-gpu-api/). It is a full rerun after an interrupted held-out attempt, not a pristine first-pass held-out run; this limitation is recorded in the report and ledger.
- Stage 3 verification: full GPU rerun 80/80 results, API judge 72/72 in-scope samples, `166 passed`, Ruff clean, and `git diff --check` clean.
- Runtime deployment now targets the VPS with Docker Compose: see [VPS operations](docs/deployment-vps.md). The CPU image is built and the private database is healthy; API startup awaits runtime provider configuration. No public production deployment is claimed.

## Architecture

```text
PDF upload -> validate -> Docling parse -> page-aware chunks -> Ollama embeddings -> PostgreSQL/pgvector
                                                                                  |
Question -> dense vector retrieval ----------------------------------------------+
         -> BM25 -> reciprocal-rank fusion -> optional cross-encoder (V2a is the runtime default; CPU is the safe default, CUDA is opt-in after VRAM checks)
         -> grounded prompt (top-1 reranked passage by default) -> Ollama generator -> answer + source/page/excerpt citations

Evaluation package (separate from API/UI): frozen questions -> config runner -> retrieval/Ragas metrics -> reports
```

## Local setup

Requirements: Python 3.12, `uv`, Docker with the pgvector image, and Ollama. The default DB is `roadmap_pg` on localhost with database `docsqa`; integration tests use `docsqa_test`. Ensure Postgres and Ollama are bound to `127.0.0.1` and are not exposed to an untrusted network.

```bash
uv sync
cp .env.example .env  # set only local values; never commit .env
# Start the project's pgvector container using the locally approved Docker setup.
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
ollama serve
uv run pytest -q
uv run docsqa-ui
```

The API can be started with `uv run python -c 'from docsqa.api import serve; serve()'`; it binds to `127.0.0.1:8000`. Gradio binds to `127.0.0.1:7860`. Never change these bind addresses to `0.0.0.0` without an explicit security review and authenticated deployment design.

## Evaluation protocol

- `eval/questions.jsonl` is the authored 44-question set; 20 questions are held out and 8 are out-of-scope. `eval/split.json` freezes the split IDs and SHA-256 before retrieval/scoring.
- Run `uv run pytest tests/eval -q` to validate eval code. The evaluation runner is programmatic; see `src/docsqa/eval/runner.py` and `src/docsqa/eval/report.py`.
- Tuning uses dev only, at most three rounds; configurations must share final `top_k` and prompt hash. Held-out must only be run once at the end and requires all four protocol configurations.
- The local judge uses Ollama. The API judge is optional and eval-only. Do not send corpus excerpts to a third-party gateway until the corpus license and gateway retention/privacy terms are explicitly confirmed. Credentials must remain in local environment settings, never logs or result metadata.
- Report only measured values. The reference markers are faithfulness > 0.8 and context recall > 0.7; they are not claims that this project has achieved them.

## Citation and safety contract

Answers must be grounded in retrieved context and cite `[N]` markers that resolve to source, PDF page range, section, chunk ID, and a plain-text excerpt. If retrieval is empty or evidence is insufficient, the system abstains with the fixed Vietnamese sentence. Retrieved document text is treated as untrusted data, not instructions. Uploaded filenames are display metadata only; bytes are stored under a content-derived ID with upload size/type limits.

## Corpus and limitations

The baseline is the local English Vocabulary in Use Pre-Intermediate PDF (fingerprint prefix `3a357be041df101c`), with 203 chunks at 512 tokens. The PDF is user-provided and intentionally not included in Git. See [docs/corpus.md](docs/corpus.md) for extraction details, known parsing limitations, and the outstanding license/data-egress confirmation. The approved Vietnamese Math PDF currently fails the native-text quality gate and is not the evaluation baseline.

Current limitations include a single baseline document, a single locally authored evaluation set, no independent annotation round, and judge/model bias. Do not present these results as general Vietnamese RAG performance. No made-up scores or deployment claims belong in the results section.

## Security notes

- `.env` and evaluation run outputs are local-only; review `git status` before committing.
- Run metadata is allow-listed; API tokens, gateway URLs/headers, and settings dumps are prohibited from reports.
- The API judge must never be imported from the web app; the package has an import-hygiene test.
- Keep PostgreSQL and Ollama bound to loopback. Do not upload arbitrary user documents to an external judge.

## Runtime API providers and VPS deployment

The selected deployment target is the **VPS**, using `compose.vps.yaml`; see
[setup, rollout status, private access and recovery](docs/deployment-vps.md).
Runtime infrastructure is **not deployment authorization**. The Render template
is retained as an alternative; no public deployment or real-provider smoke is certified. V2a still
means dense retrieval plus reranking (external reranking in cloud); all four
retrieval modes and top-1 generation context remain available. UI remains local:
`DOCSQA_API_BASE` points the loopback Gradio client at the separately approved API.

Start the API with `uv run docsqa-api` (local default `127.0.0.1:8000`), or use
`uv run uvicorn docsqa.serve:create_runtime_app --factory --host 127.0.0.1 --port 8000`

for local ASGI integration. Use the CLI/container entrypoint in deployment so the
validated bind is used; direct ASGI server bind flags are the operator's responsibility.
Local `HOST` must be loopback; `PORT` defaults to 8000. Each local provider selector
is independent. Cloud requires three explicit API selectors, HTTPS endpoints,
complete separate credentials/models and an existing absolute upload directory.
It requires assigned `PORT` and explicit `RUNTIME_DEPLOYMENT_APPROVED=true` at the
entrypoint. The Render template deliberately sets this to false and relies on
Render's assigned `PORT`, not a hard-coded port.

See `.env.example` for the complete independent provider variables:
- `RUNTIME_CHAT_BASE_URL`, `RUNTIME_CHAT_TOKEN`, `RUNTIME_CHAT_MODEL`
- `RUNTIME_EMBEDDING_BASE_URL`, `RUNTIME_EMBEDDING_TOKEN`, `RUNTIME_EMBEDDING_MODEL`
- `RUNTIME_RERANK_BASE_URL`, `RUNTIME_RERANK_TOKEN`, `RUNTIME_RERANK_MODEL`
- `RUNTIME_PROFILE=cloud`, `RUNTIME_GENERATION_PROVIDER=openai_compatible`,
  `RUNTIME_EMBEDDING_PROVIDER=openai_compatible`, `RUNTIME_RERANK_PROVIDER=api`
- `RUNTIME_UPLOAD_DIR=/var/data/uploads`, `RUNTIME_API_TIMEOUT_S=30`
- `RUNTIME_DEPLOYMENT_APPROVED=false` until separate approval



Runtime never uses `API_JUDGE_*`; those remain eval-only. Empty
`RUNTIME_EMBEDDING_QUERY_INSTRUCTION` disables the instruction wrapper; default is
`Given a web search query, retrieve relevant passages`. Document canonicalization
is unchanged. Provider requests use bounded 8 MiB JSON responses, no redirects,
no retries/fallback, and a finite socket timeout (not a hard total deadline).
Root hosts gain `/v1`; custom prefixes are preserved; recognized terminal
`chat/completions`, `embeddings`, and `rerank` suffixes are replaced as needed.
Reranking supports one Cohere-style `/v1/rerank` **full-list** contract:
`model/query/documents/top_n=len(candidates)` and complete unique
`results[index,relevance_score]`, sorted locally. A provider returning only a
partial top-N list is rejected. Confirm the selected provider's wire shape before
production. Generation uses OpenAI-compatible chat completions; embeddings require
indexed responses and stable dimension. Never silently switch to Ollama or remove
reranking when a provider fails.

### Fresh index and storage safety

Runtime persists only a versioned hashed embedding identity, not raw provider URLs
or tokens. Endpoint/model/dimension/instruction/preprocessing changes require a
fresh compatible index. **All legacy indexes (even empty ones) require explicit
fresh-index/rebuild setup for runtime**; legacy eval callers continue to use legacy
metadata but cannot open a runtime-identified index. Missing metadata over existing
chunks fails closed. No automatic adoption, reset or migration is performed.

Before first runtime startup, back up the existing DB and uploads together. With
separate operator approval, provision a **new empty database/schema** with pgvector
and configure `DATABASE_URL` for it; retain the old DB unchanged. The runtime will
create its schema atomically after the synthetic embedding dimension probe. Reupload
only approved corpus files into that fresh index. Do not point at `docsqa_test`
except during tests. Never delete or overwrite the old index as a startup fix.

Render preparation is single-instance, disk mounted at `/var/data`, uploads at
`/var/data/uploads`. The startup script requires an actual mount before creating
the upload subdirectory. App validation checks existing writable non-symlink cloud
paths, but directory writeability alone does not prove persistence. Pair disk and
Postgres backup/restore procedures. Multi-instance/shared object storage is future
work. Docker retains Docling/ML dependencies: this is **not** a lightweight-image
claim. PDF parsing still runs locally and may need model assets/network, substantial
CPU/RAM and a resource smoke before deployment; no build-time model downloads or
provider secrets are configured. Keep the existing upload/page/chunk/parse-time
limits and assess parser sandboxing for untrusted PDFs.

`/health` is process-only. `/ready` reports validated configuration/index and a
bounded DB check (runtime connect 3s, statement 2s); providers are labelled
`configured`, not live/healthy. Startup probes embeddings only; generation and
reranking availability is unverified until a separately approved smoke.

### Separate network/cost/privacy gate

`uv run python scripts/runtime_api_smoke.py` skips with **zero provider calls**.
Only after explicit endpoint/privacy/cost approval, an operator may enable
`--allow-network` with complete cloud settings in process environment (the script
never reads `.env`). It uses synthetic generation/embedding/rerank inputs and
reports counts, not provider output. The flag itself is not approval. Do not use
held-out benchmark results as a runtime smoke, and do not rerun held-out here.

Before public deployment, implement/review **auth**, **rate limits**, CORS policy,
quotas, provider budget and data-retention terms, pgvector compatibility, persistent
mount/backups, parser assets/resources and network egress. These public-security
controls are **not implemented** by the deployment boolean. Questions and retrieved
passages leave the machine for generation/reranking; document chunks leave for
embedding. Obtain corpus licensing, privacy and retention approval before egress.
