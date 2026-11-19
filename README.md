# Docs-QA

A local, handwritten document-question-answering project with page-aware citations. It compares dense retrieval (V1), dense + reranking (V2a), hybrid BM25/vector + reranking (V2b), and hybrid without reranking. The web app does not import the eval-only judge adapters.

## Status

- Stage 3 implementation, evaluation, security sweep, and UI are complete for the current local deployment scope.
- Runtime/API/UI default retrieval is **V2a (dense candidates + cross-encoder reranking)**, selected from the completed protocol-v2 GPU/API rerun. Explicit `config="v1"`, `v2b`, or `v2b_norerank` remains supported for comparison and compatibility.
- The final evaluation report is [eval/protocol-v2/report-final-rerun-gpu-api/](eval/protocol-v2/report-final-rerun-gpu-api/). It is a full rerun after an interrupted held-out attempt, not a pristine first-pass held-out run; this limitation is recorded in the report and ledger.
- Stage 3 verification: full GPU rerun 80/80 results, API judge 72/72 in-scope samples, `160 passed`, Ruff clean, and `git diff --check` clean.
- No deployment is configured; services are intended for loopback-only use.

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
- Report only measured values. The reference markers are faithfulness 0.8 and context recall 0.7; they are not claims that this project has achieved them.

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
