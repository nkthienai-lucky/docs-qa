# Task 7 report — FastAPI and upload safety

## Implemented

- Added loopback-only FastAPI service with `/health`, `/documents` GET/POST, and `/ask`.
- Enforced PDF magic bytes and 30,000,000-byte cap while reading no more than cap + 1 bytes.
- Stores uploads by content fingerprint as `{doc_id}.pdf`; client paths are reduced to a sanitized, display-only basename, bounded to 255 chars and given a `.pdf` extension.
- Preserves the original source metadata on duplicate uploads and returns the indexed chunk count.
- Enforces 300-page parser default, 900-second parse budget, and 5000-chunk cap before indexing; removes stored files after ingest failures.

## Verification

- Focused tests (API, upload, BM25, RRF, rerank, retrievers): 40 passed, 1 Starlette deprecation warning.
- Full suite after final filename sanitizer and cache test changes: 80 passed, 1 skipped, 1 warning in 386.72s.
- `git diff --check` was clean before the last small sanitizer change.

## Review status

Subagent review could not start because the parent checkout's Git worktree redirect prevents safe isolated worktrees. Implementer performed a direct source/test review. No subagent approval is claimed.
