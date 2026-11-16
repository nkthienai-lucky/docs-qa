# Task 9 report — BM25, RRF, reranker and V2 retrievers

## Implemented

- Added Vietnamese/English whitespace tokenizer with NFC/lowercase, accented and accent-stripped unigrams, plus accent-stripped adjacent bigrams.
- Added BM25Okapi index, filtered search, store-backed cached index refresh on document-set/count changes, and reciprocal-rank fusion.
- Added CPU-default CrossEncoder reranker and V2a dense+rerank, V2b dense/BM25+RRF+rerank, and V2b_norerank pipelines in API startup.
- Added defensive `doc_ids` filtering for both vector and BM25 results and for store-resolved chunks.

## Verification

- CPU CrossEncoder loader smoke succeeded for `BAAI/bge-reranker-v2-m3`.
- Real corpus smoke on `3a357be041df101c` (203 chunks) exercised v2a, v2b, and v2b_norerank on two queries; outputs and quality caveat are in `docs/v2-smoke.md`.
- Focused suite including Task 7/9 tests: 40 passed, 1 Starlette deprecation warning.
- Full suite after final changes: 80 passed, 1 skipped, 1 warning in 386.72s.

## Review status

Subagent review could not start because the parent checkout's Git worktree redirect prevents safe isolated worktrees. Implementer performed direct source/test review; no subagent approval is claimed.
