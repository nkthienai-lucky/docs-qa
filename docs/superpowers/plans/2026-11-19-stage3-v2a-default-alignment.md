# Stage 3 V2a Default Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the completed Stage 3 evaluation, runtime defaults, README, and roadmap annotation consistently identify V2a as the selected default, while preserving explicit V1/V2b compatibility and avoiding any new held-out benchmark run.

**Architecture:** Treat the final rerun report as the source of measured results and the runtime/API/UI configuration as the source of operational defaults. Update only documentation/metadata that is demonstrably stale, add regression tests at the settings/API/UI boundaries plus a report-consistency test, then run the existing suite, lint, diff checks, and a clean-checkout smoke verification without contacting external judges or rerunning held-out evaluation.

**Tech Stack:** Python 3.12, pytest, Ruff, uv, FastAPI, Gradio, Markdown/JSON evaluation artifacts, Git.

**Spec:** `AI-Engineer-Roadmap/docs/superpowers/specs/2026-11-13-docs-qa-stage-3-design.md` and the Stage 3 ledger at `.superpowers/sdd/2026-11-13-docs-qa-stage-3/progress.md`.

## Global Constraints

- Keep V2a (`dense candidates + cross-encoder reranking`) as the runtime/API/UI default; explicit `v1`, `v2b`, and `v2b_norerank` remain supported.
- Do not rerun the held-out benchmark, regenerate judge scores, or send corpus excerpts/credentials to any external service.
- Preserve the report caveat that the final held-out numbers came from a post-interruption GPU rerun, not a pristine first-pass benchmark.
- Report only measured values; retain limitations about project-authored questions, one English baseline corpus, judge/model bias, and contamination history.
- Keep loopback-only binding and local-only deployment waiver unchanged.
- Do not modify `.env` or expose secrets; only tracked documentation, tests, and source files may change.

## Review Focus

- A new API request without `config` must use V2a, while an explicit V1 request must still reach V1 — pin both in `tests/test_api.py`.
- A newly built UI must select V2a without removing the four comparison choices — pin default and choices in `tests/test_ui.py`.
- A fresh `Settings()` instance must expose the same V2a-compatible runtime defaults used by `serve()` — pin the settings values in `tests/test_settings.py`.
- The final report must not claim V1 is the default when the measured selection is V2a — add a report metadata/Markdown consistency test in `tests/eval/test_report.py`.
- The roadmap checklist must explain the local-deployment waiver and measured V2a selection instead of leaving stale unchecked deployment/default claims — verify exact annotation text in a documentation/security test.

---

### Task 1: Lock the V2a runtime default with regression tests

**Files:**
- Modify: `tests/test_api.py` (default `/ask` behavior and explicit V1 compatibility)
- Modify: `tests/test_ui.py` (default dropdown value and complete comparison choices)
- Modify: `tests/test_settings.py` (runtime default fields)
- Inspect only: `src/docsqa/api.py`, `src/docsqa/ui.py`, `src/docsqa/settings.py`

**Interfaces:**
- Consumes the existing API request model, `build_ui()`, and `Settings` fields.
- Produces executable regression coverage proving that V2a is the selected default without changing the public configuration names.

- [ ] **Step 1: Write/update failing regression assertions**
  - Assert `POST /ask` with only `question` forwards `config="v2a"`.
  - Assert an explicit `config="v1"` remains forwarded unchanged.
  - Assert the UI retrieval dropdown value is `v2a` and choices still contain exactly the supported labels/values for `v1`, `v2a`, `v2b`, and `v2b_norerank`.
  - Assert `Settings().generation_k == 1`, `Settings().reranker_device == "cpu"`, and the runtime context/prediction defaults match the documented local deployment.

- [ ] **Step 2: Run the focused tests to establish the expected failure if any stale boundary remains**

  Run: `uv run pytest tests/test_api.py tests/test_ui.py tests/test_settings.py -q`

  Expected: PASS if runtime changes are already correct; otherwise FAIL only on the stale boundary discovered by the assertions.

- [ ] **Step 3: Implement only the minimal boundary fixes**
  - Keep `config: str = "v2a"` in API/UI defaults and preserve explicit configuration forwarding.
  - Do not broaden accepted configs or alter pipeline scoring behavior.
  - Keep CPU reranking as the safe default and CUDA opt-in.

- [ ] **Step 4: Run the focused tests again**

  Run: `uv run pytest tests/test_api.py tests/test_ui.py tests/test_settings.py -q`

  Expected: all focused tests pass.

- [ ] **Step 5: Commit the runtime-default regression coverage**

  ```bash
  git add tests/test_api.py tests/test_ui.py tests/test_settings.py src/docsqa/api.py src/docsqa/ui.py src/docsqa/settings.py
  git commit -m "test: lock v2a runtime defaults"
  ```

### Task 2: Synchronize the final report with the selected default

**Files:**
- Modify: `eval/protocol-v2/report-final-rerun-gpu-api/report.md`
- Modify: `eval/protocol-v2/report-final-rerun-gpu-api/metadata.json` only if its schema has a stale default field
- Modify: `tests/eval/test_report.py`

**Interfaces:**
- Consumes the existing final rerun report, metadata, and report parser/renderer tests.
- Produces a report whose headline explicitly names V2a as the selected runtime/demo default and retains measured per-config values and caveats.

- [ ] **Step 1: Add a failing report consistency test**
  - Load the committed final report and assert its headline says `Default configuration: v2a` (or an equivalent unambiguous V2a statement).
  - Assert the report still contains all four configuration labels and the post-interruption rerun caveat.
  - Assert the final report does not contain a contradictory headline claiming V1 is the default.

- [ ] **Step 2: Run the report-focused test**

  Run: `uv run pytest tests/eval/test_report.py -q`

  Expected: FAIL before the stale headline is corrected, or PASS if an equivalent correction is already present.

- [ ] **Step 3: Update only stale report metadata/text**
  - Change the default headline to V2a.
  - Add one concise selection note tied to the held-out rerun values: V2a is the runtime default; V2b is retained for comparison; V1 remains explicitly supported.
  - Do not edit metric values, confidence intervals, NaN counts, per-question rows, or provenance caveats.

- [ ] **Step 4: Run the report-focused test again**

  Run: `uv run pytest tests/eval/test_report.py -q`

  Expected: all report tests pass.

- [ ] **Step 5: Commit the report correction**

  ```bash
  git add eval/protocol-v2/report-final-rerun-gpu-api/report.md eval/protocol-v2/report-final-rerun-gpu-api/metadata.json tests/eval/test_report.py
  git commit -m "docs: align final eval report with v2a default"
  ```

### Task 3: Complete the roadmap and README acceptance annotation

**Files:**
- Modify: `README.md`
- Modify: `/home/ad1/Code/AI-Engineer-LLM/AI-Engineer-Roadmap/docs/cores/03-RAG/07-project-docs-qa.md`
- Modify: `docs/corpus.md` only if the final README/roadmap links reveal a stale corpus-status statement
- Modify: `tests/test_security_sweep.py` or add a narrow documentation test under `tests/` if the existing sweep is the established place for tracked-document assertions

**Interfaces:**
- Consumes the selected V2a runtime behavior and final rerun report.
- Produces synchronized user-facing documentation: V2a default, explicit comparison configs, local deployment waiver, real measured report link, and limitations.

- [ ] **Step 1: Add failing documentation assertions**
  - Assert README names V2a as the runtime/API/UI default and links to the final report.
  - Assert roadmap Stage 3 annotation explicitly records the local-first deployment waiver and does not claim public deployment is complete.
  - Assert the roadmap annotation points to the measured protocol/report and preserves the threshold policy as a reference marker based on held-out confidence-interval lower bounds.

- [ ] **Step 2: Run the documentation-focused tests**

  Run: `uv run pytest tests/test_security_sweep.py -q`

  Expected: FAIL only if one of the required annotations is missing or stale.

- [ ] **Step 3: Update the documentation**
  - Keep the concise README status section as the operational summary.
  - Extend the roadmap's “Triển khai Stage 3” section with the selected V2a default, final rerun caveat, actual report path, deployment waiver, and known limitations.
  - Do not mark deployment as completed and do not replace measured results with roadmap example numbers.

- [ ] **Step 4: Run documentation tests and formatting checks**

  Run: `uv run pytest tests/test_security_sweep.py -q`

  Expected: all documentation/security tests pass.

- [ ] **Step 5: Commit the synchronized documentation**

  ```bash
  git add README.md docs/corpus.md tests/test_security_sweep.py /home/ad1/Code/AI-Engineer-LLM/AI-Engineer-Roadmap/docs/cores/03-RAG/07-project-docs-qa.md
  git commit -m "docs: annotate stage 3 v2a acceptance"
  ```

### Task 4: Verify the complete Stage 3 state without rerunning held-out evaluation

**Files:**
- Modify: `.superpowers/sdd/2026-11-13-docs-qa-stage-3/progress.md`
- Modify: `README.md` only if verification exposes a factual mismatch
- No new benchmark outputs

**Interfaces:**
- Consumes all previous tasks and the existing ignored/local evaluation outputs.
- Produces a verified ledger entry recording the exact commands and outcomes, including any clean-checkout limitation that remains.

- [ ] **Step 1: Run the full test suite**

  Run: `uv run pytest -q`

  Expected: all tests pass; record the exact count from the command output.

- [ ] **Step 2: Run static and diff verification**

  Run: `uv run ruff check src tests eval` and `git diff --check`

  Expected: Ruff exits 0 and diff check exits 0.

- [ ] **Step 3: Verify no held-out rerun or secret exposure occurred**
  - Confirm `git status --short` contains no `.env` changes and no newly tracked evaluation caches/results.
  - Confirm no command in the implementation diff invokes `eval/recover_stage3_dev.py` or a held-out runner.
  - Confirm committed report metadata contains no token, gateway URL, or settings dump.

- [ ] **Step 4: Perform a clean-checkout setup smoke test if the environment permits it**
  - Use a temporary copy/worktree outside the source tree, run `uv sync`, import checks, and the documentation/unit subset only.
  - If dependency services or the external corpus are unavailable, record the limitation rather than fabricating a clean-checkout success.

- [ ] **Step 5: Update the Stage 3 ledger**
  - Record V2a as the selected runtime default.
  - Record the exact final verification results and that no new held-out run was performed.
  - Keep the rerun-contamination caveat and remaining limitations.

- [ ] **Step 6: Commit the verification ledger**

  ```bash
  git add .superpowers/sdd/2026-11-13-docs-qa-stage-3/progress.md
  git commit -m "chore: record stage 3 v2a verification"
  ```

## Final verification checklist

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check src tests eval`
- [ ] `git diff --check`
- [ ] Report headline, README, runtime/API/UI, and roadmap all agree on V2a.
- [ ] Explicit `v1`, `v2b`, and `v2b_norerank` compatibility remains tested.
- [ ] No held-out benchmark is rerun.
- [ ] No secrets or external judge traffic are introduced.
- [ ] Clean-checkout status is reported honestly.
