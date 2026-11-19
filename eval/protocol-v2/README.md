# Evaluation protocol v2

This is a versioned replacement for the contaminated original split. The original `eval/questions.jsonl`, `eval/split.json`, and `eval/runs.jsonl` remain unchanged. q02 was probed before formal scoring and is excluded; a newly authored held-out question q45 replaces it. Dev questions are unchanged.

- Questions: 44 (24 dev, 20 held-out; 18 in-scope items per split are scored by Ragas).
- Frozen split SHA-256: `369d4a6e7e6931133ab15ade8b96824f958e6cfc659148c329ec00c32494d616`.
- q45 is project-authored and not independently human-validated.
- Primary judge for the completed run: authorized external API judge through the local eval-only adapter; score-only checkpoints contain no credentials.
- Full rerun outputs: `rerun-gpu-2026-11-19/`; API scores: `scores-api/rerun-gpu-2026-11-19/`.
- Final report: `report-final-rerun-gpu-api/`.
- The first held-out attempt was interrupted. The final result is explicitly a full GPU/API rerun after interruption, not a pristine first-pass held-out benchmark.
- Runtime default selected from the completed held-out rerun: `v2a` (dense + reranking). `v1`, `v2b`, and `v2b_norerank` remain available as comparison configurations.
- Held-out corpus excerpts were sent only within the explicitly authorized evaluation scope; credentials were supplied through process environment and never written to repository artifacts.
