"""Synthetic runtime checks. The flag is not privacy/cost approval."""

from __future__ import annotations

import argparse
import math

from docsqa.chunking import Chunk
from docsqa.runtime import build_runtime_components, validate_runtime_profile
from docsqa.settings import RuntimeProfile, Settings
from docsqa.store import Hit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_network:
        print("Runtime API smoke skipped (network not enabled).")
        return 0
    components = None
    try:
        settings = Settings(_env_file=None)
        if settings.runtime_profile != RuntimeProfile.CLOUD:
            raise ValueError
        if not settings.runtime_deployment_approved:
            raise ValueError
        validate_runtime_profile(settings)
        components = build_runtime_components(settings)
    except Exception:  # noqa: BLE001, S110 — never expose configuration/provider diagnostics
        pass
    if components is None:
        print("Runtime API smoke failed: configuration unavailable.")
        return 1
    passed = 0
    # Only synthetic text; never reads documents, a DB, or local .env files.
    checks = [
        lambda: (
            components.generator.generate(
                [
                    {
                        "role": "user",
                        "content": "Reply with exactly SYNTHETIC_OK and nothing else.",
                    }
                ]
            ).strip()
            == "SYNTHETIC_OK"
        ),
        lambda: _embedding_check(components.embedder),
        lambda: _rerank_check(components.reranker),
    ]
    for check in checks:
        try:
            if check():
                passed += 1
        except Exception:  # noqa: BLE001, S110 — no raw upstream output
            pass
    failed = len(checks) - passed
    print(f"Runtime API smoke: {passed} passed, {failed} failed.")
    return 1 if failed else 0


def _embedding_check(embedder) -> bool:
    vectors = embedder.embed_documents(
        ["A synthetic blue triangle.", "A synthetic red square."]
    )
    return (
        len(vectors) == 2
        and len(vectors[0]) > 0
        and len(vectors[0]) == len(vectors[1])
        and all(
            all(math.isfinite(x) for x in v)
            and math.isclose(math.hypot(*v), 1, rel_tol=1e-6)
            for v in vectors
        )
    )


def _rerank_check(reranker) -> bool:
    hits = [
        Hit(Chunk(str(i), "synthetic", text, 1, 1, None, i), 0)
        for i, text in enumerate(
            ["A synthetic red square.", "A synthetic blue triangle."]
        )
    ]
    ranked = reranker.rerank("Which passage mentions a blue triangle?", hits, 2)
    return (
        len(ranked) == 2
        and ranked[0].chunk is hits[1].chunk
        and ranked[1].chunk is hits[0].chunk
    )


if __name__ == "__main__":
    raise SystemExit(main())
