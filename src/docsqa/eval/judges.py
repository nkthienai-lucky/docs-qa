from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..embedding import OllamaEmbedder
from ..settings import Settings


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    kind: Literal["local", "api"]
    model: str
    temperature: float = 0.0


@dataclass(frozen=True)
class Sample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass(frozen=True)
class SampleScores:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


@dataclass(frozen=True)
class HumanLabel:
    sample_id: str
    faithful: bool


class JudgeError(RuntimeError):
    """A scoring failure with any possible credentials and endpoint removed."""


_JUDGE_CACHE_DIR = Path("eval/cache/judge")
_JUDGE_LOGGERS = ("httpx", "httpcore", "openai")


def _raise_http_loggers() -> None:
    for name in _JUDGE_LOGGERS:
        logger = logging.getLogger(name)
        if logger.level < logging.INFO:
            logger.setLevel(logging.INFO)


def make_ragas_llm(cfg: JudgeConfig, settings: Settings):
    """Build a Ragas 0.4 Instructor LLM over an OpenAI-compatible endpoint."""
    from openai import OpenAI
    from ragas.llms import llm_factory

    if cfg.kind == "local":
        base_url = settings.ollama_host.rstrip("/") + "/v1"
        api_key = "ollama-local"
    elif cfg.kind == "api":
        base_url = (
            settings.api_judge_base_url.get_secret_value()
            if settings.api_judge_base_url
            else None
        )
        api_key = (
            settings.api_judge_token.get_secret_value()
            if settings.api_judge_token
            else None
        )
        if not api_key:
            raise ValueError("API judge token is required")
        if not base_url:
            raise ValueError("API judge base URL is required")
        if not settings.api_judge_model:
            raise ValueError("API judge model is required")
    else:
        raise ValueError(f"unsupported judge kind: {cfg.kind}")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=900,
        max_retries=1,
    )
    return llm_factory(
        model=cfg.model,
        provider="openai",
        client=client,
        temperature=cfg.temperature,
        max_tokens=2048,
        extra_body={"think": False},
    )


def make_ragas_embeddings(settings: Settings):
    from ragas.embeddings.base import BaseRagasEmbedding

    class _ProjectRagasEmbedding(BaseRagasEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.embedder = OllamaEmbedder(
                host=settings.ollama_host,
                model=settings.embed_model,
                query_instruction="Given a web search query, retrieve relevant passages",
            )

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embedder.embed_documents([text])[0]

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text, **kwargs)

        def embed_query(self, text: str) -> list[float]:
            return self.embedder.embed_query(text)

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embedder.embed_documents(texts)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return _ProjectRagasEmbedding()


def _score_ragas(
    samples: list[Sample], cfg: JudgeConfig, settings: Settings
) -> list[dict[str, float | None]]:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        answer_relevancy as answer_relevancy_template,
    )
    from ragas.metrics import (
        context_precision as context_precision_template,
    )
    from ragas.metrics import (
        context_recall as context_recall_template,
    )
    from ragas.metrics import (
        faithfulness as faithfulness_template,
    )
    from ragas.run_config import RunConfig as RagasRunConfig

    llm = make_ragas_llm(cfg, settings)
    embeddings = make_ragas_embeddings(settings)
    metrics = [
        copy.deepcopy(faithfulness_template),
        copy.deepcopy(answer_relevancy_template),
        copy.deepcopy(context_precision_template),
        copy.deepcopy(context_recall_template),
    ]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "max_retries"):
            metric.max_retries = 0
    metrics[1].strictness = 1
    metrics[1].embeddings = embeddings
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample.question,
                response=sample.answer,
                retrieved_contexts=sample.contexts,
                reference=sample.ground_truth,
            )
            for sample in samples
        ]
    )
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RagasRunConfig(
            max_workers=1,
            timeout=900,
            max_retries=1,
        ),
        raise_exceptions=False,
        show_progress=False,
    )
    return [dict(row) for row in result.to_pandas().to_dict(orient="records")]


def _finite_score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_error_message(error: Exception, settings: Settings) -> str:
    message = str(error)
    forbidden = [
        settings.api_judge_token.get_secret_value() if settings.api_judge_token else "",
        settings.api_judge_base_url.get_secret_value()
        if settings.api_judge_base_url
        else "",
    ]
    for value in forbidden:
        if value:
            message = re.sub(
                re.escape(value), "[redacted]", message, flags=re.IGNORECASE
            )
    message = re.sub(
        r"https?://\S+", "[endpoint redacted]", message, flags=re.IGNORECASE
    )
    message = re.sub(
        r"(?i)\b(?:bearer\s+)?(?:token|api[_ -]?key)\s*[:=]?\s*\S+",
        "[credential redacted]",
        message,
    )
    return message[:500]


def _cache_path(samples: list[Sample], cfg: JudgeConfig, settings: Settings) -> Path:
    embedder_identity = hashlib.sha256(
        f"{settings.ollama_host}\0{settings.embed_model}".encode()
    ).hexdigest()
    payload = {
        "cache_schema": 3,
        "judge": {
            "name": cfg.name,
            "kind": cfg.kind,
            "model": cfg.model,
            "temperature": cfg.temperature,
        },
        "embedder_identity": embedder_identity,
        "samples": [
            {
                "question": sample.question,
                "answer": sample.answer,
                "contexts": sample.contexts,
                "ground_truth": sample.ground_truth,
            }
            for sample in samples
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return _JUDGE_CACHE_DIR / f"{digest}.json"


def score_samples(
    samples: list[Sample],
    cfg: JudgeConfig,
    settings: Settings,
) -> list[SampleScores]:
    _raise_http_loggers()
    if not samples:
        return []
    cache_path = _cache_path(samples, cfg, settings)
    if cache_path.exists():
        try:
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows = None
        if rows is not None:
            return [
                SampleScores(
                    faithfulness=_finite_score(row.get("faithfulness")),
                    answer_relevancy=_finite_score(row.get("answer_relevancy")),
                    context_precision=_finite_score(row.get("context_precision")),
                    context_recall=_finite_score(row.get("context_recall")),
                )
                for row in rows
            ]
    try:
        rows = _score_ragas(samples, cfg, settings)
    except Exception as error:  # noqa: BLE001 - provider SDKs expose heterogeneous exception classes
        raise JudgeError(
            f"judge scoring failed for {cfg.name}: {_safe_error_message(error, settings)}"
        ) from None

    scores = []
    for index in range(len(samples)):
        row = rows[index] if index < len(rows) else {}
        scores.append(
            SampleScores(
                faithfulness=_finite_score(row.get("faithfulness")),
                answer_relevancy=_finite_score(row.get("answer_relevancy")),
                context_precision=_finite_score(row.get("context_precision")),
                context_recall=_finite_score(row.get("context_recall")),
            )
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([score.__dict__ for score in scores], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return scores


def agreement_with_labels(
    scores: list[SampleScores], labels: list[HumanLabel]
) -> float:
    by_id = {label.sample_id: label.faithful for label in labels}
    if not by_id:
        raise ValueError("human labels must not be empty")
    if len(scores) != len(by_id):
        raise ValueError("scores and human labels must have equal lengths")
    pairs = list(zip(scores, by_id.values()))
    if any(
        score.faithfulness is None or not math.isfinite(score.faithfulness)
        for score, _ in pairs
    ):
        raise ValueError("faithfulness scores must be finite to calculate agreement")
    predicted = [score.faithfulness >= 0.5 for score, _ in pairs]
    actual = [faithful for _, faithful in pairs]
    return sum(left == right for left, right in zip(predicted, actual)) / len(actual)
