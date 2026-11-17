from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import SecretStr

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
    """Build a Ragas-compatible LangChain wrapper from configured judge settings."""
    from ragas.llms import LangchainLLMWrapper

    if cfg.kind == "local":
        from langchain_community.chat_models import ChatOllama

        llm = ChatOllama(
            base_url=settings.ollama_host,
            model=cfg.model,
            temperature=cfg.temperature,
            num_ctx=settings.num_ctx,
            format="json",
            model_kwargs={"think": False},
        )
    elif cfg.kind == "api":
        base_url = settings.api_judge_base_url.get_secret_value() if settings.api_judge_base_url else None
        token = settings.api_judge_token.get_secret_value() if settings.api_judge_token else None
        if not token:
            raise ValueError("API judge token is required")
        if not base_url:
            raise ValueError("API judge base URL is required")
        if not settings.api_judge_model:
            raise ValueError("API judge model is required")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr(token),
            model=settings.api_judge_model,
            temperature=cfg.temperature,
            timeout=900,
            max_retries=1,
        )
    else:
        raise ValueError(f"unsupported judge kind: {cfg.kind}")
    return LangchainLLMWrapper(llm)


def make_ragas_embeddings(settings: Settings):
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class _ProjectEmbedderAdapter(Embeddings):
        def __init__(self) -> None:
            self.embedder = OllamaEmbedder(
                host=settings.ollama_host,
                model=settings.embed_model,
                query_instruction="Given a web search query, retrieve relevant passages",
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embedder.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return self.embedder.embed_query(text)

    return LangchainEmbeddingsWrapper(_ProjectEmbedderAdapter())


def _score_ragas(
    samples: list[Sample], cfg: JudgeConfig, settings: Settings
) -> list[dict[str, float | None]]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.run_config import RunConfig as RagasRunConfig

    llm = make_ragas_llm(cfg, settings)
    embeddings = make_ragas_embeddings(settings)
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    dataset = Dataset.from_list(
        [
            {
                "user_input": sample.question,
                "response": sample.answer,
                "retrieved_contexts": sample.contexts,
                "reference": sample.ground_truth,
            }
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
        settings.api_judge_token.get_secret_value()
        if settings.api_judge_token
        else "",
        settings.api_judge_base_url.get_secret_value()
        if settings.api_judge_base_url
        else "",
    ]
    for value in forbidden:
        if value:
            message = re.sub(re.escape(value), "[redacted]", message, flags=re.IGNORECASE)
    message = re.sub(r"https?://\S+", "[endpoint redacted]", message, flags=re.IGNORECASE)
    message = re.sub(
        r"(?i)\b(?:bearer\s+)?(?:token|api[_ -]?key)\s*[:=]?\s*\S+",
        "[credential redacted]",
        message,
    )
    return message[:500]


def _cache_path(samples: list[Sample], cfg: JudgeConfig) -> Path:
    payload = {
        "judge": {"name": cfg.name, "kind": cfg.kind, "model": cfg.model, "temperature": cfg.temperature},
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
    cache_path = _cache_path(samples, cfg)
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
    except Exception as error:
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
    if any(score.faithfulness is None or not math.isfinite(score.faithfulness) for score, _ in pairs):
        raise ValueError("faithfulness scores must be finite to calculate agreement")
    predicted = [score.faithfulness >= 0.5 for score, _ in pairs]
    actual = [faithful for _, faithful in pairs]
    return sum(left == right for left, right in zip(predicted, actual)) / len(actual)
