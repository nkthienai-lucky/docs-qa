from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from ..generation import strip_citation_markers
from ..pipeline import Pipeline
from ..settings import Settings, get_settings
from .dataset import Question
from .judges import JudgeConfig, Sample, SampleScores, score_samples
from .metrics import RetrievalScores, retrieval_metrics
from .runlog import RunConfig, RunLog


@dataclass(frozen=True)
class SampleResult:
    question: Question
    config: str
    answer: str
    abstained: bool
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    retrieval: RetrievalScores | None
    contexts: list[str]


def save_sample_results(results: list[SampleResult], path: Path) -> None:
    """Persist generated evaluation outputs locally before any judge calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "question": asdict(result.question),
            "config": result.config,
            "answer": result.answer,
            "abstained": result.abstained,
            "retrieved_chunk_ids": result.retrieved_chunk_ids,
            "retrieval_scores": result.retrieval_scores,
            "retrieval": asdict(result.retrieval) if result.retrieval else None,
            "contexts": result.contexts,
        }
        for result in results
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_sample_results(path: Path) -> list[SampleResult]:
    """Load persisted answers to resume judge scoring without rerunning inference."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        SampleResult(
            question=Question(**item["question"]),
            config=item["config"],
            answer=item["answer"],
            abstained=item["abstained"],
            retrieved_chunk_ids=item["retrieved_chunk_ids"],
            retrieval_scores=item["retrieval_scores"],
            retrieval=RetrievalScores(**item["retrieval"]) if item["retrieval"] else None,
            contexts=item["contexts"],
        )
        for item in payload
    ]


def _config_values(
    configs: list[RunConfig] | Mapping[str, RunConfig],
) -> list[RunConfig]:
    return list(configs.values()) if isinstance(configs, Mapping) else list(configs)


def validate_comparable_configs(
    configs: list[RunConfig] | Mapping[str, RunConfig],
) -> None:
    """Ensure retrieval configurations share final top-k and prompt hash."""
    values = _config_values(configs)
    if not values:
        raise ValueError("at least one configuration is required")
    if len({config.name for config in values}) != len(values):
        raise ValueError("configuration names must be unique")
    if any(config.split != values[0].split for config in values):
        raise ValueError("comparable configurations must use the same split")
    for field in ("top_k", "generation_k", "prompt_hash"):
        if len({getattr(config, field) for config in values}) > 1:
            raise ValueError(f"configurations must share {field}")


def _run_config(
    config_name: str,
    questions: list[Question],
    pipeline: Pipeline,
    run_cfg: RunConfig,
    checkpoint_path: Path | None = None,
) -> list[SampleResult]:
    results: list[SampleResult] = []
    retriever = pipeline.retrievers[config_name]
    for question in questions:
        question_started = perf_counter()
        retrieval_started = perf_counter()
        hits = retriever.retrieve(question.text)[: pipeline.final_k]
        retrieval_s = perf_counter() - retrieval_started
        generation_started = perf_counter()
        response = pipeline.ask(question.text, config=config_name, hits=hits)
        generation_s = perf_counter() - generation_started
        retrieved_ids = list(response.retrieval.candidates)
        print(
            f"eval timing config={config_name} question={question.id} "
            f"retrieval_s={retrieval_s:.3f} generation_s={generation_s:.3f} "
            f"total_s={perf_counter() - question_started:.3f}",
            flush=True,
        )
        retrieved_scores = list(response.retrieval.scores)
        if len(hits) != len(retrieved_ids) or len(retrieved_scores) != len(
            retrieved_ids
        ):
            raise RuntimeError(
                "retrieval IDs, scores, and contexts must have matching lengths"
            )
        if [hit.chunk.chunk_id for hit in hits] != retrieved_ids:
            raise RuntimeError("retrieval snapshot does not match pipeline response")
        if [hit.score for hit in hits] != retrieved_scores:
            raise RuntimeError("retrieval scores do not match pipeline response")

        metric = None
        if question.category != "out_of_scope":
            metric = retrieval_metrics(
                [hit.chunk for hit in hits],
                question.gold_pages,
                question.gold_spans,
                pipeline.final_k,
            )
        results.append(
            SampleResult(
                question=question,
                config=config_name,
                answer=strip_citation_markers(response.answer),
                abstained=response.abstained,
                retrieved_chunk_ids=retrieved_ids,
                retrieval_scores=retrieved_scores,
                retrieval=metric,
                contexts=[
                    hit.chunk.text
                    for hit in hits[: getattr(pipeline, "generation_k", pipeline.final_k)]
                ],
            )
        )
        if checkpoint_path is not None:
            save_sample_results(results, checkpoint_path)
    return results


def run_config(
    config_name: str,
    questions: list[Question],
    pipeline: Pipeline,
    run_cfg: RunConfig,
    log: RunLog,
    checkpoint_path: Path | None = None,
) -> list[SampleResult]:
    if run_cfg.split == "held_out":
        raise ValueError(
            "held-out runs must use run_held_out_once for batch safeguards"
        )
    if run_cfg.split != "dev":
        raise ValueError("run config split must be dev or held_out")
    if config_name not in pipeline.retrievers:
        raise ValueError(f"unknown config: {config_name}")
    if run_cfg.name != config_name:
        raise ValueError("run config name must match requested configuration")
    if run_cfg.top_k != pipeline.final_k:
        raise ValueError("run config top_k must match pipeline final_k")
    if run_cfg.generation_k != getattr(pipeline, "generation_k", pipeline.final_k):
        raise ValueError("run config generation_k must match pipeline generation_k")
    if any(question.split != run_cfg.split for question in questions):
        raise ValueError("question split does not match run config")

    log.record(
        run_cfg,
        {
            "question_ids": [question.id for question in questions],
            "config": config_name,
            "final_k": pipeline.final_k,
            "generation_k": run_cfg.generation_k,
        },
    )
    return _run_config(config_name, questions, pipeline, run_cfg, checkpoint_path)


def run_configs(
    questions: list[Question],
    pipeline: Pipeline,
    configs: list[RunConfig] | Mapping[str, RunConfig],
    log: RunLog,
    checkpoint_dir: Path | None = None,
) -> dict[str, list[SampleResult]]:
    """Validate comparable configs and checkpoint dev results before judging."""
    values = _config_values(configs)
    validate_comparable_configs(values)
    if len({config.split for config in values}) != 1:
        raise ValueError("all configurations in a batch must use the same split")
    if not {config.name for config in values}.issubset(pipeline.retrievers):
        raise ValueError("all configured retrievers must exist in the pipeline")
    if any(config.top_k != pipeline.final_k for config in values):
        raise ValueError("run config top_k must match pipeline final_k")
    generation_k = getattr(pipeline, "generation_k", pipeline.final_k)
    if any(config.generation_k != generation_k for config in values):
        raise ValueError("run config generation_k must match pipeline generation_k")
    split = values[0].split
    if any(question.split != split for question in questions):
        raise ValueError("question split does not match run configs")

    if split == "held_out":
        expected = {"v1", "v2a", "v2b", "v2b_norerank"}
        if {config.name for config in values} != expected:
            raise ValueError(
                "held-out batch must include all four protocol configurations"
            )
        repeated = {config.name for config in values} & log.held_out_config_names()
        if repeated:
            raise ValueError(
                "held-out configurations already recorded: "
                + ", ".join(sorted(repeated))
            )
        if (
            log.held_out_ids
            and {question.id for question in questions} != log.held_out_ids
        ):
            raise ValueError("held-out questions do not match frozen split")
        # Consume every held-out configuration before the first retrieval. An
        # interrupted batch is therefore never eligible for a partial retry.
        log.record_batch(
            [
                (
                    config,
                    {
                        "question_ids": [question.id for question in questions],
                        "config": config.name,
                        "final_k": pipeline.final_k,
                        "generation_k": config.generation_k,
                    },
                )
                for config in values
            ]
        )
        results: dict[str, list[SampleResult]] = {}
        for config in values:
            checkpoint = (
                checkpoint_dir / f"{config.name}-{config.split}.json"
                if checkpoint_dir
                else None
            )
            results[config.name] = _run_config(
                config.name, questions, pipeline, config, checkpoint
            )
        return results

    return {
        config.name: run_config(
            config.name,
            questions,
            pipeline,
            config,
            log,
            checkpoint_dir / f"{config.name}-{config.split}-round{config.tuning_round}.json"
            if checkpoint_dir
            else None,
        )
        for config in values
    }


def run_held_out_once(
    questions: list[Question],
    pipeline: Pipeline,
    configs: list[RunConfig] | Mapping[str, RunConfig],
    log: RunLog,
    checkpoint_dir: Path | None = None,
) -> dict[str, list[SampleResult]]:
    values = _config_values(configs)
    expected = {"v1", "v2a", "v2b", "v2b_norerank"}
    if {config.name for config in values} != expected:
        raise ValueError("held-out batch must include all four protocol configurations")
    if any(config.split != "held_out" for config in values):
        raise ValueError("held-out batch configs must use held_out split")
    return run_configs(questions, pipeline, values, log, checkpoint_dir=checkpoint_dir)


def score_run(
    results: list[SampleResult],
    judges: list[JudgeConfig],
    settings: Settings | None = None,
) -> dict[str, list[SampleScores]]:
    settings = settings or get_settings()
    output: dict[str, list[SampleScores]] = {}
    for judge in judges:
        samples = [
            Sample(
                question=result.question.text,
                answer=result.answer,
                contexts=result.contexts,
                ground_truth=result.question.ground_truth,
            )
            for result in results
            if result.question.category != "out_of_scope"
        ]
        output[judge.name] = score_samples(samples, judge, settings)
    return output
