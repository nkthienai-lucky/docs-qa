from __future__ import annotations

from dataclasses import dataclass

from .corpus import normalize_text
from .generation import ABSTAIN, build_messages, parse_answer
from .retrieval import Retriever
from .store import Hit

_EXCERPT_MAX = 200


def make_excerpt(text: str, max_chars: int = _EXCERPT_MAX) -> str:
    """First ~max_chars chars of the normalized text, cut at a word boundary."""
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    excerpt = normalized[:max_chars]
    # End at a word boundary: back off to the last space inside the window. A
    # window with no space (one very long token) keeps the hard cut rather
    # than shrinking to nothing.
    cut = excerpt.rfind(" ")
    if cut > 0:
        excerpt = excerpt[:cut]
    return excerpt


@dataclass(frozen=True)
class Citation:
    source: str
    page_start: int
    page_end: int
    section: str | None
    chunk_id: str
    excerpt: str


@dataclass(frozen=True)
class RetrievalInfo:
    configuration: str
    candidates: list[str]
    scores: list[float]


@dataclass(frozen=True)
class AnswerResponse:
    answer: str
    abstained: bool
    citations: list[Citation]
    retrieval: RetrievalInfo


def _citation(hit: Hit) -> Citation:
    return Citation(
        source=hit.chunk.doc_id,
        page_start=hit.chunk.page_start,
        page_end=hit.chunk.page_end,
        section=hit.chunk.section,
        chunk_id=hit.chunk.chunk_id,
        excerpt=make_excerpt(hit.chunk.text),
    )


def _abstain(configuration: str, hits: list[Hit] | None = None) -> AnswerResponse:
    # On a model abstention the hits were still retrieved and shown to the
    # generator, so they stay visible as candidates for debugging and eval.
    hits = hits or []
    return AnswerResponse(
        answer=ABSTAIN,
        abstained=True,
        citations=[],
        retrieval=RetrievalInfo(
            configuration,
            [hit.chunk.chunk_id for hit in hits],
            [hit.score for hit in hits],
        ),
    )


class Pipeline:
    """Grounded QA pipeline: retrieve, then generate, then resolve citations."""

    def __init__(
        self,
        retrievers: dict[str, Retriever],
        generator,
        final_k: int = 5,
        generation_k: int | None = None,
    ) -> None:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        if generation_k is not None and not 0 < generation_k <= final_k:
            raise ValueError("generation_k must be between 1 and final_k")
        self.retrievers = retrievers
        self.generator = generator
        self.final_k = final_k
        self.generation_k = generation_k or final_k

    def ask(
        self,
        question: str,
        config: str = "v2a",
        doc_ids: list[str] | None = None,
        hits: list[Hit] | None = None,
    ) -> AnswerResponse:
        if config not in self.retrievers:
            raise KeyError(f"unknown retrieval configuration: {config}")
        normalized_question = normalize_text(question)
        if not normalized_question:
            return _abstain(config)

        if hits is None:
            hits = self.retrievers[config].retrieve(normalized_question, doc_ids)[
                : self.final_k
            ]
        elif len(hits) > self.final_k:
            raise ValueError("provided hits cannot exceed final_k")
        if not hits:
            return _abstain(config)

        generation_hits = hits[: self.generation_k]
        messages = build_messages(normalized_question, generation_hits)
        raw = self.generator.generate(messages)
        parsed = parse_answer(raw, n_contexts=len(generation_hits))
        if parsed.abstained:
            return _abstain(config, hits)

        cited_indices = parsed.cited
        answer = parsed.text
        citations = [_citation(generation_hits[index - 1]) for index in cited_indices]
        return AnswerResponse(
            answer=answer,
            abstained=False,
            citations=citations,
            retrieval=RetrievalInfo(
                configuration=config,
                candidates=[hit.chunk.chunk_id for hit in hits],
                scores=[hit.score for hit in hits],
            ),
        )
