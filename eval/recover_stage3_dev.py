from __future__ import annotations

import json
from pathlib import Path

from docsqa.eval.dataset import load_questions
from docsqa.eval.runner import SampleResult
from docsqa.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]
LOG = Path("/tmp/claude-1000/-home-ad1-Code-AI-Engineer-LLM/45fad9df-6414-438f-81a2-6845a4ca47f1/tasks/bxp9idlxo.output")
TIMING: dict[tuple[str, str], dict[str, str]] = {}
for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.startswith("eval timing config="):
        continue
    fields = dict(piece.split("=", 1) for piece in line.split() if "=" in piece)
    TIMING[(fields["config"], fields["question"])] = fields


def load_existing_results(config: str, questions) -> list[SampleResult]:
    """Rebuild non-answer dev evidence from stable corpus + run settings.

    This recovery deliberately does not call the generator or ask it to reproduce
    prior answers. Since the original process failed before saving answers,
    generated answer quality and abstention labels remain unavailable.
    """
    from docsqa.bm25 import BM25Index
    from docsqa.embedding import OllamaEmbedder
    from docsqa.retrieval import HybridRetriever, VectorRetriever
    from docsqa.store import EmbedderMeta, VectorStore

    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_host, settings.embed_model, "Given a web search query, retrieve relevant passages")
    embedder.embed_documents(["recovery probe"])
    store = VectorStore(settings.database_url, EmbedderMeta(embedder.model, embedder.backend, embedder.dim))
    try:
        vector = VectorRetriever(store, embedder, top_k=50)
        bm25 = BM25Index.from_store(store)
        if config == "v2b":
            retriever = HybridRetriever(vector, bm25, store, reranker=None, candidates=50, top_k=5)
            from docsqa.rerank import Reranker
            reranker = Reranker(settings.reranker_model, device="cuda", batch_size=2)
            retriever.reranker = reranker
        else:
            retriever = HybridRetriever(vector, bm25, store, reranker=None, candidates=50, top_k=5)
        output: list[SampleResult] = []
        for question in questions:
            hits = retriever.retrieve(question.text)[:5]
            metric = None
            if question.category != "out_of_scope":
                from docsqa.eval.metrics import retrieval_metrics
                metric = retrieval_metrics([h.chunk for h in hits], question.gold_pages, question.gold_spans, 5)
            output.append(SampleResult(
                question=question,
                config=config,
                answer="[answer unavailable: previous process did not checkpoint generated text]",
                abstained=False,
                retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
                retrieval_scores=[h.score for h in hits],
                retrieval=metric,
                contexts=[h.chunk.text for h in hits[:1]],
            ))
        return output
    finally:
        store.close()


def main() -> None:
    questions = [q for q in load_questions(ROOT / "eval/questions.jsonl") if q.split == "dev"]
    combined: list[SampleResult] = []
    for name in ("v2b", "v2b_norerank"):
        if len([key for key in TIMING if key[0] == name]) != len(questions):
            raise RuntimeError(f"timing audit incomplete for {name}")
        rows = load_existing_results(name, questions)
        # Ragas scores over explicit placeholder answers are prohibited.
        combined.extend(rows)
    retrieval_path = ROOT / "eval/recovery/retrieval-only.json"
    retrieval_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_path.write_text(json.dumps({
        "status": "partial_recovery_no_generated_answers",
        "run_log_configs": ["v2b", "v2b_norerank"],
        "timing_samples": len(TIMING),
        "metrics": {name: [row.retrieval.__dict__ if row.retrieval else None for row in combined if row.config == name] for name in ("v2b", "v2b_norerank")},
        "generated_answers": "unavailable; original process failed before persistence",
        "held_out": "not run",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved retrieval-only recovery to {retrieval_path}")
    print("No answer/Ragas scoring is emitted because original generated answers were not persisted.")


if __name__ == "__main__":
    main()
