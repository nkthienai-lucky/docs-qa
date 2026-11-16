from __future__ import annotations

import unicodedata

from docsqa.chunking import Chunk
from docsqa.corpus import normalize_text
from docsqa.generation import ABSTAIN
from docsqa.pipeline import AnswerResponse, Pipeline
from docsqa.retrieval import VectorRetriever
from docsqa.store import Hit

DIM = 4


class FakeEmbedder:
    """Text-to-vector map the test controls explicitly."""

    model = "fake-embed"
    backend = "fake"
    dim = DIM
    query_instruction = "retrieve passages"

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.query_texts: list[str] = []

    def learn(self, text: str, vector: list[float]) -> None:
        self.vectors[text] = vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, q: str) -> list[float]:
        self.query_texts.append(q)
        return self.vectors[q]


class FakeStore:
    """In-memory store: hits are rows whose stored vector equals the query vector."""

    def __init__(self) -> None:
        self.rows: list[tuple[Chunk, list[float], float]] = []

    def index(self, chunk: Chunk, vector: list[float], score: float) -> None:
        self.rows.append((chunk, vector, score))

    def search_vector(
        self, qvec: list[float], k: int, doc_ids: list[str] | None = None
    ) -> list[Hit]:
        matching = [
            (score, chunk)
            for chunk, vector, score in self.rows
            if vector == qvec and (doc_ids is None or chunk.doc_id in doc_ids)
        ]
        matching.sort(key=lambda row: -row[0])
        return [Hit(chunk, score) for score, chunk in matching[:k]]


class FakeGenerator:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


def _chunk(ordinal: int, text: str, page: int = 1, section: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"doc:{ordinal}",
        doc_id="doc",
        text=text,
        page_start=page,
        page_end=page + 1,
        section=section,
        ordinal=ordinal,
    )


def _pipeline(
    store: FakeStore, embedder: FakeEmbedder, generator: FakeGenerator
) -> Pipeline:
    return Pipeline(
        {"v1": VectorRetriever(store, embedder, top_k=10)}, generator, final_k=5
    )


def test_citations_resolve_to_retrieved_chunks_and_excerpts() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    question = "what does the  vocabulary unit cover?"  # double space: must collapse
    qvec = [1.0, 0.0, 0.0, 0.0]
    embedder.learn("what does the vocabulary unit cover?", qvec)
    long_text = " ".join(f"word{index}" for index in range(60))  # > 200 chars
    texts = [
        long_text,
        "second chunk text about grammar",
        "third chunk",
        "fourth chunk",
        "fifth chunk",
        "sixth chunk",
        "seventh chunk",
    ]
    for ordinal, text in enumerate(texts):
        store.index(
            _chunk(ordinal, text, page=ordinal + 3, section="Unit 3" if ordinal == 1 else None),
            qvec,
            score=0.9 - 0.1 * ordinal,
        )
    generator = FakeGenerator("First claim [2]. Second claim [1].")
    pipeline = _pipeline(store, embedder, generator)

    response = pipeline.ask(question)

    assert isinstance(response, AnswerResponse)
    assert response.answer == "First claim [2]. Second claim [1]."
    assert response.abstained is False
    # final_k cuts the 7 matching rows to 5, ranked by score.
    assert response.retrieval.configuration == "v1"
    assert response.retrieval.candidates == [f"doc:{index}" for index in range(5)]
    assert response.retrieval.scores == sorted(
        response.retrieval.scores, reverse=True
    )
    # Citations follow the answer's citation order: [2] first, then [1].
    assert [citation.chunk_id for citation in response.citations] == ["doc:1", "doc:0"]
    first, second = response.citations
    assert (first.source, first.page_start, first.page_end, first.section) == (
        "doc",
        4,
        5,
        "Unit 3",
    )
    assert (second.source, second.page_start, second.page_end, second.section) == (
        "doc",
        3,
        4,
        None,
    )
    # Excerpt of a short chunk is the whole normalized text.
    assert first.excerpt == "second chunk text about grammar"
    # Excerpt of a long chunk is a prefix of at most 200 characters, cut
    # between words.
    normalized = normalize_text(long_text)
    assert len(second.excerpt) <= 200
    assert normalized.startswith(second.excerpt)
    assert normalized[len(second.excerpt) : len(second.excerpt) + 1] == " "
    # The generator saw exactly the 5 final hits and the normalized question.
    messages = generator.calls[0]
    user = messages[-1]["content"]
    assert user.count("<context id=") == 5
    assert "does the vocabulary unit" in user
    assert "the  vocabulary" not in user


def test_empty_index_returns_abstention_without_calling_generator() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    embedder.learn("any question at all", [0.0, 1.0, 0.0, 0.0])
    generator = FakeGenerator("should never be produced [1]")
    pipeline = _pipeline(store, embedder, generator)

    response = pipeline.ask("any question at all")

    assert response.abstained is True
    assert response.answer == ABSTAIN
    assert response.citations == []
    assert response.retrieval.candidates == []
    assert response.retrieval.scores == []
    assert generator.calls == []


def test_empty_or_whitespace_question_returns_abstention() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    generator = FakeGenerator("never [1]")
    pipeline = _pipeline(store, embedder, generator)

    for question in ("", "   ", " \n\t "):
        response = pipeline.ask(question)
        assert response.abstained is True
        assert response.answer == ABSTAIN
        assert response.citations == []

    assert generator.calls == []
    assert embedder.query_texts == []  # an empty question is never embedded


def test_diacritic_less_and_nfd_query_hit_same_chunk_as_nfc() -> None:
    store, embedder = FakeStore(), FakeEmbedder()
    vector = [0.5, 0.5, 0.5, 0.5]
    # The canonical embedding form of every query variant below, by hand.
    embedder.learn("cau hoi ve gdp cua viet nam", vector)
    store.index(
        _chunk(0, "Câu hỏi về GDP của Việt Nam", page=12, section="Kinh tế"),
        vector,
        score=0.9,
    )
    generator = FakeGenerator("Đáp án [1]")
    pipeline = _pipeline(store, embedder, generator)

    nfc = "câu hỏi về gdp của việt nam"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc  # the variants really differ
    bare = "cau hoi ve gdp cua viet nam"

    responses = [pipeline.ask(question) for question in (nfc, nfd, bare)]

    for response in responses:
        assert response.abstained is False
        assert response.retrieval.candidates == ["doc:0"]
        assert [citation.chunk_id for citation in response.citations] == ["doc:0"]
        assert response.citations[0].page_start == 12
        assert response.citations[0].section == "Kinh tế"
    # The generator sees the faithful NFC, whitespace-normalized question for
    # every variant — the accent-stripped form is used for embedding only.
    # NFD input is folded back to NFC; diacritic-less input is shown as typed
    # (its diacritics cannot be recovered).
    user_contents = [messages[-1]["content"] for messages in generator.calls]
    assert len(user_contents) == 3
    assert nfc in user_contents[0]
    assert nfc in user_contents[1]
    assert nfd not in user_contents[1]  # the NFD variant was folded to NFC
    assert bare in user_contents[2]
    assert all("cau hoi" not in content for content in user_contents[:2])
