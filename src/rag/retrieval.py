"""Filter, retrieve, rank, and measure chunks independently of generation."""

from dataclasses import asdict, dataclass
import math
from time import perf_counter
from typing import Protocol

from rag.categories import CATEGORIES


class QueryCollection(Protocol):
    def query(self, **kwargs: object) -> dict: ...


@dataclass(frozen=True)
class RankedChunk:
    text: str
    source: str
    page: int
    category: str | None
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, not a probability of relevance."""
        return 1.0 - self.distance


@dataclass(frozen=True)
class RetrievalMetrics:
    category: str | None
    candidates_requested: int
    candidates_returned: int
    invalid_candidates: int
    selected_count: int
    unique_sources: int
    retrieval_ms: float
    selected_distances: tuple[float, ...]
    mean_similarity: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[RankedChunk, ...]
    metrics: RetrievalMetrics


def retrieve(
    collection: QueryCollection,
    question: str,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
    category: str | None = None,
) -> RetrievalResult:
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("candidate_k must be >= top_k > 0")
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    started = perf_counter()
    kwargs = dict(
        query_texts=[question.strip()], n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    if category is not None:
        kwargs["where"] = {"category": category}
    results = collection.query(**kwargs)

    def first(key: str) -> list:
        rows = results.get(key)
        return rows[0] if rows and rows[0] is not None else []

    documents, metadatas, distances = (first(key) for key in (
        "documents", "metadatas", "distances",
    ))
    candidates = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else None
        distance = distances[index] if index < len(distances) else None
        if not isinstance(document, str) or not document.strip():
            continue
        if not isinstance(metadata, dict):
            continue
        source, page = metadata.get("source"), metadata.get("page")
        if not isinstance(source, str) or not source or type(page) is not int or page < 1:
            continue
        if type(distance) not in (float, int) or not math.isfinite(distance):
            continue
        if category is not None and metadata.get("category") != category:
            continue
        candidates.append(RankedChunk(
            document.strip(), source, page, metadata.get("category"), float(distance),
        ))
    # Chroma normally returns this order already. Explicit sorting makes selection
    # deterministic and ensures malformed candidates are removed before top-k.
    candidates.sort(key=lambda chunk: chunk.distance)
    selected = tuple(candidates[:top_k])
    metrics = RetrievalMetrics(
        category=category, candidates_requested=candidate_k,
        candidates_returned=len(documents),
        invalid_candidates=len(documents) - len(candidates),
        selected_count=len(selected), unique_sources=len({c.source for c in selected}),
        retrieval_ms=round((perf_counter() - started) * 1000, 3),
        selected_distances=tuple(c.distance for c in selected),
        mean_similarity=(sum(c.similarity for c in selected) / len(selected)) if selected else None,
    )
    return RetrievalResult(selected, metrics)
