"""Immutable contracts for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from raglab.contracts import Citation

type JsonScalar = str | int | float | bool | None
type FilterValue = JsonScalar | tuple[JsonScalar, ...]


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """A typed filter over document/chunk metadata or an allow-listed document field."""

    field: str
    operator: str
    value: FilterValue

    def __post_init__(self) -> None:
        parts = self.field.split(".")
        metadata_path = (
            len(parts) >= 3
            and parts[0] in {"document", "chunk"}
            and parts[1] == "metadata"
            and all(parts[2:])
        )
        document_field = self.field in {
            "document.source_uri",
            "document.source_name",
            "document.title",
            "document.media_type",
        }
        if not metadata_path and not document_field:
            raise ValueError(f"Unsupported filter field: {self.field!r}")
        if self.operator not in {"eq", "ne", "in", "contains"}:
            raise ValueError(f"Unsupported filter operator: {self.operator!r}")
        if self.operator == "in" and not isinstance(self.value, tuple):
            raise ValueError("The 'in' operator requires a tuple value")
        if self.operator != "in" and isinstance(self.value, tuple):
            raise ValueError(f"The {self.operator!r} operator requires a scalar value")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_k: int = 50
    top_k: int = 5
    rrf_k: int = 60
    semantic_weight: float = 1.0
    bm25_weight: float = 1.0
    ef_search: int = 100
    exact: bool = False
    rewrite: bool = False
    expansions: int = 0
    rerank: bool = True
    mmr: bool = True
    mmr_lambda: float = 0.7
    small_to_big: bool = True
    parent_max_tokens: int = 1500

    def __post_init__(self) -> None:
        if self.candidate_k < 1 or self.top_k < 1 or self.rrf_k < 1:
            raise ValueError("candidate_k, top_k, and rrf_k must be positive")
        if self.ef_search < 1 or self.parent_max_tokens < 1:
            raise ValueError("ef_search and parent_max_tokens must be positive")
        if self.semantic_weight < 0 or self.bm25_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.semantic_weight == self.bm25_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        if not 0 <= self.expansions <= 2:
            raise ValueError("expansions must be between zero and two")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between zero and one")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    collection: str = "documents"
    filters: tuple[MetadataFilter, ...] = ()
    history: tuple[str, ...] = ()
    config: RetrievalConfig = field(default_factory=RetrievalConfig)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if not self.collection.strip():
            raise ValueError("collection cannot be empty")


@dataclass(frozen=True, slots=True)
class QueryRewrite:
    standalone_query: str
    expansions: tuple[str, ...] = ()


class QueryRewriter(Protocol):
    def rewrite(
        self, query: str, history: Sequence[str], *, max_expansions: int
    ) -> QueryRewrite: ...


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class QueryEmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    heading_path: tuple[str, ...]
    embedding: tuple[float, ...]
    citation: Citation
    document_metadata: Mapping[str, Any] = field(default_factory=dict)
    chunk_metadata: Mapping[str, Any] = field(default_factory=dict)
    ann_distance: float | None = None
    bm25_score: float | None = None


@dataclass(frozen=True, slots=True)
class NeighborChunk:
    chunk: RetrievedChunk
    matches_filters: bool


@dataclass(frozen=True, slots=True)
class RankingTrace:
    ann_rank: int | None
    bm25_rank: int | None
    ann_distance: float | None
    bm25_score: float | None
    rrf_score: float
    reranker_score: float | None
    mmr_score: float | None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    id: str
    document_id: str
    content: str
    citation: Citation
    matched_chunk_ids: tuple[str, ...]
    first_chunk_index: int
    last_chunk_index: int
    trace: RankingTrace


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    query: str
    rewritten_query: str | None
    query_variants: tuple[str, ...]
    filters: tuple[MetadataFilter, ...]
    results: tuple[RetrievalResult, ...]
