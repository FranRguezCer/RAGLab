from __future__ import annotations

from collections.abc import Sequence

from raglab.contracts import Citation, ProvenanceStatus
from raglab.retrieval import (
    MetadataFilter,
    NeighborChunk,
    QueryRewrite,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalRequest,
    RetrievedChunk,
)


def _chunk(
    index: int,
    *,
    document: str = "doc",
    heading: tuple[str, ...] = ("Faults",),
    embedding: tuple[float, ...] = (1.0, 0.0),
    ann: float | None = None,
    bm25: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{document}-c{index}",
        document_id=document,
        chunk_index=index,
        content=f"chunk {index}",
        token_count=2,
        heading_path=heading,
        embedding=embedding,
        citation=Citation(
            source_uri=f"memory://{document}",
            source_name=document,
            title="Manual",
            heading_path=heading,
            start_page=index + 1,
            end_page=index + 1,
            start_line=index * 10 + 1,
            end_line=index * 10 + 5,
            provenance_status=ProvenanceStatus.COMPLETE,
        ),
        ann_distance=ann,
        bm25_score=bm25,
    )


class _Embeddings:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class _Repository:
    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        self.chunks = list(chunks)
        self.semantic_calls: list[tuple[str, tuple[MetadataFilter, ...]]] = []
        self.lexical_calls: list[tuple[str, tuple[MetadataFilter, ...]]] = []
        self.neighbors: dict[str, list[NeighborChunk]] = {}

    def semantic_search(
        self,
        collection: str,
        embedding: Sequence[float],
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
        exact: bool,
        ef_search: int,
    ) -> list[RetrievedChunk]:
        self.semantic_calls.append((collection, tuple(filters)))
        return self.chunks[:limit]

    def lexical_search(
        self,
        collection: str,
        query: str,
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        self.lexical_calls.append((query, tuple(filters)))
        return list(reversed(self.chunks[:limit]))

    def document_chunks(
        self, document_id: str, filters: Sequence[MetadataFilter]
    ) -> list[NeighborChunk]:
        return self.neighbors.get(document_id, [])


class _Rewriter:
    def rewrite(self, query: str, history: Sequence[str], *, max_expansions: int) -> QueryRewrite:
        return QueryRewrite("What causes fault E17?", ("E17 root cause", "e17 ROOT CAUSE"))


class _BrokenRewriter:
    def rewrite(self, query: str, history: Sequence[str], *, max_expansions: int) -> QueryRewrite:
        raise RuntimeError("offline")


class _Reranker:
    def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [float(index) for index in range(len(documents))]


def test_rewrite_preserves_original_deduplicates_and_uses_same_filters() -> None:
    repo = _Repository([_chunk(0)])
    embeddings = _Embeddings()
    tenant = MetadataFilter("document.metadata.tenant", "eq", "acme")
    request = RetrievalRequest(
        "Why E17?",
        filters=(tenant,),
        config=RetrievalConfig(
            rewrite=True, expansions=2, rerank=False, mmr=False, small_to_big=False
        ),
    )

    response = RetrievalPipeline(repo, embeddings, rewriter=_Rewriter()).retrieve(request)

    assert response.query_variants == (
        "Why E17?",
        "What causes fault E17?",
        "E17 root cause",
    )
    assert response.rewritten_query == "What causes fault E17?"
    assert embeddings.texts == list(response.query_variants)
    assert all(filters == (tenant,) for _, filters in repo.semantic_calls)
    assert all(filters == (tenant,) for _, filters in repo.lexical_calls)


def test_rewrite_failure_falls_back_to_original_query() -> None:
    repo = _Repository([])
    response = RetrievalPipeline(repo, _Embeddings(), rewriter=_BrokenRewriter()).retrieve(
        RetrievalRequest(
            "original",
            config=RetrievalConfig(rewrite=True, rerank=False, mmr=False),
        )
    )

    assert response.query_variants == ("original",)
    assert response.rewritten_query is None


def test_history_enables_rewriting_without_explicit_flag() -> None:
    repo = _Repository([])
    response = RetrievalPipeline(repo, _Embeddings(), rewriter=_Rewriter()).retrieve(
        RetrievalRequest(
            "And E17?",
            history=("We discussed faults",),
            config=RetrievalConfig(rerank=False, mmr=False),
        )
    )

    assert response.rewritten_query == "What causes fault E17?"


def test_reranker_changes_order_and_trace() -> None:
    first, second = _chunk(0), _chunk(1)
    repo = _Repository([first, second])
    response = RetrievalPipeline(repo, _Embeddings(), reranker=_Reranker()).retrieve(
        RetrievalRequest("fault", config=RetrievalConfig(mmr=False, small_to_big=False))
    )

    assert response.results[0].matched_chunk_ids == (second.chunk_id,)
    assert response.results[0].trace.reranker_score == 1.0
    assert response.results[0].trace.rrf_score > 0


def test_small_to_big_stops_at_heading_filter_gap_and_budget() -> None:
    matched = _chunk(2)
    repo = _Repository([matched])
    repo.neighbors["doc"] = [
        NeighborChunk(_chunk(0), True),
        NeighborChunk(_chunk(1), False),
        NeighborChunk(matched, True),
        NeighborChunk(_chunk(3), True),
        NeighborChunk(_chunk(4, heading=("Other",)), True),
    ]
    request = RetrievalRequest(
        "fault",
        filters=(MetadataFilter("document.metadata.tenant", "eq", "acme"),),
        config=RetrievalConfig(rerank=False, mmr=False, parent_max_tokens=4),
    )

    result = RetrievalPipeline(repo, _Embeddings()).retrieve(request).results[0]

    assert result.id == "doc:2-3"
    assert result.content == "chunk 2\n\nchunk 3"
    assert result.citation.start_line == 21
    assert result.citation.end_line == 35


def test_empty_heading_uses_centered_window_across_neighbor_headings() -> None:
    matched = _chunk(1, heading=())
    repo = _Repository([matched])
    repo.neighbors["doc"] = [
        NeighborChunk(_chunk(0, heading=("Previous",)), True),
        NeighborChunk(matched, True),
        NeighborChunk(_chunk(2, heading=("Next",)), True),
    ]

    result = (
        RetrievalPipeline(repo, _Embeddings())
        .retrieve(
            RetrievalRequest(
                "fault",
                config=RetrievalConfig(rerank=False, mmr=False, parent_max_tokens=6),
            )
        )
        .results[0]
    )

    assert result.id == "doc:0-2"


def test_mmr_uses_best_child_embedding_and_diversifies_results() -> None:
    chunks = [
        _chunk(0, document="a", embedding=(1.0, 0.0)),
        _chunk(0, document="b", embedding=(0.99, 0.01)),
        _chunk(0, document="c", embedding=(0.0, 1.0)),
    ]
    repo = _Repository(chunks)
    response = RetrievalPipeline(repo, _Embeddings()).retrieve(
        RetrievalRequest(
            "fault",
            config=RetrievalConfig(top_k=2, rerank=False, small_to_big=False, mmr_lambda=0.5),
        )
    )

    assert [result.document_id for result in response.results] == ["a", "c"]
    assert all(result.trace.mmr_score is not None for result in response.results)
