"""Orchestration for query rewriting, hybrid ranking, expansion, and diversity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from raglab.contracts import Citation
from raglab.retrieval.models import (
    CollectionMetadata,
    NeighborChunk,
    QueryEmbeddingProvider,
    QueryRewriter,
    RankingTrace,
    Reranker,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievedChunk,
)
from raglab.retrieval.ranking import (
    FusedCandidate,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from raglab.retrieval.repository import RetrievalRepository
from raglab.retrieval.reranking import BGEReranker


@dataclass(frozen=True, slots=True)
class _Parent:
    id: str
    document_id: str
    content: str
    citation: Citation
    matched_chunk_ids: tuple[str, ...]
    first_chunk_index: int
    last_chunk_index: int
    embedding: tuple[float, ...]
    source: FusedCandidate
    reranker_score: float | None


class RetrievalPipeline:
    def __init__(
        self,
        repository: RetrievalRepository,
        embedding_provider: QueryEmbeddingProvider,
        *,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.rewriter = rewriter
        self.reranker = reranker or BGEReranker()

    def collection_metadata(self, collection: str) -> CollectionMetadata | None:
        return self.repository.collection_metadata(collection)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        config = request.config
        variants, rewritten = self._query_variants(request)
        vectors = self.embedding_provider.embed_documents(variants)
        if len(vectors) != len(variants):
            raise ValueError("Embedding provider returned a different number of vectors")
        semantic = [
            self.repository.semantic_search(
                request.collection,
                vector,
                request.filters,
                limit=config.candidate_k,
                exact=config.exact,
                ef_search=config.ef_search,
            )
            for vector in vectors
        ]
        lexical = [
            self.repository.lexical_search(
                request.collection, variant, request.filters, limit=config.candidate_k
            )
            for variant in variants
        ]
        fused = reciprocal_rank_fusion(
            semantic,
            lexical,
            k=config.rrf_k,
            semantic_weight=config.semantic_weight,
            bm25_weight=config.bm25_weight,
        )
        reranker_scores = self._rerank(request, rewritten or request.query, fused)

        def score(index: int) -> float:
            reranker_score = reranker_scores[index]
            return fused[index].rrf_score if reranker_score is None else reranker_score

        order = sorted(
            range(len(fused)),
            key=lambda index: (
                -score(index),
                -fused[index].rrf_score,
                fused[index].chunk.chunk_id,
            ),
        )
        parents: dict[str, _Parent] = {}
        for index in order:
            candidate = fused[index]
            parent = self._expand(request, candidate, reranker_scores[index])
            previous = parents.get(parent.id)
            if previous is None:
                parents[parent.id] = parent
            else:
                parents[parent.id] = replace(
                    previous,
                    matched_chunk_ids=tuple(
                        dict.fromkeys(previous.matched_chunk_ids + parent.matched_chunk_ids)
                    ),
                )
        ranked = list(parents.values())
        selected = self._select(ranked, request)
        results = tuple(self._result(parent, mmr_score) for parent, mmr_score in selected)
        return RetrievalResponse(
            query=request.query,
            rewritten_query=rewritten,
            query_variants=tuple(variants),
            filters=request.filters,
            results=results,
        )

    def _query_variants(self, request: RetrievalRequest) -> tuple[list[str], str | None]:
        enabled = request.config.rewrite or bool(request.history)
        if not enabled or self.rewriter is None:
            return [request.query], None
        try:
            rewrite = self.rewriter.rewrite(
                request.query, request.history, max_expansions=request.config.expansions
            )
        except Exception:
            return [request.query], None
        standalone = rewrite.standalone_query.strip()
        candidates = [request.query, standalone, *rewrite.expansions[: request.config.expansions]]
        variants: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                variants.append(normalized)
        return variants or [request.query], standalone if standalone else None

    def _rerank(
        self, request: RetrievalRequest, query: str, candidates: Sequence[FusedCandidate]
    ) -> list[float | None]:
        if not request.config.rerank or not candidates:
            return [None] * len(candidates)
        scores = self.reranker.rerank(query, [candidate.chunk.content for candidate in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned a different number of scores")
        return [float(score) for score in scores]

    def _expand(
        self, request: RetrievalRequest, candidate: FusedCandidate, reranker_score: float | None
    ) -> _Parent:
        match = candidate.chunk
        if not request.config.small_to_big:
            return self._parent(candidate, [match], (match.chunk_id,), reranker_score)
        rows = self.repository.document_chunks(match.document_id, request.filters)
        position = next(
            (index for index, row in enumerate(rows) if row.chunk.chunk_id == match.chunk_id), None
        )
        if position is None:
            return self._parent(candidate, [match], (match.chunk_id,), reranker_score)
        chosen = self._centered_window(
            rows, position, match.heading_path, request.config.parent_max_tokens
        )
        return self._parent(candidate, chosen, (match.chunk_id,), reranker_score)

    def _centered_window(
        self,
        rows: Sequence[NeighborChunk],
        position: int,
        heading_path: tuple[str, ...],
        max_tokens: int,
    ) -> list[RetrievedChunk]:
        center = rows[position]
        if not center.matches_filters:
            return [center.chunk]
        chosen: dict[int, RetrievedChunk] = {position: center.chunk}
        used = center.chunk.token_count
        blocked_left = blocked_right = False
        distance = 1
        while not blocked_left or not blocked_right:
            progressed = False
            for index, is_left in ((position - distance, True), (position + distance, False)):
                if (is_left and blocked_left) or (not is_left and blocked_right):
                    continue
                if index < 0 or index >= len(rows):
                    if is_left:
                        blocked_left = True
                    else:
                        blocked_right = True
                    continue
                row = rows[index]
                expected_index = center.chunk.chunk_index + (-distance if is_left else distance)
                valid = row.matches_filters and row.chunk.chunk_index == expected_index
                if heading_path:
                    valid = valid and row.chunk.heading_path == heading_path
                size = row.chunk.token_count
                if not valid or used + size > max_tokens:
                    if is_left:
                        blocked_left = True
                    else:
                        blocked_right = True
                    continue
                chosen[index] = row.chunk
                used += size
                progressed = True
            if not progressed and blocked_left and blocked_right:
                break
            distance += 1
        return [chosen[index] for index in sorted(chosen)]

    @staticmethod
    def _parent(
        candidate: FusedCandidate,
        chunks: Sequence[RetrievedChunk],
        matched_ids: tuple[str, ...],
        reranker_score: float | None,
    ) -> _Parent:
        first, last = chunks[0], chunks[-1]
        base = candidate.chunk.citation
        citation = replace(
            base,
            start_page=min(
                (
                    chunk.citation.start_page
                    for chunk in chunks
                    if chunk.citation.start_page is not None
                ),
                default=None,
            ),
            end_page=max(
                (
                    chunk.citation.end_page
                    for chunk in chunks
                    if chunk.citation.end_page is not None
                ),
                default=None,
            ),
            start_line=min(
                (
                    chunk.citation.start_line
                    for chunk in chunks
                    if chunk.citation.start_line is not None
                ),
                default=None,
            ),
            end_line=max(
                (
                    chunk.citation.end_line
                    for chunk in chunks
                    if chunk.citation.end_line is not None
                ),
                default=None,
            ),
        )
        return _Parent(
            id=f"{first.document_id}:{first.chunk_index}-{last.chunk_index}",
            document_id=first.document_id,
            content="\n\n".join(chunk.content for chunk in chunks),
            citation=citation,
            matched_chunk_ids=matched_ids,
            first_chunk_index=first.chunk_index,
            last_chunk_index=last.chunk_index,
            embedding=candidate.chunk.embedding,
            source=candidate,
            reranker_score=reranker_score,
        )

    def _select(
        self, parents: Sequence[_Parent], request: RetrievalRequest
    ) -> list[tuple[_Parent, float | None]]:
        if not request.config.mmr:
            return [(parent, None) for parent in parents[: request.config.top_k]]
        ids = [parent.id for parent in parents]
        relevance = {parent.id: 1.0 / (index + 1) for index, parent in enumerate(parents)}
        embeddings = {parent.id: parent.embedding for parent in parents}
        selected = maximal_marginal_relevance(
            ids,
            embeddings,
            relevance,
            limit=request.config.top_k,
            lambda_=request.config.mmr_lambda,
        )
        by_id = {parent.id: parent for parent in parents}
        return [(by_id[item_id], score) for item_id, score in selected]

    @staticmethod
    def _result(parent: _Parent, mmr_score: float | None) -> RetrievalResult:
        chunk = parent.source.chunk
        return RetrievalResult(
            id=parent.id,
            document_id=parent.document_id,
            content=parent.content,
            citation=parent.citation,
            matched_chunk_ids=parent.matched_chunk_ids,
            first_chunk_index=parent.first_chunk_index,
            last_chunk_index=parent.last_chunk_index,
            trace=RankingTrace(
                ann_rank=parent.source.ann_rank,
                bm25_rank=parent.source.bm25_rank,
                ann_distance=chunk.ann_distance,
                bm25_score=chunk.bm25_score,
                rrf_score=parent.source.rrf_score,
                reranker_score=parent.reranker_score,
                mmr_score=mmr_score,
            ),
        )
