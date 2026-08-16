"""Pure ranking operations used by the retrieval pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from raglab.retrieval.models import RetrievedChunk


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk: RetrievedChunk
    rrf_score: float
    ann_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    semantic_rankings: Sequence[Sequence[RetrievedChunk]],
    lexical_rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[FusedCandidate]:
    """Fuse every query-variant/channel ranking, de-duplicating within each ranking."""
    if k < 1:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    ann_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    def add(
        rankings: Sequence[Sequence[RetrievedChunk]],
        weight: float,
        ranks: dict[str, int],
        *,
        channel: str,
    ) -> None:
        if weight == 0:
            return
        for ranking in rankings:
            seen: set[str] = set()
            for rank, chunk in enumerate(ranking, start=1):
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                previous = chunks.get(chunk.chunk_id)
                previous_rank = ranks.get(chunk.chunk_id)
                if previous is None:
                    chunks[chunk.chunk_id] = chunk
                else:
                    chunks[chunk.chunk_id] = replace(
                        previous,
                        ann_distance=(
                            chunk.ann_distance
                            if channel == "ann" and (previous_rank is None or rank < previous_rank)
                            else previous.ann_distance
                        ),
                        bm25_score=(
                            chunk.bm25_score
                            if channel == "bm25" and (previous_rank is None or rank < previous_rank)
                            else previous.bm25_score
                        ),
                    )
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
                ranks[chunk.chunk_id] = min(rank, ranks.get(chunk.chunk_id, rank))

    add(semantic_rankings, semantic_weight, ann_ranks, channel="ann")
    add(lexical_rankings, bm25_weight, bm25_ranks, channel="bm25")
    return sorted(
        (
            FusedCandidate(
                chunk=chunk,
                rrf_score=scores[chunk_id],
                ann_rank=ann_ranks.get(chunk_id),
                bm25_rank=bm25_ranks.get(chunk_id),
            )
            for chunk_id, chunk in chunks.items()
        ),
        key=lambda item: (-item.rrf_score, item.chunk.chunk_id),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def maximal_marginal_relevance(
    ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
    relevance: Mapping[str, float],
    *,
    limit: int,
    lambda_: float = 0.7,
) -> list[tuple[str, float]]:
    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between zero and one")
    selected: list[str] = []
    result: list[tuple[str, float]] = []
    remaining = list(dict.fromkeys(ids))
    while remaining and len(selected) < limit:
        scored: list[tuple[float, str]] = []
        for item_id in remaining:
            redundancy = max(
                (cosine_similarity(embeddings[item_id], embeddings[chosen]) for chosen in selected),
                default=0.0,
            )
            score = lambda_ * relevance[item_id] - (1 - lambda_) * redundancy
            scored.append((score, item_id))
        score, winner = max(scored, key=lambda item: (item[0], -ids.index(item[1])))
        selected.append(winner)
        result.append((winner, score))
        remaining.remove(winner)
    return result
