from __future__ import annotations

import pytest

from raglab.contracts import Citation, ProvenanceStatus
from raglab.retrieval import RetrievedChunk, maximal_marginal_relevance, reciprocal_rank_fusion


def _chunk(
    identifier: str, *, ann: float | None = None, bm25: float | None = None
) -> RetrievedChunk:
    citation = Citation(
        source_uri="memory://doc",
        source_name="doc",
        title=None,
        heading_path=("Section",),
        start_page=None,
        end_page=None,
        start_line=1,
        end_line=1,
        provenance_status=ProvenanceStatus.COMPLETE,
    )
    return RetrievedChunk(
        chunk_id=identifier,
        document_id="document",
        chunk_index=int(identifier[-1]),
        content=identifier,
        token_count=1,
        heading_path=("Section",),
        embedding=(1.0, 0.0),
        citation=citation,
        ann_distance=ann,
        bm25_score=bm25,
    )


def test_rrf_fuses_channels_and_preserves_channel_scores() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("c1", ann=0.1), _chunk("c2", ann=0.2)]],
        [[_chunk("c2", bm25=9.0), _chunk("c3", bm25=8.0)]],
    )

    assert [item.chunk.chunk_id for item in fused] == ["c2", "c1", "c3"]
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].ann_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].chunk.ann_distance == 0.2
    assert fused[0].chunk.bm25_score == 9.0


def test_rrf_does_not_count_duplicate_within_one_ranking() -> None:
    fused = reciprocal_rank_fusion([[_chunk("c1"), _chunk("c1")]], [])

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1 / 61)


def test_rrf_zero_weight_excludes_that_channel() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("c1", ann=0.1)]],
        [[_chunk("c2", bm25=9.0)]],
        semantic_weight=0,
    )

    assert [item.chunk.chunk_id for item in fused] == ["c2"]


def test_mmr_prefers_a_relevant_but_diverse_result() -> None:
    selected = maximal_marginal_relevance(
        ["a", "b", "c"],
        {"a": (1.0, 0.0), "b": (0.99, 0.01), "c": (0.0, 1.0)},
        {"a": 1.0, "b": 0.9, "c": 0.8},
        limit=2,
        lambda_=0.5,
    )

    assert [identifier for identifier, _ in selected] == ["a", "c"]
