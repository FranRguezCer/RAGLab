from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from raglab.errors import EvaluationError
from raglab.evaluation import definition_fingerprint, load_manifest
from raglab.evaluation.metrics import (
    evidence_retrieval_metrics,
    normalize,
    retrieval_metrics,
)


def _write_manifest(
    path: Path, *, version: int, required_facts: list[object]
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": version,
                "profile": "core",
                "sources": [{"id": "manual", "path": "manual.md"}],
                "cases": [
                    {
                        "id": "low-flow",
                        "query": "What happens?",
                        "expected_source_ids": ["manual"],
                        "required_facts": required_facts,
                    }
                ],
                "config": {"top_k": 5},
            }
        )
    )
    return path


def test_manifest_v1_facts_are_upgraded_to_deterministic_expectations(
    tmp_path: Path,
) -> None:
    path = _write_manifest(
        tmp_path / "v1.json", version=1, required_facts=["raises `E17`"]
    )

    fact = load_manifest(path).cases[0].required_facts[0]

    assert fact.id == "low-flow-fact-1"
    assert fact.evidence_anchors == ("raises `E17`",)
    assert fact.answer_variants == ("raises `E17`",)


def test_manifest_v2_loads_distinct_evidence_anchors_and_answer_variants(
    tmp_path: Path,
) -> None:
    path = _write_manifest(
        tmp_path / "v2.json",
        version=2,
        required_facts=[
            {
                "id": "low-flow-e17",
                "evidence_anchors": ["raises `E17`"],
                "answer_variants": ["raises E17", "raises fault E17"],
            }
        ],
    )

    fact = load_manifest(path).cases[0].required_facts[0]

    assert fact.id == "low-flow-e17"
    assert fact.evidence_anchors == ("raises `E17`",)
    assert fact.answer_variants == ("raises E17", "raises fault E17")


def test_packaged_manifest_v3_loads_typed_bounded_semantic_contract() -> None:
    manifest = load_manifest()
    semantic = manifest.semantic

    assert manifest.schema_version == 3
    assert semantic is not None
    assert semantic.enabled is True
    assert semantic.model.name == "tasksource/deberta-small-long-nli"
    assert semantic.model.revision == "9a77395d4d3751be9e2a69c4ae318491d9b3fffb"
    assert semantic.model.device == "cpu"
    assert semantic.model.batch_size == 8
    assert semantic.model.max_tokens == 512
    assert semantic.calibration.threshold == 0.3138722777366639
    assert semantic.calibration.contradiction_threshold == 0.47856047749519354
    assert semantic.calibration.fingerprint == (
        "66d26dc318099c8dff68622a3b32b653840abb25429930f828e89b887177c125"
    )
    assert semantic.template.premise == "{answer}"
    assert semantic.template.hypothesis == "{semantic_claim}"
    assert all(
        fact.semantic_claim
        for case in manifest.cases
        for fact in case.required_facts
    )


@pytest.mark.parametrize(
    "facts, message",
    [
        (
            [
                {"id": "", "evidence_anchors": ["anchor"], "answer_variants": ["answer"]}
            ],
            "fact ids",
        ),
        (
            [{"id": "fact", "evidence_anchors": [], "answer_variants": ["answer"]}],
            "evidence anchors",
        ),
        (
            [{"id": "fact", "evidence_anchors": ["anchor"], "answer_variants": []}],
            "answer variants",
        ),
    ],
)
def test_manifest_v2_rejects_empty_fact_contracts(
    tmp_path: Path, facts: list[object], message: str
) -> None:
    path = _write_manifest(tmp_path / "invalid.json", version=2, required_facts=facts)

    with pytest.raises(EvaluationError, match=message):
        load_manifest(path)


def test_manifest_v2_rejects_duplicate_fact_ids(tmp_path: Path) -> None:
    fact = {
        "id": "duplicate",
        "evidence_anchors": ["anchor"],
        "answer_variants": ["answer"],
    }
    path = _write_manifest(
        tmp_path / "duplicate.json", version=2, required_facts=[fact, fact]
    )

    with pytest.raises(EvaluationError, match="fact ids"):
        load_manifest(path)


def test_normalize_is_unicode_markdown_and_punctuation_insensitive() -> None:
    assert normalize("  Raises **`Ｅ１７`**, now!  ") == "raises e17 now"
    assert normalize("Use surrogate-keys.") == normalize("use surrogate keys")


def test_definition_fingerprint_changes_with_any_ground_truth_change(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "v2.json",
            version=2,
            required_facts=[
                {
                    "id": "fact",
                    "evidence_anchors": ["anchor"],
                    "answer_variants": ["answer"],
                }
            ],
        )
    )
    case = manifest.cases[0]
    changed = replace(manifest, cases=(replace(case, query="Changed question?"),))

    assert definition_fingerprint(manifest) != definition_fingerprint(changed)
    assert definition_fingerprint(manifest) == definition_fingerprint(
        replace(manifest, base_path="/a/different/checkout")
    )


def test_evidence_retrieval_metrics_cover_distinct_facts_by_rank() -> None:
    metrics = evidence_retrieval_metrics(
        [(), ("fact-a",), ("fact-a", "fact-b"), ("noise",)],
        ["fact-a", "fact-b", "fact-c"],
    )

    assert metrics == {
        "hit_at_1": 0.0,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "recall_at_5": pytest.approx(2 / 3),
        "mrr": 0.5,
    }


def test_evidence_retrieval_metrics_are_null_without_required_facts() -> None:
    assert set(evidence_retrieval_metrics([("noise",)], []).values()) == {None}


def test_source_ndcg_deduplicates_repeated_relevant_sources() -> None:
    metrics = retrieval_metrics(["a", "a", "a", "b", "b"], ["a", "b"])

    assert metrics["recall_at_5"] == 1.0
    assert 0.0 <= metrics["ndcg_at_5"] <= 1.0
