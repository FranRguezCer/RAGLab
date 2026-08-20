from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from raglab.errors import EvaluationError, StorageError
from raglab.evaluation import (
    EvaluationApplication,
    HermeticEvaluationExecutor,
    corpus_fingerprint,
    load_manifest,
)
from raglab.evaluation.metrics import conservative_verdict, retrieval_metrics
from raglab.evaluation.models import EvaluationCase
from raglab.storage import PostgresRepository


def metadata(*, dirty: bool = False, hardware: str = "host-a") -> dict[str, Any]:
    return {
        "commit": "abc123",
        "dirty": dirty,
        "generation_model": "generator",
        "embedding_model": "embedding",
        "hardware_fingerprint": hardware,
    }


def test_core_manifest_is_packaged_valid_and_fingerprinted() -> None:
    manifest = load_manifest()

    assert manifest.profile == "core"
    assert len(manifest.sources) == 4
    assert len(manifest.cases) == 12
    assert sum(bool(case.history) for case in manifest.cases) == 2
    assert manifest.config["chunking"]["semantic_percentile"] == 85
    fingerprint, hashes = corpus_fingerprint(manifest)
    assert len(fingerprint) == 64
    assert set(hashes) == {source.id for source in manifest.sources}

    live = load_manifest(profile="live")
    assert {source.domain for source in live.sources} == {
        "research",
        "computers",
        "accessories",
        "microcontrollers",
    }


def test_manifest_rejects_unknown_source_and_unset_live_path(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "core",
                "sources": [{"id": "known", "path": "missing.md"}],
                "cases": [
                    {
                        "id": "bad",
                        "query": "Question?",
                        "expected_source_ids": ["unknown"],
                        "required_facts": [],
                    }
                ],
            }
        )
    )
    with pytest.raises(EvaluationError, match="unknown source"):
        load_manifest(invalid)

    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "live",
                "sources": [
                    {
                        "id": "local",
                        "path": "$RAGLAB_TEST_MISSING/source.pdf",
                        "domain": "test",
                    }
                ],
                "cases": [
                    {
                        "id": "case",
                        "query": "Q?",
                        "expected_source_ids": [],
                        "required_facts": [],
                        "should_abstain": True,
                        "domain": "test",
                    }
                ],
            }
        )
    )
    with pytest.raises(EvaluationError, match="unset environment variable"):
        corpus_fingerprint(load_manifest(missing, profile="live"))


def test_retrieval_metrics_and_conservative_verdicts() -> None:
    values = retrieval_metrics(["noise", "a", "b"], ["a", "b"])
    assert values["hit_at_1"] == 0
    assert values["hit_at_3"] == 1
    assert values["recall_at_5"] == 1
    assert values["mrr"] == 0.5
    assert conservative_verdict({"a": 1}, {"a": 2}) == "improved"
    assert conservative_verdict({"a": 2}, {"a": 1}) == "regressed"
    assert conservative_verdict({"a": 1, "b": 2}, {"a": 2, "b": 1}) == "mixed"
    assert conservative_verdict({"a": 1}, {"a": 1}) == "no_clear_change"


def test_application_runs_three_repetitions_and_persists_artifacts(tmp_path: Path) -> None:
    manifest = load_manifest()
    application = EvaluationApplication(
        HermeticEvaluationExecutor(),
        artifact_dir=tmp_path,
        generation_model="generator",
        metadata_provider=metadata,
    )

    run = application.run(manifest)

    assert run["status"] == "complete"
    assert run["partial"] is False
    assert run["errors"]["hard"] == []
    assert all(case["generation"]["stability"] == "3/3" for case in run["cases"])
    assert (tmp_path / f"{run['run_id']}.json").exists()
    assert "## Quality axes" in (tmp_path / f"{run['run_id']}.md").read_text()


def test_compare_checks_compatibility_and_hardware(tmp_path: Path) -> None:
    manifest = load_manifest()
    baseline = EvaluationApplication(
        HermeticEvaluationExecutor(),
        artifact_dir=tmp_path,
        metadata_provider=lambda: metadata(hardware="host-a"),
    ).run(manifest, persist=False)
    candidate = EvaluationApplication(
        HermeticEvaluationExecutor(latency_scale=2),
        artifact_dir=tmp_path,
        metadata_provider=lambda: metadata(hardware="host-b"),
    ).run(manifest, persist=False)

    comparison = EvaluationApplication(
        HermeticEvaluationExecutor(), artifact_dir=tmp_path
    ).compare(candidate, baseline)
    assert comparison["verdict"] == "no_clear_change"
    assert comparison["latency_compatible"] is False
    assert "latency" not in comparison

    candidate["corpus"]["fingerprint"] = "changed"
    with pytest.raises(EvaluationError, match="Corpus fingerprints differ"):
        EvaluationApplication(HermeticEvaluationExecutor()).compare(candidate, baseline)


def test_baseline_promotion_rejects_partial_dirty_and_failed_runs(tmp_path: Path) -> None:
    manifest = load_manifest()
    application = EvaluationApplication(
        HermeticEvaluationExecutor(),
        artifact_dir=tmp_path,
        metadata_provider=metadata,
    )
    run = application.run(manifest, persist=False)
    destination = application.promote(run)
    assert json.loads(destination.read_text())["run_id"] == run["run_id"]

    run["partial"] = True
    with pytest.raises(EvaluationError, match="reuse-index"):
        application.promote(run)
    run["partial"] = False
    run["metadata"]["dirty"] = True
    with pytest.raises(EvaluationError, match="dirty"):
        application.promote(run)
    run["metadata"]["dirty"] = False
    run["errors"]["hard"] = ["failure"]
    with pytest.raises(EvaluationError, match="hard failures"):
        application.promote(run)


def test_reuse_index_marks_run_partial_and_non_promotable(tmp_path: Path) -> None:
    application = EvaluationApplication(
        HermeticEvaluationExecutor(), artifact_dir=tmp_path, metadata_provider=metadata
    )
    run = application.run(load_manifest(), reuse_index=True, persist=False)
    assert run["partial"] is True
    with pytest.raises(EvaluationError, match="reuse-index"):
        application.promote(run)


def test_judge_is_advisory_and_must_use_a_different_model(tmp_path: Path) -> None:
    class Judge:
        model = "judge"

        def evaluate(self, case: EvaluationCase, answer: str) -> dict[str, Any]:
            if case.id == "aster-low-flow":
                raise MemoryError("simulated OOM")
            return {"grounded": bool(answer), "relevant": True, "reason": "controlled"}

    with pytest.raises(EvaluationError, match="must differ"):
        EvaluationApplication(
            HermeticEvaluationExecutor(), judge=Judge(), generation_model="judge"
        )
    run = EvaluationApplication(
        HermeticEvaluationExecutor(),
        artifact_dir=tmp_path,
        judge=Judge(),
        generation_model="generator",
        metadata_provider=metadata,
    ).run(load_manifest(), persist=False)
    assert run["status"] == "complete"
    assert run["errors"]["hard"] == []
    assert run["errors"]["advisory"]
    assert run["judge"]["status"] == "partial"


def test_collection_reset_rejects_personal_names_before_connecting() -> None:
    repository = PostgresRepository("not-used")
    with pytest.raises(StorageError, match="protected prefix"):
        repository.reset_evaluation_collection("documents")
    with pytest.raises(StorageError, match="protected prefix"):
        repository.reset_evaluation_collection("raglab-eval-")
