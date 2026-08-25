from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from raglab.errors import (
    EvaluationError,
    GenerationContractError,
    GenerationError,
    StorageError,
)
from raglab.evaluation import (
    EvaluationApplication,
    HermeticEvaluationExecutor,
    corpus_fingerprint,
    load_manifest,
)
from raglab.evaluation.metrics import conservative_verdict, retrieval_metrics
from raglab.evaluation.models import (
    RUN_SCHEMA_VERSION,
    EvaluationCase,
    GenerationObservation,
)
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
    assert run["schema_version"] == RUN_SCHEMA_VERSION == 4
    assert run["partial"] is False
    assert run["errors"]["hard"] == []
    assert all(case["generation"]["stability"] == "3/3" for case in run["cases"])
    assert (tmp_path / f"{run['run_id']}.json").exists()
    assert "## Quality axes" in (tmp_path / f"{run['run_id']}.md").read_text()


def test_contract_failures_are_recorded_without_aborting_the_full_run(
    tmp_path: Path,
) -> None:
    class ContractFailingExecutor(HermeticEvaluationExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.generation_calls = 0

        def generate(self, case: EvaluationCase, *, collection: str):  # type: ignore[no-untyped-def]
            self.generation_calls += 1
            if self.generation_calls == 1:
                raise GenerationContractError(
                    "A non-abstaining answer must cite retrieved evidence",
                    raw_output='{"answer":"uncited","abstained":false,"source_ids":[]}',
                    answer="uncited",
                    abstained=False,
                    cited_source_ids=(),
                )
            return super().generate(case, collection=collection)

    executor = ContractFailingExecutor()
    application = EvaluationApplication(
        executor,
        artifact_dir=tmp_path,
        generation_model="generator",
        metadata_provider=metadata,
    )

    run = application.run(load_manifest())

    assert run["status"] == "complete"
    assert len(run["cases"]) == 12
    assert executor.generation_calls == 36
    failed = run["cases"][0]["generation"]["repetitions"][0]
    assert failed == {
        "status": "failed",
        "error": "A non-abstaining answer must cite retrieved evidence",
        "raw_output": '{"answer":"uncited","abstained":false,"source_ids":[]}',
        "latency_ms": pytest.approx(failed["latency_ms"]),
        "answer": "uncited",
        "abstained": False,
        "cited_source_ids": [],
    }
    assert "prompt_tokens" not in failed
    assert "model_calls" not in failed
    assert run["summary"]["quality"]["generation_pass_rate"] == pytest.approx(35 / 36)
    assert run["errors"]["hard"][0].startswith("aster-low-flow repetition 1:")
    with pytest.raises(EvaluationError, match="hard failures"):
        application.promote(run)
    markdown = (tmp_path / f"{run['run_id']}.md").read_text()
    assert "Repetition 1 failed: A non-abstaining answer" in markdown


@pytest.mark.parametrize("failed_check", ["required_facts", "abstention", "citations"])
def test_failed_generation_checks_are_recorded_per_repetition(
    failed_check: str, tmp_path: Path
) -> None:
    class CheckFailingExecutor(HermeticEvaluationExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.generation_calls = 0

        def generate(
            self, case: EvaluationCase, *, collection: str
        ) -> GenerationObservation:
            observation = super().generate(case, collection=collection)
            self.generation_calls += 1
            if self.generation_calls != 1:
                return observation
            if failed_check == "required_facts":
                return replace(observation, answer="Unsupported answer [S1].")
            if failed_check == "abstention":
                return replace(observation, abstained=not observation.abstained)
            return replace(observation, cited_source_ids=())

    application = EvaluationApplication(
        CheckFailingExecutor(), artifact_dir=tmp_path, metadata_provider=metadata
    )

    run = application.run(load_manifest())

    failed = run["cases"][0]["generation"]["repetitions"][0]
    assert failed["status"] == "failed"
    expected_check = "low-flow-e17" if failed_check == "required_facts" else failed_check
    assert failed["error"] == f"Generation checks failed: {expected_check}"
    assert "raw_output" not in failed
    assert failed["prompt_tokens"] == 100
    assert failed["generated_tokens"] == 20
    assert failed["model_calls"] == 1
    assert failed["latency_ms"] == 3.0
    assert run["summary"]["quality"]["generation_pass_rate"] == pytest.approx(35 / 36)
    assert run["errors"]["hard"] == [
        f"aster-low-flow repetition 1: Generation checks failed: {expected_check}"
    ]
    markdown = (tmp_path / f"{run['run_id']}.md").read_text()
    assert f"Repetition 1 failed: Generation checks failed: {expected_check}" in markdown


def test_operational_generation_error_still_aborts_the_run(tmp_path: Path) -> None:
    class OperationalFailureExecutor(HermeticEvaluationExecutor):
        def generate(self, case: EvaluationCase, *, collection: str):  # type: ignore[no-untyped-def]
            raise GenerationError("Ollama connection failed")

    application = EvaluationApplication(
        OperationalFailureExecutor(), artifact_dir=tmp_path, metadata_provider=metadata
    )

    with pytest.raises(GenerationError, match="connection failed"):
        application.run(load_manifest())

    artifacts = list(tmp_path.glob("*.json"))
    assert len(artifacts) == 1
    failed_run = json.loads(artifacts[0].read_text())
    assert failed_run["status"] == "failed"
    assert failed_run["cases"] == []


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
