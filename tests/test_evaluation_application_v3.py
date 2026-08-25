from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from raglab.errors import EvaluationError, GenerationContractError
from raglab.evaluation import EvaluationApplication, HermeticEvaluationExecutor, load_manifest
from raglab.evaluation import cli as evaluation_cli
from raglab.evaluation.models import EvaluationCase


def _metadata(*, dirty: bool = False) -> dict[str, object]:
    return {"dirty": dirty, "hardware_fingerprint": "test-host"}


def _run(tmp_path: Path) -> dict[str, object]:
    return EvaluationApplication(
        HermeticEvaluationExecutor(),
        artifact_dir=tmp_path,
        metadata_provider=_metadata,
    ).run(load_manifest(), persist=False)


def test_v3_run_records_definition_evidence_checks_and_separate_quality_axes(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)

    assert run["schema_version"] == 3
    assert len(run["definition"]["fingerprint"]) == 64
    assert run["definition"]["manifest"]["profile"] == "core"
    assert len(run["ingestion"]["checks"]) == 6
    assert {check["type"] for check in run["ingestion"]["checks"]} == {
        "must_separate",
        "must_keep",
    }
    abstention = next(
        case
        for case in run["cases"]
        if case["id"] == "unsupported-password-rotation"
    )
    assert all(value is None for value in abstention["retrieval"]["metrics"].values())
    assert abstention["retrieval"]["ranges"] == []
    assert {"ingestion_separation_rate", "ingestion_cohesion_rate"} <= set(
        run["summary"]["quality"]
    )
    assert all(0.0 <= value <= 1.0 for value in run["summary"]["quality"].values())


def test_contract_failure_uses_null_for_checks_without_observable_fields(
    tmp_path: Path,
) -> None:
    class MissingFieldsExecutor(HermeticEvaluationExecutor):
        calls = 0

        def generate(self, case: EvaluationCase, *, collection: str):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise GenerationContractError("invalid JSON", raw_output="not-json")
            return super().generate(case, collection=collection)

    run = EvaluationApplication(
        MissingFieldsExecutor(), metadata_provider=_metadata
    ).run(load_manifest(), persist=False)
    checks = run["cases"][0]["generation"]["checks"][0]

    assert checks["contract"] is False
    assert checks["low-flow-e17"] is None
    assert checks["abstention"] is None
    assert checks["citations"] is None
    assert run["summary"]["quality"]["generation_pass_rate"] == pytest.approx(35 / 36)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda run: run.update(schema_version=2), "schema v3"),
        (lambda run: run.update(partial=True), "partial"),
        (lambda run: run["metadata"].update(dirty=True), "dirty"),
        (lambda run: run["errors"]["hard"].append("failure"), "hard failures"),
        (
            lambda run: run["definition"].update(fingerprint="changed"),
            "definition fingerprints differ",
        ),
    ],
)
def test_compare_rejects_ineligible_or_incompatible_v3_runs(
    tmp_path: Path, mutate, message: str  # type: ignore[no-untyped-def]
) -> None:
    baseline = _run(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "candidate"
    mutate(candidate)

    with pytest.raises(EvaluationError, match=message):
        EvaluationApplication(HermeticEvaluationExecutor()).compare(candidate, baseline)


def test_promotion_rejects_legacy_v2_run(tmp_path: Path) -> None:
    run = _run(tmp_path)
    run["schema_version"] = 2

    with pytest.raises(EvaluationError, match="schema v3"):
        EvaluationApplication(HermeticEvaluationExecutor()).promote(run)


def test_promotion_rejects_definition_mismatch_with_existing_baseline(
    tmp_path: Path,
) -> None:
    application = EvaluationApplication(
        HermeticEvaluationExecutor(), artifact_dir=tmp_path, metadata_provider=_metadata
    )
    baseline = application.run(load_manifest(), persist=False)
    destination = application.promote(baseline)
    candidate = copy.deepcopy(baseline)
    candidate["definition"]["fingerprint"] = "different"

    with pytest.raises(EvaluationError, match="fingerprint differs"):
        application.promote(candidate, destination=destination)


def test_run_cli_prints_compact_receipt_and_full_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        evaluation_cli,
        "LiveEvaluationExecutor",
        lambda **_kwargs: HermeticEvaluationExecutor(),
    )

    assert evaluation_cli.main(["--artifact-dir", str(tmp_path), "run"]) == 0
    compact = json.loads(capsys.readouterr().out)
    assert set(compact) == {"run_id", "artifacts", "status", "quality", "error_count"}
    assert compact["error_count"] == 0

    assert evaluation_cli.main(
        ["--artifact-dir", str(tmp_path), "run", "--full-json"]
    ) == 0
    full = json.loads(capsys.readouterr().out)
    assert full["schema_version"] == 3
    assert len(full["cases"]) == 12
