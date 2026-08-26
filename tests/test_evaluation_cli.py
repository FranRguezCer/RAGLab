from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.errors import GenerationContractError
from raglab.evaluation.application import HermeticEvaluationExecutor
from raglab.evaluation.cli import main
from raglab.evaluation.semantic import CalibrationResult, NLIScores


@pytest.fixture(autouse=True)
def _stub_semantic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class Scorer:
        def __init__(self, _config: object) -> None:
            pass

        def score(self, pairs):  # type: ignore[no-untyped-def]
            return [NLIScores(0.99, 0.0) for _ in pairs]

    monkeypatch.setattr("raglab.evaluation.application.TransformersNLIScorer", Scorer)


def test_run_cli_uses_evaluation_application(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "raglab.evaluation.cli.LiveEvaluationExecutor",
        lambda **_kwargs: HermeticEvaluationExecutor(),
    )
    assert main(["--artifact-dir", str(tmp_path), "run", "--profile", "core"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["error_count"] == 0
    assert set(payload["artifacts"]) == {"json", "markdown"}


def test_compare_and_promote_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "raglab.evaluation.cli.LiveEvaluationExecutor",
        lambda **_kwargs: HermeticEvaluationExecutor(),
    )
    assert main(["--artifact-dir", str(tmp_path), "run", "--full-json"]) == 0
    run = json.loads(capsys.readouterr().out)
    run["metadata"]["dirty"] = False
    run_path = tmp_path / "candidate.json"
    run_path.write_text(json.dumps(run))

    assert main(["--artifact-dir", str(tmp_path), "baseline", "promote", str(run_path)]) == 0
    capsys.readouterr()
    assert main(["--artifact-dir", str(tmp_path), "compare", str(run_path)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "no_clear_change"


def test_run_cli_prints_complete_report_and_exits_one_for_hard_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    class ContractFailingExecutor(HermeticEvaluationExecutor):
        def generate(self, case, *, collection):  # type: ignore[no-untyped-def]
            raise GenerationContractError(
                "invalid grounding",
                raw_output='{"answer":"uncited","abstained":false,"source_ids":[]}',
                answer="uncited",
                abstained=False,
                cited_source_ids=(),
            )

    monkeypatch.setattr(
        "raglab.evaluation.cli.LiveEvaluationExecutor",
        lambda **_kwargs: ContractFailingExecutor(),
    )

    assert main(["--artifact-dir", str(tmp_path), "run", "--profile", "core", "--full-json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert len(payload["cases"]) == 12
    assert payload["summary"]["quality"]["generation_pass_rate"] == 0.0
    assert len(payload["errors"]["hard"]) == 36


def test_calibrate_semantic_cli_reports_each_locked_split(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = CalibrationResult(
        enabled=True,
        threshold=0.81,
        contradiction_threshold=0.92,
        calibration_pairs=128,
        holdout_pairs=112,
        false_promotions=0,
        valid_promotions=48,
        calibration_valid_pairs=48,
        calibration_valid_promotions=48,
        calibration_false_promotions=0,
        calibration_false_vetoes=0,
        calibration_contradiction_vetoes=16,
        holdout_valid_pairs=32,
        holdout_valid_promotions=32,
        holdout_false_promotions=0,
        holdout_false_vetoes=0,
        holdout_contradiction_vetoes=16,
        frozen_rescues=12,
        frozen_rejections=6,
        model="cross-encoder/nli-deberta-v3-small",
        revision="fa2804872c3b4bd748f38c0185cc85775361e735",
        fixture_fingerprint="fixture-fingerprint",
    )
    monkeypatch.setattr("raglab.evaluation.cli.TransformersNLIScorer", lambda _config: object())
    monkeypatch.setattr(
        "raglab.evaluation.cli.calibrate", lambda _config, _scorer, _manifest: result
    )

    assert main(["calibrate-semantic", "--profile", "core"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["valid_promotions"] == 48
    assert payload["false_promotions"] == 0
    assert payload["calibration_valid_pairs"] == 48
    assert payload["calibration_valid_promotions"] == 48
    assert payload["calibration_false_promotions"] == 0
    assert payload["holdout_valid_pairs"] == 32
    assert payload["holdout_valid_promotions"] == 32
    assert payload["holdout_false_promotions"] == 0
    assert payload["authority"] == "semantic_fact_rescue_and_contradiction_veto"
