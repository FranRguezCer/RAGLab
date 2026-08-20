from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.evaluation.application import HermeticEvaluationExecutor
from raglab.evaluation.cli import main


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
    assert len(payload["cases"]) == 12


def test_compare_and_promote_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "raglab.evaluation.cli.LiveEvaluationExecutor",
        lambda **_kwargs: HermeticEvaluationExecutor(),
    )
    assert main(["--artifact-dir", str(tmp_path), "run"]) == 0
    run = json.loads(capsys.readouterr().out)
    run["metadata"]["dirty"] = False
    run_path = tmp_path / "candidate.json"
    run_path.write_text(json.dumps(run))

    assert main(["--artifact-dir", str(tmp_path), "baseline", "promote", str(run_path)]) == 0
    capsys.readouterr()
    assert main(["--artifact-dir", str(tmp_path), "compare", str(run_path)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "no_clear_change"
