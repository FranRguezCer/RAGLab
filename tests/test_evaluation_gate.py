from __future__ import annotations

import pytest

from raglab.evaluation_gate import main, validate_promotion


def _run() -> dict[str, object]:
    return {
        "report": {
            "dataset_id": "demo",
            "retrieval_summary": {"recall_at_k": 1.0},
            "generation_summary": {"abstention_accuracy": 1.0},
        },
        "traces": [
            {
                "generation": {
                    "case_id": "case",
                    "forbidden_phrase_hits": [],
                    "invalid_citation_source_ids": [],
                    "abstention_correct": True,
                }
            }
        ],
    }


def test_gate_passes_clean_run_and_blocks_hard_failure_or_regression() -> None:
    baseline = {
        "dataset_id": "demo",
        "retrieval_summary": {"recall_at_k": 1.0},
        "generation_summary": {"abstention_accuracy": 1.0},
    }
    validate_promotion(_run(), baseline)

    failed = _run()
    failed["traces"][0]["generation"]["abstention_correct"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="wrong abstention"):
        validate_promotion(failed, baseline)

    regressed = _run()
    regressed["report"]["retrieval_summary"]["recall_at_k"] = 0.94  # type: ignore[index]
    with pytest.raises(ValueError, match="regression"):
        validate_promotion(regressed, baseline)


def test_gate_cli_reads_artifacts(tmp_path: object, capsys: pytest.CaptureFixture[str]) -> None:
    import json
    from pathlib import Path

    directory = Path(str(tmp_path))
    run_path = directory / "run.json"
    baseline_path = directory / "baseline.json"
    baseline = {
        "dataset_id": "demo",
        "retrieval_summary": {"recall_at_k": 1.0},
        "generation_summary": {"abstention_accuracy": 1.0},
    }
    run_path.write_text(json.dumps(_run()))
    baseline_path.write_text(json.dumps(baseline))

    assert main([str(run_path), str(baseline_path)]) == 0
    assert "passed" in capsys.readouterr().out
