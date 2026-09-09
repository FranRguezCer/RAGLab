from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from raglab.evaluation import load_cases
from raglab.evaluation_gate import main, validate_promotion

DATASET_SHA = "a" * 64


def _result(case_id: str, outcome: str) -> tuple[dict[str, object], dict[str, object]]:
    retrieval = {"case_id": case_id, "top_k": 3, "precision_at_k": 1 / 3,
                 "recall_at_k": 1.0, "mrr_at_k": 1.0}
    generation = {
        "case_id": case_id, "expected_outcome": outcome, "fact_coverage": 1.0,
        "grounded_fact_coverage": 1.0, "citation_precision": 1.0,
        "forbidden_phrase_hits": [], "invalid_citation_source_ids": [],
        "abstention_correct": True,
    }
    return retrieval, generation


def _run() -> dict[str, object]:
    answer_r, answer_g = _result("answer", "answer")
    abstain_r, abstain_g = _result("abstain", "abstain")
    traces = [
        {"case": {"id": "answer", "expected_outcome": "answer"},
         "response": {"answer": "grounded"}, "retrieval": answer_r, "generation": answer_g},
        {"case": {"id": "abstain", "expected_outcome": "abstain"},
         "response": {"answer": ""}, "retrieval": abstain_r, "generation": abstain_g},
    ]
    return {
        "metadata": {"dataset_sha256": DATASET_SHA}, "traces": traces,
        "report": {
            "schema_version": 1, "dataset_id": "demo", "case_ids": ["answer", "abstain"],
            "top_k": 3, "retrieval": [answer_r, abstain_r],
            "generation": [answer_g, abstain_g],
            "retrieval_summary": {"precision_at_k": 1 / 3, "recall_at_k": 1.0, "mrr_at_k": 1.0},
            "generation_summary": {"fact_coverage": 1.0, "grounded_fact_coverage": 1.0,
                                   "citation_precision": 1.0, "abstention_accuracy": 1.0},
        },
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": 1, "report_schema_version": 1, "dataset_id": "demo",
        "dataset_sha256": DATASET_SHA, "case_ids": ["answer", "abstain"], "top_k": 3,
        "retrieval_summary": {"precision_at_k": 1 / 3, "recall_at_k": 1.0, "mrr_at_k": 1.0},
        "generation_summary": {"fact_coverage": 1.0, "grounded_fact_coverage": 1.0,
                               "citation_precision": 1.0, "abstention_accuracy": 1.0},
    }


def test_gate_passes_complete_clean_run() -> None:
    validate_promotion(_run(), _baseline())


def test_promoted_baseline_is_bound_to_repository_dataset() -> None:
    dataset = load_cases("data/evaluation/raspberry_pi_demo_v1.json")
    baseline = json.loads(
        Path("data/evaluation/baselines/raspberry_pi_demo_v1.json").read_text()
    )

    assert baseline["schema_version"] == 1
    assert baseline["report_schema_version"] == 1
    assert baseline["dataset_sha256"] == dataset.content_sha256
    assert baseline["case_ids"] == [case.id for case in dataset.cases]


def _empty(run: dict[str, object]) -> None:
    run["traces"] = []


def _missing(run: dict[str, object]) -> None:
    cast_traces(run).pop()


def _duplicate(run: dict[str, object]) -> None:
    cast_traces(run)[1] = copy.deepcopy(cast_traces(run)[0])


def _wrong_hash(run: dict[str, object]) -> None:
    run["metadata"] = {"dataset_sha256": "b" * 64}


def _wrong_identity(run: dict[str, object]) -> None:
    cast_traces(run)[0]["generation"]["case_id"] = "other"  # type: ignore[index]


def cast_traces(run: dict[str, object]) -> list[dict[str, object]]:
    return run["traces"]  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [(_empty, "cannot be empty"), (_missing, "exactly the expected"),
     (_duplicate, "missing, duplicate, or reordered"), (_wrong_hash, "different dataset content"),
     (_wrong_identity, "mismatched case identities")],
)
def test_gate_rejects_incomplete_or_mismatched_evidence(
    mutate: Callable[[dict[str, object]], None], message: str
) -> None:
    run = _run()
    mutate(run)
    with pytest.raises(ValueError, match=message):
        validate_promotion(run, _baseline())


def test_gate_recalculates_aggregates_and_rejects_metric_tampering() -> None:
    run = _run()
    run["report"]["retrieval_summary"]["recall_at_k"] = 0.99  # type: ignore[index]
    with pytest.raises(ValueError, match="does not match trace results"):
        validate_promotion(run, _baseline())
    run = _run()
    run["report"]["retrieval_summary"]["invented"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="invalid metric keys"):
        validate_promotion(run, _baseline())


def test_gate_blocks_hard_failure_and_trace_regression() -> None:
    failed = _run()
    cast_traces(failed)[0]["generation"]["abstention_correct"] = False  # type: ignore[index]
    failed["report"]["generation"][0]["abstention_correct"] = False  # type: ignore[index]
    failed["report"]["generation_summary"]["abstention_accuracy"] = 0.5  # type: ignore[index]
    with pytest.raises(ValueError, match="wrong abstention"):
        validate_promotion(failed, _baseline())
    regressed = _run()
    cast_traces(regressed)[0]["retrieval"]["recall_at_k"] = 0.9  # type: ignore[index]
    regressed["report"]["retrieval"][0]["recall_at_k"] = 0.9  # type: ignore[index]
    regressed["report"]["retrieval_summary"]["recall_at_k"] = 0.9  # type: ignore[index]
    with pytest.raises(ValueError, match="regression"):
        validate_promotion(regressed, _baseline(), tolerance=0.05)


def test_gate_cli_reads_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_path, baseline_path = tmp_path / "run.json", tmp_path / "baseline.json"
    run_path.write_text(json.dumps(_run()))
    baseline_path.write_text(json.dumps(_baseline()))
    assert main([str(run_path), str(baseline_path)]) == 0
    assert "passed" in capsys.readouterr().out
