from __future__ import annotations

from dataclasses import replace

from raglab.evaluation.application import EvaluationApplication, HermeticEvaluationExecutor
from raglab.evaluation.manifest import load_manifest
from raglab.evaluation.models import GenerationObservation
from raglab.evaluation.semantic import calibrate, load_calibration_fixture


def _enabled_manifest():  # type: ignore[no-untyped-def]
    manifest = load_manifest()
    assert manifest.semantic is not None
    semantic = replace(
        manifest.semantic,
        enabled=True,
        calibration=replace(
            manifest.semantic.calibration,
            threshold=0.8,
            fingerprint="locked-calibration",
        ),
    )
    return replace(manifest, semantic=semantic)


class _Scores:
    def __init__(self, scores: list[float | None]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs):  # type: ignore[no-untyped-def]
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


def test_semantic_rescue_is_lexical_first_and_records_v4_fact_audit() -> None:
    class ParaphrasingExecutor(HermeticEvaluationExecutor):
        def generate(self, case, *, collection):  # type: ignore[no-untyped-def]
            observation = super().generate(case, collection=collection)
            if case.id == "aster-low-flow":
                return replace(
                    observation,
                    answer=(
                        "Insufficient flow causes the controller to report the configured fault."
                    ),
                )
            return observation

    scorer = _Scores([0.91])
    run = EvaluationApplication(
        ParaphrasingExecutor(),
        semantic_scorer=scorer,
        metadata_provider=lambda: {"dirty": False, "hardware_fingerprint": "test"},
    ).run(_enabled_manifest(), persist=False)

    grading = run["cases"][0]["generation"]["fact_grading"][0][0]
    assert grading["lexical"] is False
    assert grading["semantic"] is True
    assert grading["final"] is True
    assert grading["semantic_score"] == 0.91
    assert grading["model"] == "cross-encoder/nli-deberta-v3-small"
    assert len(grading["revision"]) == 40
    assert len(grading["fingerprint"]) == 64
    assert run["summary"]["quality"]["generation_semantic_rescue_rate"] > 0
    assert run["errors"]["hard"] == []
    assert len(scorer.calls) == 3


def test_semantic_rescue_never_runs_for_lexical_hits_or_abstentions() -> None:
    scorer = _Scores([0.99])
    run = EvaluationApplication(
        HermeticEvaluationExecutor(),
        semantic_scorer=scorer,
        metadata_provider=lambda: {"dirty": False, "hardware_fingerprint": "test"},
    ).run(_enabled_manifest(), persist=False)

    assert scorer.calls == []
    abstention = next(
        case for case in run["cases"] if case["id"] == "unsupported-password-rotation"
    )
    assert abstention["generation"]["fact_grading"] == [[], [], []]


def test_unscorable_semantic_pair_fails_closed_without_changing_other_checks() -> None:
    application = EvaluationApplication(
        HermeticEvaluationExecutor(), semantic_scorer=_Scores([None])
    )
    manifest = _enabled_manifest()
    application._active_manifest = manifest
    case = manifest.cases[0]
    checks, grading = application._generation_checks(
        case,
        GenerationObservation(
            "A vague description without the required fact.",
            False,
            case.expected_source_ids,
            1,
            1,
            1,
            1.0,
        ),
        {"hard": [], "advisory": []},
    )

    assert checks[case.required_facts[0].id] is False
    assert checks["abstention"] is True
    assert checks["citations"] is True
    assert grading[0]["semantic"] is None
    assert grading[0]["final"] is False


def test_locked_calibration_fixture_has_expected_per_fact_shape() -> None:
    fixture = load_calibration_fixture()
    calibration = fixture["calibration"]
    holdout = fixture["holdout"]
    fact_ids = {row["fact_id"] for row in calibration}

    assert len(fact_ids) == 16
    assert len(calibration) == 96
    assert len(holdout) == 32
    for fact_id in fact_ids:
        labels = [row["label"] for row in calibration if row["fact_id"] == fact_id]
        assert labels.count("valid") == 3
        assert labels.count("neutral") == 2
        assert labels.count("contradiction") == 1


def test_calibration_selects_smallest_zero_false_promotion_threshold() -> None:
    manifest = load_manifest()
    assert manifest.semantic is not None
    fixture = load_calibration_fixture()
    calibration_scores = [
        0.7 + index * 0.001 if row["label"] == "valid" else 0.6
        for index, row in enumerate(fixture["calibration"])
    ]
    holdout_scores = [
        0.8 if row["label"] == "valid" else 0.5 for row in fixture["holdout"]
    ]
    class SequentialScores(_Scores):
        def score(self, pairs):  # type: ignore[no-untyped-def]
            result = self.scores[: len(pairs)]
            del self.scores[: len(pairs)]
            return result

    result = calibrate(
        manifest.semantic, SequentialScores(calibration_scores + holdout_scores)
    )

    assert result.enabled is True
    assert result.threshold == min(
        score
        for score, row in zip(calibration_scores, fixture["calibration"], strict=True)
        if row["label"] == "valid"
    )
    assert result.false_promotions == 0


def test_calibration_disables_when_zero_false_promotion_is_impossible() -> None:
    manifest = load_manifest()
    assert manifest.semantic is not None
    result = calibrate(manifest.semantic, _Scores([0.5] * 128))

    assert result.enabled is False
    assert result.threshold is None
    assert "threshold" in (result.reason or "")
