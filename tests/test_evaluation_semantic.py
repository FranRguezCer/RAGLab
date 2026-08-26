from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from raglab.evaluation.application import EvaluationApplication, HermeticEvaluationExecutor
from raglab.evaluation.manifest import load_manifest
from raglab.evaluation.models import EvaluationManifest
from raglab.evaluation.semantic import (
    SEMANTIC_SNAPSHOT_FILES,
    NLIScores,
    TransformersNLIScorer,
    calibrate,
    load_calibration_fixture,
)


def _enabled_manifest() -> EvaluationManifest:
    manifest = load_manifest()
    assert manifest.semantic is not None
    calibration = replace(
        manifest.semantic.calibration,
        threshold=0.8,
        contradiction_threshold=0.8,
        fingerprint="locked-calibration",
    )
    return replace(
        manifest, semantic=replace(manifest.semantic, enabled=True, calibration=calibration)
    )


class _Scores:
    def __init__(self, scores: list[NLIScores | None] | list[NLIScores]) -> None:
        self.scores: list[NLIScores | None] = list(scores)
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs):  # type: ignore[no-untyped-def]
        self.calls.append(list(pairs))
        result = self.scores[: len(pairs)]
        del self.scores[: len(pairs)]
        return result


def test_transformers_scorer_resolves_only_pinned_safetensors_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = load_manifest()
    assert manifest.semantic is not None
    snapshot_calls: list[dict[str, object]] = []
    load_calls: list[tuple[str, dict[str, object]]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        snapshot_calls.append(kwargs)
        return str(tmp_path)

    class FakeModel:
        class Config:
            id2label = {2: "neutral", 0: "contradiction", 1: "entailment"}

        config = Config()

        def to(self, device: str):  # type: ignore[no-untyped-def]
            assert device == "cpu"
            return self

        def eval(self) -> None:
            return None

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            load_calls.append((path, kwargs))
            return object()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            load_calls.append((path, kwargs))
            return FakeModel()

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizerFactory)
    monkeypatch.setattr("transformers.AutoModelForSequenceClassification", FakeModelFactory)
    scorer = TransformersNLIScorer(manifest.semantic)
    scorer._load()

    assert snapshot_calls == [
        {
            "repo_id": manifest.semantic.model.name,
            "revision": manifest.semantic.model.revision,
            "local_files_only": True,
            "allow_patterns": list(SEMANTIC_SNAPSHOT_FILES),
        }
    ]
    assert load_calls == [
        (str(tmp_path), {"local_files_only": True}),
        (str(tmp_path), {"local_files_only": True, "use_safetensors": True}),
    ]
    assert scorer._entailment_index == 1
    assert scorer._contradiction_index == 0


def test_v2_fixture_has_locked_routing_shape_without_duplicated_claims() -> None:
    fixture = load_calibration_fixture()
    expected = {
        "calibration": {
            "semantic_positive": 48,
            "semantic_negative": 32,
            "lexical_valid": 16,
            "lexical_neutral": 16,
            "lexical_contradiction": 16,
        },
        "holdout": {
            "semantic_positive": 32,
            "semantic_negative": 32,
            "lexical_valid": 16,
            "lexical_neutral": 16,
            "lexical_contradiction": 16,
        },
    }
    for split, counts in expected.items():
        rows = fixture[split]
        assert {label: sum(row["label"] == label for row in rows) for label in counts} == counts
        assert all(set(row) == {"fact_id", "answer", "label"} for row in rows)
        assert all(row["fact_id"] not in row["answer"] for row in rows)


def test_calibration_derives_independent_strict_boundaries_and_enforces_gates() -> None:
    manifest = load_manifest()
    assert manifest.semantic is not None
    fixture = load_calibration_fixture()
    calibration = [
        NLIScores(0.9, 0.01)
        if r["label"] == "semantic_positive"
        else NLIScores(0.1, 0.9 if r["label"] == "lexical_contradiction" else 0.1)
        for r in fixture["calibration"]
    ]
    holdout = [
        NLIScores(0.9, 0.01)
        if r["label"] == "semantic_positive"
        else NLIScores(0.1, 0.9 if r["label"] == "lexical_contradiction" else 0.1)
        for r in fixture["holdout"]
    ]
    frozen = [NLIScores(0.9, 0.01)] * 12 + [NLIScores(0.1, 0.01)] * 6
    result = calibrate(manifest.semantic, _Scores(calibration + holdout + frozen), manifest)

    assert result.enabled is True
    assert result.threshold == math.nextafter(0.1, math.inf)
    assert result.contradiction_threshold == math.nextafter(0.1, math.inf)
    assert result.calibration_valid_promotions == 48
    assert result.holdout_valid_promotions == 32
    assert result.calibration_contradiction_vetoes == 16
    assert result.holdout_contradiction_vetoes == 16
    assert (result.frozen_rescues, result.frozen_rejections) == (12, 6)


def test_semantic_grading_rescues_miss_and_vetoes_lexical_contradiction() -> None:
    class Answers(HermeticEvaluationExecutor):
        def generate(self, case, *, collection):  # type: ignore[no-untyped-def]
            observation = super().generate(case, collection=collection)
            if case.id == "aster-low-flow":
                return replace(observation, answer="Insufficient flow triggers error code E17.")
            return observation

    scorer = _Scores([NLIScores(0.91, 0.01)] * 100)
    run = EvaluationApplication(
        Answers(),
        semantic_scorer=scorer,
        metadata_provider=lambda: {"dirty": False, "hardware_fingerprint": "test"},
    ).run(_enabled_manifest(), persist=False)
    grade = run["cases"][0]["generation"]["fact_grading"][0][0]
    assert grade["lexical"] is False
    assert grade["semantic"] is True
    assert grade["contradiction_veto"] is False
    assert grade["final"] is True
    assert grade["contradiction_score"] == 0.01

    application = EvaluationApplication(
        HermeticEvaluationExecutor(), semantic_scorer=_Scores([NLIScores(0.01, 0.99)])
    )
    manifest = _enabled_manifest()
    application._active_manifest = manifest
    case = manifest.cases[0]
    observation = HermeticEvaluationExecutor().generate(case, collection="unused")
    checks, grading = application._generation_checks(
        case, observation, {"hard": [], "advisory": []}
    )
    assert grading[0]["lexical"] is True
    assert grading[0]["contradiction_veto"] is True
    assert checks[case.required_facts[0].id] is False


def test_enabled_semantic_grading_fails_closed_when_pair_is_unscorable() -> None:
    application = EvaluationApplication(
        HermeticEvaluationExecutor(), semantic_scorer=_Scores([None])
    )
    manifest = _enabled_manifest()
    application._active_manifest = manifest
    case = manifest.cases[0]
    observation = HermeticEvaluationExecutor().generate(case, collection="unused")
    checks, grading = application._generation_checks(
        case, observation, {"hard": [], "advisory": []}
    )
    assert checks[case.required_facts[0].id] is False
    assert grading[0]["final"] is False


def _passing_calibration_scores() -> tuple[
    dict[str, Any], list[NLIScores], list[NLIScores]
]:
    fixture = load_calibration_fixture()
    calibration = [
        NLIScores(0.9, 0.01)
        if row["label"] == "semantic_positive"
        else NLIScores(0.1, 0.9 if row["label"] == "lexical_contradiction" else 0.1)
        for row in fixture["calibration"]
    ]
    holdout = [
        NLIScores(0.9, 0.01)
        if row["label"] == "semantic_positive"
        else NLIScores(0.1, 0.9 if row["label"] == "lexical_contradiction" else 0.1)
        for row in fixture["holdout"]
    ]
    return fixture, calibration, holdout


def test_transformers_scorer_batches_and_returns_both_probabilities_once() -> None:
    from types import SimpleNamespace

    import torch

    manifest = load_manifest()
    assert manifest.semantic is not None
    scorer = TransformersNLIScorer(manifest.semantic)
    batch_sizes: list[int] = []

    class Tokenizer:
        def __call__(self, premises, hypotheses, **kwargs):  # type: ignore[no-untyped-def]
            del hypotheses
            if isinstance(premises, str):
                return {"input_ids": [0] * (513 if premises == "too long" else 3)}
            batch_sizes.append(len(premises))
            return {"input_ids": torch.zeros((len(premises), 3), dtype=torch.long)}

    class Model:
        def __call__(self, **encoded):  # type: ignore[no-untyped-def]
            size = encoded["input_ids"].shape[0]
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.0, 4.0]] * size))

    scorer._tokenizer = Tokenizer()
    scorer._model = Model()
    scorer._torch = torch
    scorer._contradiction_index = 0
    scorer._entailment_index = 2
    pairs = [("short", "claim")] * 9 + [("too long", "claim")]

    scores = scorer.score(pairs)

    assert batch_sizes == [8, 1]
    assert scores[-1] is None
    assert all(score is not None for score in scores[:-1])
    assert scores[0] is not None
    assert scores[0].entailment > scores[0].contradiction


@pytest.mark.parametrize(
    "failure",
    [
        "calibration_recall",
        "holdout_recall",
        "holdout_false_promotion",
        "holdout_false_veto",
        "calibration_missed_contradiction",
        "holdout_missed_contradiction",
        "frozen_mismatch",
    ],
)
def test_calibration_disables_for_each_independent_gate(failure: str) -> None:
    manifest = load_manifest()
    assert manifest.semantic is not None
    fixture, calibration, holdout = _passing_calibration_scores()
    if failure == "calibration_recall":
        indexes = [
            i for i, row in enumerate(fixture["calibration"]) if row["label"] == "semantic_positive"
        ][:10]
        for index in indexes:
            calibration[index] = NLIScores(0.05, 0.01)
    elif failure == "holdout_recall":
        indexes = [
            i for i, row in enumerate(fixture["holdout"]) if row["label"] == "semantic_positive"
        ][:7]
        for index in indexes:
            holdout[index] = NLIScores(0.05, 0.01)
    elif failure == "holdout_false_promotion":
        index = next(
            i for i, row in enumerate(fixture["holdout"]) if row["label"] == "semantic_negative"
        )
        holdout[index] = NLIScores(0.9, 0.01)
    elif failure == "holdout_false_veto":
        index = next(
            i for i, row in enumerate(fixture["holdout"]) if row["label"] == "lexical_valid"
        )
        holdout[index] = NLIScores(0.1, 0.9)
    elif failure == "calibration_missed_contradiction":
        index = next(
            i
            for i, row in enumerate(fixture["calibration"])
            if row["label"] == "lexical_contradiction"
        )
        calibration[index] = NLIScores(0.1, 0.1)
    elif failure == "holdout_missed_contradiction":
        index = next(
            i for i, row in enumerate(fixture["holdout"]) if row["label"] == "lexical_contradiction"
        )
        holdout[index] = NLIScores(0.1, 0.1)
    frozen_rescues = 11 if failure == "frozen_mismatch" else 12
    frozen = [NLIScores(0.9, 0.01)] * frozen_rescues + [NLIScores(0.1, 0.01)] * (
        18 - frozen_rescues
    )

    result = calibrate(manifest.semantic, _Scores(calibration + holdout + frozen), manifest)

    assert result.enabled is False
    assert result.reason is not None


def test_application_preserves_lexical_hit_below_veto_and_records_v5_audit() -> None:
    scorer = _Scores([NLIScores(0.2, 0.1)] * 100)
    run = EvaluationApplication(
        HermeticEvaluationExecutor(),
        semantic_scorer=scorer,
        metadata_provider=lambda: {"dirty": False, "hardware_fingerprint": "test"},
    ).run(_enabled_manifest(), persist=False)
    grade = run["cases"][0]["generation"]["fact_grading"][0][0]

    assert run["schema_version"] == 5
    assert grade["lexical"] is True
    assert grade["contradiction_veto"] is False
    assert grade["final"] is True
    assert grade["model"] == "tasksource/deberta-small-long-nli"
    assert len(grade["revision"]) == 40
    assert len(grade["fingerprint"]) == 64
    assert "generation_semantic_contradiction_veto_rate" in run["summary"]["quality"]


def test_application_fails_closed_when_semantic_model_raises_and_skips_abstentions() -> None:
    class Raising:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, pairs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("model unavailable")

    scorer = Raising()
    run = EvaluationApplication(HermeticEvaluationExecutor(), semantic_scorer=scorer).run(
        _enabled_manifest(), persist=False
    )
    first = run["cases"][0]["generation"]["fact_grading"][0][0]
    abstention = next(
        case for case in run["cases"] if case["id"] == "unsupported-password-rotation"
    )

    assert first["final"] is False
    assert run["errors"]["advisory"]
    assert abstention["generation"]["fact_grading"] == [[], [], []]

    abstention_case = next(case for case in _enabled_manifest().cases if case.should_abstain)
    application = EvaluationApplication(HermeticEvaluationExecutor(), semantic_scorer=scorer)
    application._active_manifest = _enabled_manifest()
    before = scorer.calls
    observation = HermeticEvaluationExecutor().generate(abstention_case, collection="unused")
    application._generation_checks(abstention_case, observation, {"hard": [], "advisory": []})
    assert scorer.calls == before
