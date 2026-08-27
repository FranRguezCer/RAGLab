from __future__ import annotations

from dataclasses import replace

import pytest

from raglab.evaluation import (
    GenerationCase,
    GenerationEvidence,
    GenerationOutput,
    evaluate_generation,
)
from raglab.generation import GenerationMetrics, GenerationResponse, GenerationStrategy
from raglab.retrieval import RetrievalResponse


def _case(**changes: object) -> GenerationCase:
    case = GenerationCase(
        question="What should the operator record?",
        evidence=(GenerationEvidence("aster-manual", "Record fault E41."),),
        required_phrases=("record fault E41",),
        forbidden_phrases=("fault raised for low flow is E17",),
        expected_source_ids=("aster-manual",),
        expected_model_calls=2,
    )
    return replace(case, **changes)


def _output(**changes: object) -> GenerationOutput:
    output = GenerationOutput(
        answer="The operator must record fault E41.",
        abstained=False,
        used_source_ids=("aster-manual",),
        model_calls=2,
    )
    return replace(output, **changes)


def _checks(output: GenerationOutput, case: GenerationCase | None = None) -> dict[str, object]:
    result = evaluate_generation(case or _case(), output)
    return {check.name: check for check in result.checks}


def test_every_check_is_visible_and_all_must_pass() -> None:
    result = evaluate_generation(_case(), _output())

    assert result.status == "PASS"
    assert result.passed is True
    assert [check.name for check in result.checks] == [
        "required_phrases",
        "forbidden_phrases",
        "abstained",
        "used_source_ids",
        "model_calls",
    ]
    assert all(check.status == "PASS" and check.passed for check in result.checks)


def test_required_phrases_use_existing_deterministic_normalization() -> None:
    case = _case(required_phrases=("record fault E41", "check the isolation banner"))
    check = _checks(
        _output(answer="**CHECK** the isolation banner, then record fault `Ｅ４１`!"), case
    )["required_phrases"]

    assert check.expected == ("record fault E41", "check the isolation banner")
    assert check.observed == ("record fault E41", "check the isolation banner")
    assert check.status == "PASS"


def test_missing_required_phrase_fails_only_its_check() -> None:
    result = evaluate_generation(_case(), _output(answer="Inspect the isolation banner."))

    failed = [check.name for check in result.checks if not check.passed]
    assert result.status == "FAIL"
    assert failed == ["required_phrases"]


def test_present_forbidden_phrase_is_reported_after_normalization() -> None:
    output = _output(
        answer="Record fault E41. The fault raised for LOW-FLOW is `Ｅ１７`."
    )
    check = _checks(output)["forbidden_phrases"]

    assert check.expected == ()
    assert check.observed == ("fault raised for low flow is E17",)
    assert check.status == "FAIL"


def test_abstention_must_match_exactly() -> None:
    check = _checks(_output(abstained=True))["abstained"]

    assert check.expected is False
    assert check.observed is True
    assert check.status == "FAIL"


def test_used_sources_are_compared_as_an_exact_order_independent_set() -> None:
    case = _case(
        evidence=(
            GenerationEvidence("source-a", "Alpha."),
            GenerationEvidence("source-b", "Beta."),
        ),
        expected_source_ids=("source-a", "source-b"),
    )
    output = _output(used_source_ids=("source-b", "source-a"))

    assert _checks(output, case)["used_source_ids"].status == "PASS"

    extra_case = replace(
        case,
        evidence=(*case.evidence, GenerationEvidence("source-c", "Gamma.")),
    )
    extra = _checks(
        replace(output, used_source_ids=("source-a", "source-b", "source-c")), extra_case
    )
    assert extra["used_source_ids"].status == "FAIL"


def test_model_calls_must_match_exactly() -> None:
    check = _checks(_output(model_calls=3))["model_calls"]

    assert check.expected == 2
    assert check.observed == 3
    assert check.status == "FAIL"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"question": " "}, "question"),
        ({"evidence": ()}, "evidence"),
        ({"expected_model_calls": -1}, "expected_model_calls"),
        ({"expected_source_ids": ("aster-manual", "aster-manual")}, "unique"),
        ({"expected_source_ids": ("unknown",)}, "not present"),
        ({"required_phrases": ("!!!",)}, "normalize to an empty"),
        ({"required_phrases": ("E41", "`Ｅ４１`")}, "normalization"),
        ({"required_phrases": ("E41",), "forbidden_phrases": ("`Ｅ４１`",)}, "both"),
    ],
)
def test_generation_case_validates_ambiguous_contracts(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_case(), **changes)


def test_duplicate_evidence_and_output_sources_are_rejected() -> None:
    with pytest.raises(ValueError, match="evidence source_ids"):
        GenerationCase(
            "question",
            (GenerationEvidence("source", "text"), GenerationEvidence("source", "other")),
        )
    with pytest.raises(ValueError, match="used_source_ids"):
        GenerationOutput("answer", False, ("source", "source"), 1)


def test_unknown_observed_source_is_rejected_before_scoring() -> None:
    with pytest.raises(ValueError, match="not present"):
        evaluate_generation(_case(), _output(used_source_ids=("unknown",)))


def test_generation_output_validates_calls_and_source_ids() -> None:
    with pytest.raises(ValueError, match="model_calls"):
        GenerationOutput("answer", False, (), -1)
    with pytest.raises(ValueError, match="source_id"):
        GenerationOutput("answer", False, (" ",), 0)


def test_generation_response_adapter_uses_used_source_ids_and_metrics() -> None:
    retrieval = RetrievalResponse("question", None, ("question",), (), ())
    response = GenerationResponse(
        answer="Grounded answer.",
        abstained=False,
        sources=(),
        retrieval=retrieval,
        strategy=GenerationStrategy.HIERARCHICAL,
        source_shortfall=False,
        minimum_sources=1,
        source_count=2,
        metrics=GenerationMetrics(3, 120, 100, 20),
        source_ids=("S2",),
    )

    assert GenerationOutput.from_generation_response(response) == GenerationOutput(
        "Grounded answer.", False, ("S2",), 3
    )


def test_three_recorded_failures_are_explained_and_corrected() -> None:
    cases_and_outputs = (
        (
            _case(),
            _output(answer="The fault raised for low flow is E17."),
            _output(),
            {"required_phrases", "forbidden_phrases"},
        ),
        (
            GenerationCase(
                "How often must Meridian employees rotate their passwords?",
                (
                    GenerationEvidence(
                        "meridian-handbook", "No password rotation policy is stated."
                    ),
                ),
                expected_abstained=True,
                expected_model_calls=1,
            ),
            GenerationOutput("Rotate passwords every 90 days.", False, ("meridian-handbook",), 2),
            GenerationOutput("The evidence does not specify a rotation schedule.", True, (), 1),
            {"abstained", "used_source_ids", "model_calls"},
        ),
        (
            GenerationCase(
                "Which proposed labels must be removed?",
                (
                    GenerationEvidence(
                        "orbit-telemetry",
                        "Remove user ID and raw request path from the proposed labels.",
                    ),
                ),
                required_phrases=("user ID", "raw request path"),
                forbidden_phrases=("region", "session ID"),
                expected_source_ids=("orbit-telemetry",),
                expected_model_calls=2,
            ),
            GenerationOutput(
                "Remove region, user ID, session ID, and raw request path.",
                False,
                ("orbit-telemetry",),
                2,
            ),
            GenerationOutput(
                "Remove user ID and raw request path.", False, ("orbit-telemetry",), 2
            ),
            {"forbidden_phrases"},
        ),
    )

    for case, wrong, correct, expected_failures in cases_and_outputs:
        wrong_result = evaluate_generation(case, wrong)
        actual_failures = {check.name for check in wrong_result.checks if not check.passed}
        assert actual_failures == expected_failures
        assert wrong_result.status == "FAIL"
        assert evaluate_generation(case, correct).status == "PASS"
