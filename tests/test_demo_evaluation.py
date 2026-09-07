from __future__ import annotations

from pathlib import Path

from raglab.evaluation import GenerationOutput, evaluate_generation, load_cases


def test_raspberry_pi_dataset_routes_two_cases_per_collection() -> None:
    dataset = load_cases(Path("data/evaluation/raspberry_pi_demo_v1.json"))

    assert len(dataset.cases) == 6
    assert {name: sum(case.collection == name for case in dataset.cases) for name in {
        "rpi-computers",
        "rpi-microcontrollers",
        "rpi-camera-ai",
    }} == {
        "rpi-computers": 2,
        "rpi-microcontrollers": 2,
        "rpi-camera-ai": 2,
    }
    assert sum(case.expected_outcome == "abstain" for case in dataset.cases) == 3


def test_expected_abstention_without_citations_is_fully_correct() -> None:
    case = next(
        case
        for case in load_cases("data/evaluation/raspberry_pi_demo_v1.json").cases
        if case.expected_outcome == "abstain"
    )

    (result,) = evaluate_generation(
        (case,), {case.id: GenerationOutput("I cannot answer from this corpus.", (), True)}
    )

    assert result.abstention_correct is True
    assert result.citation_precision == 1.0
    assert result.fact_coverage == 1.0
