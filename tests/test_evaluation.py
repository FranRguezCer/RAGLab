from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.contracts import Citation, ProvenanceStatus
from raglab.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    ExpectedFact,
    GenerationOutput,
    RetrievalOutput,
    adapt_generation_response,
    adapt_retrieval_response,
    build_report,
    citation_evidence_id,
    compare_reports,
    evaluate_generation,
    evaluate_retrieval,
    load_cases,
    normalize_text,
)
from raglab.generation import (
    GeneratedSource,
    GenerationMetrics,
    GenerationResponse,
    GenerationStrategy,
)
from raglab.retrieval import RankingTrace, RetrievalResponse, RetrievalResult

DATASET = Path("data/evaluation/aster_greenhouse_controller_v1.json")


def _case() -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        question="What happens?",
        expected_facts=(
            ExpectedFact("f1", "Alpha happens.", ("alpha happens",), ("s1",)),
            ExpectedFact("f2", "Beta follows.", ("beta follows", "then beta"), ("s2",)),
        ),
        forbidden_phrases=("invented action",),
    )


def _citation(name: str, *heading_path: str) -> Citation:
    return Citation(
        source_uri=f"memory://{name}",
        source_name=name,
        title=None,
        heading_path=heading_path,
        start_page=None,
        end_page=None,
        start_line=None,
        end_line=None,
        provenance_status=ProvenanceStatus.COMPLETE,
    )


def _retrieval_result(source_id: str, citation: Citation | None = None) -> RetrievalResult:
    return RetrievalResult(
        id=source_id,
        document_id=f"doc-{source_id}",
        content=source_id,
        citation=citation or _citation(source_id),
        matched_chunk_ids=(source_id,),
        first_chunk_index=0,
        last_chunk_index=0,
        trace=RankingTrace(None, None, None, None, 1.0, None, None),
    )


def test_loads_four_strict_aster_cases() -> None:
    dataset = load_cases(DATASET)

    assert dataset.schema_version == 1
    assert dataset.dataset_id == "aster-greenhouse-controller-v1"
    assert len(dataset.cases) == 4
    assert all(case.expected_facts for case in dataset.cases)
    assert all(fact.answer_variants for case in dataset.cases for fact in case.expected_facts)
    assert all(fact.evidence_source_ids for case in dataset.cases for fact in case.expected_facts)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=2), "unsupported dataset schema"),
        (lambda value: value.update(extra=True), "invalid fields"),
        (lambda value: value.update(cases=[]), "cannot be empty"),
        (
            lambda value: value["cases"].append(value["cases"][0]),
            "case IDs must be unique",
        ),
        (
            lambda value: value["cases"][0]["expected_facts"][0].update(
                evidence_source_ids=[]
            ),
            "requires answer variants and evidence source IDs",
        ),
    ],
)
def test_rejects_invalid_dataset(
    tmp_path: Path, mutation: object, message: str
) -> None:
    raw = json.loads(DATASET.read_text())
    mutation(raw)  # type: ignore[operator]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=message):
        load_cases(path)


def test_normalization_is_case_punctuation_unicode_and_whitespace_insensitive() -> None:
    assert normalize_text("  SKIPPED_WET_BED — É41  ") == "skipped_wet_bed é41"
    assert normalize_text("ＡＬＰＨＡ\n happens!") == "alpha happens"


def test_citation_evidence_identity_is_stable_and_human_readable() -> None:
    citation = _citation(
        "aster_greenhouse_controller_manual.md",
        "Aster Greenhouse Controller: Operator Manual",
        "3. Climate targets and irrigation rules",
        "Irrigation schedule",
    )

    assert citation_evidence_id(citation) == (
        "aster-greenhouse-controller-manual-md/"
        "aster-greenhouse-controller-operator-manual/"
        "3-climate-targets-and-irrigation-rules/irrigation-schedule"
    )


def test_retrieval_metrics_expose_ranks_and_evidence_outside_k() -> None:
    case = _case()
    (result,) = evaluate_retrieval(
        (case,),
        {case.id: RetrievalOutput(("noise", "s1", "more-noise", "s2"))},
        top_k=3,
    )

    assert result.ranked_source_ids == ("noise", "s1", "more-noise")
    assert result.relevant_positions == {"s1": 2, "s2": 4}
    assert result.retrieved_relevant_source_ids == ("s1",)
    assert result.missed_relevant_source_ids == ("s2",)
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == 0.5
    assert result.mrr_at_k == 0.5


def test_retrieval_regression_with_only_irrelevant_results_scores_zero() -> None:
    (result,) = evaluate_retrieval(
        (_case(),), {"case-1": RetrievalOutput(("noise-1", "noise-2"))}, top_k=2
    )

    assert (result.precision_at_k, result.recall_at_k, result.mrr_at_k) == (0.0, 0.0, 0.0)
    assert result.relevant_positions == {"s1": None, "s2": None}


def test_retrieval_rejects_invalid_top_k_case_sets_and_duplicate_rankings() -> None:
    case = _case()
    with pytest.raises(ValueError, match="positive"):
        evaluate_retrieval((case,), {case.id: RetrievalOutput(())}, top_k=0)
    with pytest.raises(ValueError, match="case IDs"):
        evaluate_retrieval((case,), {}, top_k=1)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_retrieval((case,), {case.id: RetrievalOutput(("s1", "s1"))}, top_k=2)


def test_generation_metrics_expose_omissions_invalid_citations_and_forbidden_phrases() -> None:
    (result,) = evaluate_generation(
        (_case(),),
        {
            "case-1": GenerationOutput(
                "ALPHA happens. Take the invented action.", ("s1", "unrelated")
            )
        },
    )

    assert result.found_fact_ids == ("f1",)
    assert result.missed_fact_ids == ("f2",)
    assert result.grounded_fact_ids == ("f1",)
    assert result.ungrounded_fact_ids == ()
    assert result.valid_citation_source_ids == ("s1",)
    assert result.invalid_citation_source_ids == ("unrelated",)
    assert result.forbidden_phrase_hits == ("invented action",)
    assert result.fact_coverage == 0.5
    assert result.grounded_fact_coverage == 0.5
    assert result.citation_precision == 0.5


def test_found_fact_without_its_evidence_is_ungrounded() -> None:
    (result,) = evaluate_generation(
        (_case(),), {"case-1": GenerationOutput("Alpha happens and then beta.", ("s1",))}
    )

    assert result.found_fact_ids == ("f1", "f2")
    assert result.grounded_fact_ids == ("f1",)
    assert result.ungrounded_fact_ids == ("f2",)
    assert result.fact_coverage == 1.0
    assert result.grounded_fact_coverage == 0.5
    assert result.citation_precision == 1.0


def test_no_citations_have_zero_citation_precision() -> None:
    (result,) = evaluate_generation(
        (_case(),), {"case-1": GenerationOutput("Alpha happens.", ())}
    )

    assert result.fact_coverage == 0.5
    assert result.grounded_fact_coverage == 0.0
    assert result.citation_precision == 0.0


def test_contract_adapters_preserve_rank_and_resolve_generation_source_ids() -> None:
    irrigation = _citation(
        "aster_greenhouse_controller_manual.md",
        "Aster Greenhouse Controller: Operator Manual",
        "3. Climate targets and irrigation rules",
        "Irrigation schedule",
    )
    fault = _citation(
        "aster_greenhouse_controller_manual.md",
        "Aster Greenhouse Controller: Operator Manual",
        "5. Fault diagnosis",
    )
    irrigation_id = citation_evidence_id(irrigation)
    fault_id = citation_evidence_id(fault)
    retrieval = RetrievalResponse(
        query="question",
        rewritten_query=None,
        query_variants=("question",),
        filters=(),
        results=(
            _retrieval_result("dynamic-1", irrigation),
            _retrieval_result("dynamic-2", fault),
            _retrieval_result("dynamic-3", irrigation),
        ),
    )
    generated = GenerationResponse(
        answer="Alpha happens.",
        abstained=False,
        sources=(GeneratedSource("S1", "dynamic-2", "doc", fault),),
        retrieval=retrieval,
        strategy=GenerationStrategy.SINGLE_PASS,
        source_shortfall=False,
        minimum_sources=1,
        source_count=1,
        metrics=GenerationMetrics(1, 10, 10, 5),
        source_ids=("S1",),
    )

    assert adapt_retrieval_response(retrieval) == RetrievalOutput((irrigation_id, fault_id))
    assert adapt_generation_response(generated) == GenerationOutput("Alpha happens.", (fault_id,))


def test_generation_adapter_rejects_unknown_local_citation_id() -> None:
    response = GenerationResponse(
        answer="answer",
        abstained=False,
        sources=(),
        retrieval=RetrievalResponse("q", None, ("q",), (), ()),
        strategy=GenerationStrategy.SINGLE_PASS,
        source_shortfall=False,
        minimum_sources=1,
        source_count=0,
        metrics=GenerationMetrics(1, 1, None, None),
        source_ids=("S9",),
    )

    with pytest.raises(ValueError, match="unknown source IDs"):
        adapt_generation_response(response)


def _report(
    *, retrieval_ids: tuple[str, ...], answer: str, citations: tuple[str, ...], top_k: int = 2
):
    case = _case()
    dataset = EvaluationDataset(1, "dataset", (case,))
    retrieval = evaluate_retrieval(
        (case,), {case.id: RetrievalOutput(retrieval_ids)}, top_k=top_k
    )
    generation = evaluate_generation(
        (case,), {case.id: GenerationOutput(answer, citations)}
    )
    return build_report(dataset, retrieval, generation)


def test_report_serializes_traces_and_comparison_shows_previous_current_difference() -> None:
    previous = _report(retrieval_ids=("noise", "s1"), answer="Alpha happens.", citations=("s1",))
    current = _report(
        retrieval_ids=("s1", "s2"),
        answer="Alpha happens and then beta.",
        citations=("s1", "s2"),
    )

    serialized = json.loads(current.to_json())
    assert serialized["schema_version"] == 1
    assert serialized["retrieval"][0]["relevant_positions"] == {"s1": 1, "s2": 2}
    assert serialized["generation"][0]["found_fact_ids"] == ["f1", "f2"]

    comparison = compare_reports(previous, current)
    assert comparison.retrieval["recall_at_k"].previous == 0.5
    assert comparison.retrieval["recall_at_k"].current == 1.0
    assert comparison.retrieval["recall_at_k"].difference == 0.5
    assert comparison.generation["fact_coverage"].difference == 0.5


def test_compare_reports_rejects_incompatible_dataset_cases_top_k_and_version() -> None:
    report = _report(retrieval_ids=("s1",), answer="Alpha happens.", citations=("s1",))
    different_top_k = _report(
        retrieval_ids=("s1",), answer="Alpha happens.", citations=("s1",), top_k=1
    )
    with pytest.raises(ValueError, match="top_k"):
        compare_reports(report, different_top_k)

    different_dataset = EvaluationDataset(1, "other", (_case(),))
    other = build_report(different_dataset, report.retrieval, report.generation)
    with pytest.raises(ValueError, match="different datasets"):
        compare_reports(report, other)

    changed_case = EvaluationCase(
        "renamed", _case().question, _case().expected_facts, _case().forbidden_phrases
    )
    dataset = EvaluationDataset(1, "dataset", (changed_case,))
    renamed = build_report(
        dataset,
        evaluate_retrieval(
            (changed_case,), {"renamed": RetrievalOutput(("s1",))}, top_k=2
        ),
        evaluate_generation(
            (changed_case,), {"renamed": GenerationOutput("Alpha happens.", ("s1",))}
        ),
    )
    with pytest.raises(ValueError, match="different cases"):
        compare_reports(report, renamed)
