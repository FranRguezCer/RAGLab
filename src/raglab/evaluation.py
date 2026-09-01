"""Transparent, deterministic metrics for the first RAG evaluation dataset."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

from raglab.contracts import Citation
from raglab.generation import GenerationResponse
from raglab.retrieval import RetrievalResponse

REPORT_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1


def normalize_text(value: str) -> str:
    """Normalize text for visible, deterministic phrase matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def citation_evidence_id(citation: Citation) -> str:
    """Build a stable evidence identity from a citation's document and section."""

    components = (citation.source_name, *citation.heading_path)
    slugs = tuple(_slug(component) for component in components)
    if not slugs[0]:
        raise ValueError("citation source_name must contain letters or numbers")
    return "/".join(slug for slug in slugs if slug)


@dataclass(frozen=True, slots=True)
class ExpectedFact:
    id: str
    text: str
    answer_variants: tuple[str, ...]
    evidence_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    expected_facts: tuple[ExpectedFact, ...]
    forbidden_phrases: tuple[str, ...]

    @property
    def relevant_source_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                source_id
                for fact in self.expected_facts
                for source_id in fact.evidence_source_ids
            )
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    schema_version: int
    dataset_id: str
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    ranked_source_ids: tuple[str, ...]

    @classmethod
    def from_response(cls, response: RetrievalResponse) -> Self:
        return cls(
            tuple(
                dict.fromkeys(citation_evidence_id(result.citation) for result in response.results)
            )
        )


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    answer: str
    cited_source_ids: tuple[str, ...]

    @classmethod
    def from_response(cls, response: GenerationResponse) -> Self:
        local_source_ids = tuple(source.id for source in response.sources)
        if len(local_source_ids) != len(set(local_source_ids)):
            raise ValueError("generation response contains duplicate local source IDs")
        sources_by_id = {source.id: source for source in response.sources}
        cited_ids = response.source_ids or tuple(sources_by_id)
        if len(cited_ids) != len(set(cited_ids)):
            raise ValueError("generation response contains duplicate cited source IDs")
        unknown = [source_id for source_id in cited_ids if source_id not in sources_by_id]
        if unknown:
            raise ValueError(f"generation response cites unknown source IDs: {unknown}")
        return cls(
            response.answer,
            tuple(
                dict.fromkeys(
                    citation_evidence_id(sources_by_id[source_id].citation)
                    for source_id in cited_ids
                )
            ),
        )


def adapt_retrieval_response(response: RetrievalResponse) -> RetrievalOutput:
    return RetrievalOutput.from_response(response)


def adapt_generation_response(response: GenerationResponse) -> GenerationOutput:
    return GenerationOutput.from_response(response)


@dataclass(frozen=True, slots=True)
class RetrievalCaseResult:
    case_id: str
    top_k: int
    ranked_source_ids: tuple[str, ...]
    relevant_source_ids: tuple[str, ...]
    relevant_positions: Mapping[str, int | None]
    retrieved_relevant_source_ids: tuple[str, ...]
    missed_relevant_source_ids: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float


@dataclass(frozen=True, slots=True)
class GenerationCaseResult:
    case_id: str
    answer: str
    cited_source_ids: tuple[str, ...]
    found_fact_ids: tuple[str, ...]
    missed_fact_ids: tuple[str, ...]
    grounded_fact_ids: tuple[str, ...]
    ungrounded_fact_ids: tuple[str, ...]
    valid_citation_source_ids: tuple[str, ...]
    invalid_citation_source_ids: tuple[str, ...]
    forbidden_phrase_hits: tuple[str, ...]
    fact_coverage: float
    grounded_fact_coverage: float
    citation_precision: float


def evaluate_retrieval(
    cases: Sequence[EvaluationCase],
    outputs: Mapping[str, RetrievalOutput],
    *,
    top_k: int,
) -> tuple[RetrievalCaseResult, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    _validate_output_ids(cases, outputs, "retrieval")
    evaluated: list[RetrievalCaseResult] = []
    for case in cases:
        ranked = outputs[case.id].ranked_source_ids
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"retrieval output for {case.id!r} contains duplicate source IDs")
        relevant = case.relevant_source_ids
        limited = ranked[:top_k]
        retrieved = tuple(source_id for source_id in limited if source_id in relevant)
        positions = {
            source_id: (ranked.index(source_id) + 1 if source_id in ranked else None)
            for source_id in relevant
        }
        first_rank = next(
            (rank for rank, source_id in enumerate(limited, 1) if source_id in relevant), None
        )
        evaluated.append(
            RetrievalCaseResult(
                case_id=case.id,
                top_k=top_k,
                ranked_source_ids=limited,
                relevant_source_ids=relevant,
                relevant_positions=positions,
                retrieved_relevant_source_ids=retrieved,
                missed_relevant_source_ids=tuple(
                    source_id for source_id in relevant if source_id not in limited
                ),
                precision_at_k=len(retrieved) / top_k,
                recall_at_k=len(retrieved) / len(relevant),
                mrr_at_k=0.0 if first_rank is None else 1 / first_rank,
            )
        )
    return tuple(evaluated)


def evaluate_generation(
    cases: Sequence[EvaluationCase], outputs: Mapping[str, GenerationOutput]
) -> tuple[GenerationCaseResult, ...]:
    _validate_output_ids(cases, outputs, "generation")
    evaluated: list[GenerationCaseResult] = []
    for case in cases:
        output = outputs[case.id]
        if len(set(output.cited_source_ids)) != len(output.cited_source_ids):
            raise ValueError(f"generation output for {case.id!r} contains duplicate citations")
        answer = normalize_text(output.answer)
        found = tuple(
            fact.id
            for fact in case.expected_facts
            if any(normalize_text(variant) in answer for variant in fact.answer_variants)
        )
        found_set = set(found)
        grounded = tuple(
            fact.id
            for fact in case.expected_facts
            if fact.id in found_set
            and any(source_id in output.cited_source_ids for source_id in fact.evidence_source_ids)
        )
        relevant_to_found = {
            source_id
            for fact in case.expected_facts
            if fact.id in found_set
            for source_id in fact.evidence_source_ids
        }
        valid = tuple(
            source_id for source_id in output.cited_source_ids if source_id in relevant_to_found
        )
        invalid = tuple(
            source_id for source_id in output.cited_source_ids if source_id not in relevant_to_found
        )
        forbidden_hits = tuple(
            phrase
            for phrase in case.forbidden_phrases
            if normalize_text(phrase) in answer
        )
        fact_count = len(case.expected_facts)
        evaluated.append(
            GenerationCaseResult(
                case_id=case.id,
                answer=output.answer,
                cited_source_ids=output.cited_source_ids,
                found_fact_ids=found,
                missed_fact_ids=tuple(
                    fact.id for fact in case.expected_facts if fact.id not in found_set
                ),
                grounded_fact_ids=grounded,
                ungrounded_fact_ids=tuple(fact_id for fact_id in found if fact_id not in grounded),
                valid_citation_source_ids=valid,
                invalid_citation_source_ids=invalid,
                forbidden_phrase_hits=forbidden_hits,
                fact_coverage=len(found) / fact_count,
                grounded_fact_coverage=len(grounded) / fact_count,
                citation_precision=(
                    len(valid) / len(output.cited_source_ids)
                    if output.cited_source_ids
                    else 0.0
                ),
            )
        )
    return tuple(evaluated)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: int
    dataset_id: str
    case_ids: tuple[str, ...]
    top_k: int
    retrieval: tuple[RetrievalCaseResult, ...]
    generation: tuple[GenerationCaseResult, ...]
    retrieval_summary: Mapping[str, float]
    generation_summary: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


def build_report(
    dataset: EvaluationDataset,
    retrieval: Sequence[RetrievalCaseResult],
    generation: Sequence[GenerationCaseResult],
) -> EvaluationReport:
    case_ids = tuple(case.id for case in dataset.cases)
    if tuple(result.case_id for result in retrieval) != case_ids:
        raise ValueError("retrieval results must match dataset case order")
    if tuple(result.case_id for result in generation) != case_ids:
        raise ValueError("generation results must match dataset case order")
    top_values = {result.top_k for result in retrieval}
    if len(top_values) != 1:
        raise ValueError("retrieval results must use one top_k value")
    top_k = next(iter(top_values))
    return EvaluationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        dataset_id=dataset.dataset_id,
        case_ids=case_ids,
        top_k=top_k,
        retrieval=tuple(retrieval),
        generation=tuple(generation),
        retrieval_summary={
            "precision_at_k": _mean(item.precision_at_k for item in retrieval),
            "recall_at_k": _mean(item.recall_at_k for item in retrieval),
            "mrr_at_k": _mean(item.mrr_at_k for item in retrieval),
        },
        generation_summary={
            "fact_coverage": _mean(item.fact_coverage for item in generation),
            "grounded_fact_coverage": _mean(item.grounded_fact_coverage for item in generation),
            "citation_precision": _mean(item.citation_precision for item in generation),
        },
    )


@dataclass(frozen=True, slots=True)
class MetricComparison:
    previous: float
    current: float
    difference: float


@dataclass(frozen=True, slots=True)
class ReportComparison:
    dataset_id: str
    case_ids: tuple[str, ...]
    top_k: int
    retrieval: Mapping[str, MetricComparison]
    generation: Mapping[str, MetricComparison]


def compare_reports(previous: EvaluationReport, current: EvaluationReport) -> ReportComparison:
    for report in (previous, current):
        if report.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"unsupported report schema version: {report.schema_version}")
    if previous.dataset_id != current.dataset_id:
        raise ValueError("reports use different datasets")
    if previous.case_ids != current.case_ids:
        raise ValueError("reports contain different cases or case order")
    if previous.top_k != current.top_k:
        raise ValueError("reports use different top_k values")
    return ReportComparison(
        dataset_id=current.dataset_id,
        case_ids=current.case_ids,
        top_k=current.top_k,
        retrieval=_compare_summaries(previous.retrieval_summary, current.retrieval_summary),
        generation=_compare_summaries(previous.generation_summary, current.generation_summary),
    )


def load_cases(path: str | Path) -> EvaluationDataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _object(raw, "dataset")
    _exact_keys(root, {"schema_version", "dataset_id", "cases"}, "dataset")
    schema_version = _integer(root["schema_version"], "dataset.schema_version")
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema version: {schema_version}")
    dataset_id = _text(root["dataset_id"], "dataset.dataset_id")
    rows = _list(root["cases"], "dataset.cases")
    if not rows:
        raise ValueError("dataset.cases cannot be empty")
    cases = tuple(_parse_case(row, index) for index, row in enumerate(rows))
    _require_unique((case.id for case in cases), "case IDs")
    return EvaluationDataset(schema_version, dataset_id, cases)


def _parse_case(raw: object, index: int) -> EvaluationCase:
    location = f"dataset.cases[{index}]"
    row = _object(raw, location)
    _exact_keys(row, {"id", "question", "expected_facts", "forbidden_phrases"}, location)
    facts_raw = _list(row["expected_facts"], f"{location}.expected_facts")
    if not facts_raw:
        raise ValueError(f"{location}.expected_facts cannot be empty")
    facts = tuple(
        _parse_fact(value, location, fact_index)
        for fact_index, value in enumerate(facts_raw)
    )
    _require_unique((fact.id for fact in facts), f"fact IDs in {location}")
    forbidden = _text_tuple(row["forbidden_phrases"], f"{location}.forbidden_phrases")
    return EvaluationCase(
        id=_text(row["id"], f"{location}.id"),
        question=_text(row["question"], f"{location}.question"),
        expected_facts=facts,
        forbidden_phrases=forbidden,
    )


def _parse_fact(raw: object, case_location: str, index: int) -> ExpectedFact:
    location = f"{case_location}.expected_facts[{index}]"
    row = _object(raw, location)
    _exact_keys(row, {"id", "text", "answer_variants", "evidence_source_ids"}, location)
    variants = _text_tuple(row["answer_variants"], f"{location}.answer_variants")
    evidence = _text_tuple(row["evidence_source_ids"], f"{location}.evidence_source_ids")
    if not variants or not evidence:
        raise ValueError(f"{location} requires answer variants and evidence source IDs")
    _require_unique(variants, f"answer variants in {location}")
    _require_unique(evidence, f"evidence source IDs in {location}")
    return ExpectedFact(
        id=_text(row["id"], f"{location}.id"),
        text=_text(row["text"], f"{location}.text"),
        answer_variants=variants,
        evidence_source_ids=evidence,
    )


def _validate_output_ids(
    cases: Sequence[EvaluationCase], outputs: Mapping[str, object], label: str
) -> None:
    expected = {case.id for case in cases}
    actual = set(outputs)
    if expected != actual:
        raise ValueError(
            f"{label} output case IDs do not match dataset; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    if not collected:
        raise ValueError("cannot average an empty sequence")
    return sum(collected) / len(collected)


def _compare_summaries(
    previous: Mapping[str, float], current: Mapping[str, float]
) -> dict[str, MetricComparison]:
    if previous.keys() != current.keys():
        raise ValueError("reports contain different summary metrics")
    return {
        name: MetricComparison(previous[name], current[name], current[name] - previous[name])
        for name in previous
    }


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return cast(list[object], value)


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _text_tuple(value: object, location: str) -> tuple[str, ...]:
    return tuple(
        _text(item, f"{location}[{index}]")
        for index, item in enumerate(_list(value, location))
    )


def _exact_keys(value: Mapping[str, object], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} has invalid fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_unique(values: Iterable[str], label: str) -> None:
    collected = tuple(values)
    if len(collected) != len(set(collected)):
        raise ValueError(f"{label} must be unique")


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"-+", "-", re.sub(r"[^\w]+", "-", normalized).replace("_", "-")).strip(
        "-"
    )
