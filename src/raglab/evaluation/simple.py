"""Small, deterministic checks for generation over fixed evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from raglab.evaluation.metrics import normalize
from raglab.generation.models import GenerationResponse

CheckStatus = Literal["PASS", "FAIL"]


def _require_nonblank(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} cannot be empty")


def _duplicates(values: tuple[str, ...]) -> set[str]:
    return {value for value in values if values.count(value) > 1}


@dataclass(frozen=True, slots=True)
class GenerationEvidence:
    """One fixed source shown to the generation system under evaluation."""

    source_id: str
    text: str

    def __post_init__(self) -> None:
        _require_nonblank(self.source_id, "evidence source_id")
        _require_nonblank(self.text, "evidence text")


@dataclass(frozen=True, slots=True)
class GenerationCase:
    """Visible expectations for one generation example."""

    question: str
    evidence: tuple[GenerationEvidence, ...]
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_abstained: bool = False
    expected_source_ids: tuple[str, ...] = ()
    expected_model_calls: int = 1

    def __post_init__(self) -> None:
        _require_nonblank(self.question, "question")
        if not self.evidence:
            raise ValueError("evidence cannot be empty")
        if self.expected_model_calls < 0:
            raise ValueError("expected_model_calls cannot be negative")

        evidence_ids = tuple(item.source_id for item in self.evidence)
        if duplicates := _duplicates(evidence_ids):
            raise ValueError(f"evidence source_ids must be unique: {sorted(duplicates)}")
        if duplicates := _duplicates(self.expected_source_ids):
            raise ValueError(f"expected_source_ids must be unique: {sorted(duplicates)}")
        unknown = set(self.expected_source_ids) - set(evidence_ids)
        if unknown:
            raise ValueError(f"expected_source_ids are not present in evidence: {sorted(unknown)}")

        for field, phrases in (
            ("required_phrases", self.required_phrases),
            ("forbidden_phrases", self.forbidden_phrases),
        ):
            for phrase in phrases:
                _require_nonblank(phrase, field)
            normalized = tuple(normalize(phrase) for phrase in phrases)
            if any(not phrase for phrase in normalized):
                raise ValueError(f"{field} cannot normalize to an empty phrase")
            if duplicates := _duplicates(normalized):
                raise ValueError(
                    f"{field} must be unique after normalization: {sorted(duplicates)}"
                )
        overlap = set(map(normalize, self.required_phrases)) & set(
            map(normalize, self.forbidden_phrases)
        )
        if overlap:
            raise ValueError(
                f"phrases cannot be both required and forbidden: {sorted(overlap)}"
            )


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """The only generation behavior inspected by this evaluator."""

    answer: str
    abstained: bool
    used_source_ids: tuple[str, ...]
    model_calls: int

    def __post_init__(self) -> None:
        if self.model_calls < 0:
            raise ValueError("model_calls cannot be negative")
        if duplicates := _duplicates(self.used_source_ids):
            raise ValueError(f"used_source_ids must be unique: {sorted(duplicates)}")
        for source_id in self.used_source_ids:
            _require_nonblank(source_id, "used source_id")

    @classmethod
    def from_generation_response(cls, response: GenerationResponse) -> GenerationOutput:
        """Adapt the production response without interpreting retrieval results."""

        return cls(
            answer=response.answer,
            abstained=response.abstained,
            used_source_ids=response.source_ids,
            model_calls=response.metrics.model_calls,
        )


@dataclass(frozen=True, slots=True)
class GenerationCheck:
    """One expectation with its directly observable result."""

    name: str
    expected: object
    observed: object
    status: CheckStatus

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """All visible checks for one case, with no weighted aggregation."""

    checks: tuple[GenerationCheck, ...]
    status: CheckStatus

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _check(name: str, expected: object, observed: object, passed: bool) -> GenerationCheck:
    return GenerationCheck(name, expected, observed, "PASS" if passed else "FAIL")


def evaluate_generation(case: GenerationCase, output: GenerationOutput) -> GenerationResult:
    """Evaluate generation using five deterministic, independently visible checks."""

    evidence_ids = {item.source_id for item in case.evidence}
    unknown = set(output.used_source_ids) - evidence_ids
    if unknown:
        raise ValueError(f"used_source_ids are not present in evidence: {sorted(unknown)}")

    answer = normalize(output.answer)
    present_required = tuple(
        phrase for phrase in case.required_phrases if normalize(phrase) in answer
    )
    present_forbidden = tuple(
        phrase for phrase in case.forbidden_phrases if normalize(phrase) in answer
    )
    checks = (
        _check(
            "required_phrases",
            case.required_phrases,
            present_required,
            present_required == case.required_phrases,
        ),
        _check("forbidden_phrases", (), present_forbidden, not present_forbidden),
        _check(
            "abstained",
            case.expected_abstained,
            output.abstained,
            output.abstained is case.expected_abstained,
        ),
        _check(
            "used_source_ids",
            case.expected_source_ids,
            output.used_source_ids,
            set(output.used_source_ids) == set(case.expected_source_ids),
        ),
        _check(
            "model_calls",
            case.expected_model_calls,
            output.model_calls,
            output.model_calls == case.expected_model_calls,
        ),
    )
    return GenerationResult(checks, "PASS" if all(check.passed for check in checks) else "FAIL")


__all__ = [
    "GenerationCase",
    "GenerationCheck",
    "GenerationEvidence",
    "GenerationOutput",
    "GenerationResult",
    "evaluate_generation",
]
