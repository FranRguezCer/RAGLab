"""Public, versioned contracts for reproducible RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

RUN_SCHEMA_VERSION = 1
PROTECTED_COLLECTION_PREFIX = "raglab-eval-"
VERDICTS = {"improved", "regressed", "mixed", "no_clear_change"}
RUN_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://raglab.local/schemas/evaluation-run-v1.json",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "profile",
        "status",
        "partial",
        "metadata",
        "corpus",
        "config",
        "ingestion",
        "cases",
        "summary",
        "errors",
    ],
    "properties": {
        "schema_version": {"const": RUN_SCHEMA_VERSION},
        "run_id": {"type": "string"},
        "profile": {"enum": ["core", "live"]},
        "status": {"enum": ["running", "complete", "failed"]},
        "partial": {"type": "boolean"},
        "metadata": {"type": "object"},
        "corpus": {"type": "object"},
        "config": {"type": "object"},
        "ingestion": {"type": "object"},
        "cases": {"type": "array"},
        "summary": {"type": "object"},
        "errors": {"type": "object"},
    },
    "additionalProperties": True,
}


@dataclass(frozen=True, slots=True)
class EvaluationSource:
    id: str
    location: str
    sha256: str | None = None
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkCheck:
    id: str
    source_id: str
    left_anchor: str
    right_anchor: str
    reason: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    query: str
    expected_source_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    should_abstain: bool = False
    history: tuple[str, ...] = ()
    domain: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    schema_version: int
    profile: str
    sources: tuple[EvaluationSource, ...]
    cases: tuple[EvaluationCase, ...]
    must_separate: tuple[ChunkCheck, ...] = ()
    must_keep: tuple[ChunkCheck, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    base_path: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    source_ids: tuple[str, ...]
    anchors_by_rank: tuple[tuple[str, ...], ...]
    latency_ms: float
    aggregate_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationObservation:
    answer: str
    abstained: bool
    cited_source_ids: tuple[str, ...]
    prompt_tokens: int | None
    generated_tokens: int | None
    model_calls: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class IngestionObservation:
    documents: int
    chunks: int
    token_counts: tuple[int, ...]
    separation_passed: int
    separation_total: int
    cohesion_passed: int
    cohesion_total: int
    latency_ms: float


class EvaluationExecutor(Protocol):
    """Internal execution boundary used by real and hermetic evaluators."""

    def prepare(
        self, manifest: EvaluationManifest, *, collection: str, reuse_index: bool
    ) -> IngestionObservation: ...

    def retrieve(
        self, case: EvaluationCase, *, collection: str, exact: bool
    ) -> RetrievalObservation: ...

    def generate(
        self, case: EvaluationCase, *, collection: str
    ) -> GenerationObservation: ...

    def release_generator(self) -> None: ...


class EvaluationJudge(Protocol):
    """Optional advisory judge; deterministic checks remain authoritative."""

    @property
    def model(self) -> str: ...

    def evaluate(self, case: EvaluationCase, answer: str) -> dict[str, Any]: ...
