"""Immutable contracts for strict, citation-checked RAG generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from raglab.contracts import Citation
from raglab.ollama import validate_positive_duration
from raglab.retrieval import RetrievalRequest, RetrievalResponse


class GenerationStrategy(StrEnum):
    SINGLE_PASS = "single_pass"
    HIERARCHICAL = "hierarchical"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model: str = "qwen3:4b"
    num_ctx: int = 12_288
    num_predict: int = 512
    parallelism: int = 1
    keep_alive: str = "5m"
    minimum_sources: int = 5

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("generation model cannot be empty")
        if self.num_ctx < 1 or self.num_predict < 1:
            raise ValueError("num_ctx and num_predict must be positive")
        if self.num_predict >= self.num_ctx:
            raise ValueError("num_predict must be smaller than num_ctx")
        if self.parallelism != 1:
            raise ValueError("local generation currently requires parallelism=1")
        object.__setattr__(self, "keep_alive", validate_positive_duration(self.keep_alive))
        if self.minimum_sources < 1:
            raise ValueError("minimum_sources must be positive")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    retrieval: RetrievalRequest
    config: GenerationConfig = GenerationConfig()


@dataclass(frozen=True, slots=True)
class ModelInvocation:
    payload: dict[str, Any]
    prompt_tokens: int | None = None
    generated_tokens: int | None = None


class GenerationModel(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, Any],
        config: GenerationConfig,
    ) -> ModelInvocation: ...


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    id: str
    retrieval_result_id: str
    document_id: str
    citation: Citation


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    model_calls: int
    estimated_prompt_tokens: int
    prompt_tokens: int | None
    generated_tokens: int | None


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    answer: str
    abstained: bool
    sources: tuple[GeneratedSource, ...]
    retrieval: RetrievalResponse
    strategy: GenerationStrategy
    source_shortfall: bool
    minimum_sources: int
    source_count: int
    metrics: GenerationMetrics
