"""Application boundary for running a dataset through the real RAG pipeline."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from raglab.embeddings import OllamaEmbeddingProvider
from raglab.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    GenerationCaseResult,
    GenerationOutput,
    RetrievalCaseResult,
    RetrievalOutput,
    build_report,
    evaluate_generation,
    evaluate_retrieval,
)
from raglab.generation import (
    EvidenceClaimVerifier,
    GenerationConfig,
    GenerationPipeline,
    GenerationRequest,
    GenerationResponse,
    OllamaGenerationModel,
)
from raglab.nli import NLI_MODEL, NLI_REVISION
from raglab.retrieval import (
    OllamaQueryRewriter,
    PostgresRetrievalRepository,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalRequest,
)
from raglab.retrieval.reranking import BGE_MODEL, BGE_REVISION

DEFAULT_EVALUATION_DSN = "postgresql://raglab:raglab@127.0.0.1:5432/raglab"


class GenerationStage(Protocol):
    """Small seam that keeps evaluation orchestration independent of infrastructure."""

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


@dataclass(frozen=True, slots=True)
class EvaluationCaseTrace:
    """The real pipeline response and deterministic scores for one dataset case."""

    case: EvaluationCase
    response: GenerationResponse
    retrieval: RetrievalCaseResult
    generation: GenerationCaseResult


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Complete inspectable execution plus its deterministic aggregate report."""

    traces: tuple[EvaluationCaseTrace, ...]
    report: EvaluationReport
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


class EvaluationApplication:
    """Run every evaluation case through one configured generation pipeline."""

    def __init__(
        self,
        generation_pipeline: GenerationStage,
        *,
        retrieval_config: RetrievalConfig | None = None,
        generation_config: GenerationConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.generation_pipeline = generation_pipeline
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.generation_config = generation_config or GenerationConfig()
        self.metadata = metadata or {}

    def run(self, dataset: EvaluationDataset, *, collection: str = "documents") -> EvaluationRun:
        """Execute all cases in dataset order and score the exact responses returned."""

        started = time.perf_counter()
        responses: list[GenerationResponse] = []
        retrieval_outputs: dict[str, RetrievalOutput] = {}
        generation_outputs: dict[str, GenerationOutput] = {}
        for case in dataset.cases:
            request = GenerationRequest(
                RetrievalRequest(
                    case.question,
                    collection=case.collection or collection,
                    config=self.retrieval_config,
                ),
                self.generation_config,
            )
            response = self.generation_pipeline.generate(request)
            responses.append(response)
            retrieval_outputs[case.id] = RetrievalOutput.from_response(response.retrieval)
            generation_outputs[case.id] = GenerationOutput.from_response(response)

        retrieval_results = evaluate_retrieval(
            dataset.cases,
            retrieval_outputs,
            top_k=self.retrieval_config.top_k,
        )
        generation_results = evaluate_generation(dataset.cases, generation_outputs)
        report = build_report(dataset, retrieval_results, generation_results)
        traces = tuple(
            EvaluationCaseTrace(case, response, retrieval, generation)
            for case, response, retrieval, generation in zip(
                dataset.cases,
                responses,
                retrieval_results,
                generation_results,
                strict=True,
            )
        )
        return EvaluationRun(
            traces,
            report,
            {
                "duration_seconds": round(time.perf_counter() - started, 6),
                "build": os.environ.get("RAGLAB_BUILD_SHA", "development"),
                "dataset_sha256": dataset.content_sha256,
                "retrieval_config": asdict(self.retrieval_config),
                "generation_config": asdict(self.generation_config),
                **self.metadata,
            },
        )


@dataclass(frozen=True, slots=True)
class LiveEvaluationSettings:
    """Infrastructure settings shared by the CLI, notebooks, and integrations."""

    dsn: str = DEFAULT_EVALUATION_DSN
    embedding_model: str = "qwen3-embedding:0.6b"
    generation_model: str = "qwen3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    keep_alive: str = "5m"
    num_ctx: int = 12_288
    embedding_num_gpu: int | None = None

    @classmethod
    def from_env(cls) -> LiveEvaluationSettings:
        return cls(
            dsn=os.environ.get("RAGLAB_DSN", DEFAULT_EVALUATION_DSN),
            embedding_model=os.environ.get("RAGLAB_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            generation_model=os.environ.get("RAGLAB_GENERATION_MODEL", "qwen3:4b"),
            ollama_base_url=os.environ.get("RAGLAB_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            keep_alive=os.environ.get("RAGLAB_KEEP_ALIVE", "5m"),
            num_ctx=_environment_integer("RAGLAB_NUM_CTX", 12_288),
            embedding_num_gpu=_environment_optional_integer("RAGLAB_EMBEDDING_NUM_GPU"),
        )

    def generation_config(
        self,
        *,
        minimum_sources: int = 5,
        num_predict: int = 512,
    ) -> GenerationConfig:
        return GenerationConfig(
            model=self.generation_model,
            num_ctx=self.num_ctx,
            num_predict=num_predict,
            keep_alive=self.keep_alive,
            minimum_sources=minimum_sources,
        )


def create_live_evaluation_application(
    settings: LiveEvaluationSettings | None = None,
    *,
    retrieval_config: RetrievalConfig | None = None,
    generation_config: GenerationConfig | None = None,
) -> EvaluationApplication:
    """Compose PostgreSQL, Ollama, retrieval, generation, and evaluation once."""

    resolved = settings or LiveEvaluationSettings.from_env()
    configured_generation = generation_config or resolved.generation_config()
    embedding = OllamaEmbeddingProvider(
        model=resolved.embedding_model,
        dimension=1024,
        base_url=resolved.ollama_base_url,
        num_gpu=resolved.embedding_num_gpu,
        num_ctx=4096,
        keep_alive=resolved.keep_alive,
    )
    retrieval = RetrievalPipeline(
        PostgresRetrievalRepository(resolved.dsn),
        embedding,
        rewriter=OllamaQueryRewriter(
            model=resolved.generation_model,
            base_url=resolved.ollama_base_url,
        ),
    )
    generation = GenerationPipeline(
        retrieval,
        OllamaGenerationModel(base_url=resolved.ollama_base_url),
        grounding_verifier=EvidenceClaimVerifier(),
        embedding_model=resolved.embedding_model,
    )
    return EvaluationApplication(
        generation,
        retrieval_config=retrieval_config or RetrievalConfig(),
        generation_config=configured_generation,
        metadata={
            "models": {
                "embedding": resolved.embedding_model,
                "generation": resolved.generation_model,
                "reranker": {"name": BGE_MODEL, "revision": BGE_REVISION},
                "nli": {"name": NLI_MODEL, "revision": NLI_REVISION, "device": "cpu"},
            }
        },
    )


def _environment_integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _environment_optional_integer(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


__all__ = [
    "EvaluationApplication",
    "EvaluationCaseTrace",
    "EvaluationRun",
    "DEFAULT_EVALUATION_DSN",
    "GenerationStage",
    "LiveEvaluationSettings",
    "create_live_evaluation_application",
]
