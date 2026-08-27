"""Public API for reproducible RAG evaluation."""

from raglab.evaluation.application import EvaluationApplication, HermeticEvaluationExecutor
from raglab.evaluation.manifest import corpus_fingerprint, definition_fingerprint, load_manifest
from raglab.evaluation.models import (
    RUN_JSON_SCHEMA,
    RUN_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationJudge,
    EvaluationManifest,
    FactExpectation,
    SemanticCalibrationConfig,
    SemanticConfig,
    SemanticModelConfig,
    SemanticTemplateConfig,
)
from raglab.evaluation.simple import (
    GenerationCase,
    GenerationCheck,
    GenerationEvidence,
    GenerationOutput,
    GenerationResult,
    evaluate_generation,
)

__all__ = [
    "RUN_JSON_SCHEMA",
    "RUN_SCHEMA_VERSION",
    "EvaluationApplication",
    "EvaluationCase",
    "EvaluationJudge",
    "EvaluationManifest",
    "FactExpectation",
    "GenerationCase",
    "GenerationCheck",
    "GenerationEvidence",
    "GenerationOutput",
    "GenerationResult",
    "SemanticCalibrationConfig",
    "SemanticConfig",
    "SemanticModelConfig",
    "SemanticTemplateConfig",
    "HermeticEvaluationExecutor",
    "corpus_fingerprint",
    "definition_fingerprint",
    "evaluate_generation",
    "load_manifest",
]
