from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab.contracts import Citation, ProvenanceStatus
from raglab.evaluation import EvaluationCase, EvaluationDataset, ExpectedFact
from raglab.evaluation_application import (
    EvaluationApplication,
    LiveEvaluationSettings,
    create_live_evaluation_application,
)
from raglab.evaluation_cli import main
from raglab.generation import (
    GeneratedSource,
    GenerationConfig,
    GenerationMetrics,
    GenerationRequest,
    GenerationResponse,
    GenerationStrategy,
)
from raglab.retrieval import RankingTrace, RetrievalConfig, RetrievalResponse, RetrievalResult


def _citation() -> Citation:
    return Citation(
        source_uri="file:///manual.md",
        source_name="manual.md",
        title="Manual",
        heading_path=("Section",),
        start_page=None,
        end_page=None,
        start_line=1,
        end_line=2,
        provenance_status=ProvenanceStatus.COMPLETE,
    )


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        1,
        "dataset",
        (
            EvaluationCase(
                "case-1",
                "What happens?",
                (
                    ExpectedFact(
                        "fact-1",
                        "Alpha happens.",
                        ("Alpha happens",),
                        ("manual-md/section",),
                    ),
                ),
                (),
            ),
        ),
    )


class _Pipeline:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        citation = _citation()
        retrieval = RetrievalResponse(
            request.retrieval.query,
            None,
            (request.retrieval.query,),
            (),
            (
                RetrievalResult(
                    "chunk-1",
                    "document-1",
                    "Alpha happens.",
                    citation,
                    ("chunk-1",),
                    0,
                    0,
                    RankingTrace(1, 1, 0.1, 1.0, 0.5, None, None),
                ),
            ),
        )
        return GenerationResponse(
            "Alpha happens.",
            False,
            (GeneratedSource("S1", "chunk-1", "document-1", citation),),
            retrieval,
            GenerationStrategy.HIERARCHICAL,
            False,
            1,
            1,
            GenerationMetrics(2, 20, 10, 5, selection_calls=1),
            ("S1",),
        )


def test_application_runs_real_generation_boundary_and_returns_traces_and_report() -> None:
    pipeline = _Pipeline()
    application = EvaluationApplication(
        pipeline,
        retrieval_config=RetrievalConfig(top_k=1, candidate_k=2),
        generation_config=GenerationConfig(minimum_sources=1),
    )

    run = application.run(_dataset(), collection="manuals")

    assert len(pipeline.requests) == 1
    assert pipeline.requests[0].retrieval.collection == "manuals"
    assert run.traces[0].response.answer == "Alpha happens."
    assert run.traces[0].retrieval.recall_at_k == 1.0
    assert run.traces[0].generation.grounded_fact_coverage == 1.0
    assert run.report.retrieval_summary["recall_at_k"] == 1.0
    assert run.metadata["dataset_sha256"] is None
    assert json.loads(run.to_json())["traces"][0]["response"]["metrics"]["selection_calls"] == 1


def test_application_runs_every_dataset_case_in_order() -> None:
    dataset = _dataset()
    second = EvaluationCase(
        "case-2",
        "And again?",
        dataset.cases[0].expected_facts,
        (),
    )
    pipeline = _Pipeline()
    application = EvaluationApplication(
        pipeline,
        retrieval_config=RetrievalConfig(top_k=1),
        generation_config=GenerationConfig(minimum_sources=1),
    )

    run = application.run(EvaluationDataset(1, "dataset", (*dataset.cases, second)))

    assert [request.retrieval.query for request in pipeline.requests] == [
        "What happens?",
        "And again?",
    ]
    assert run.report.case_ids == ("case-1", "case-2")


def test_live_factory_composes_postgres_and_ollama_behind_application_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    pipeline = _Pipeline()

    monkeypatch.setattr(
        "raglab.evaluation_application.OllamaEmbeddingProvider",
        lambda **kwargs: captured.setdefault("embedding", kwargs),
    )
    monkeypatch.setattr(
        "raglab.evaluation_application.PostgresRetrievalRepository",
        lambda dsn: captured.setdefault("repository", dsn),
    )
    monkeypatch.setattr(
        "raglab.evaluation_application.OllamaQueryRewriter",
        lambda **kwargs: captured.setdefault("rewriter", kwargs),
    )

    def retrieval(repository: object, embedding: object, **kwargs: object) -> object:
        captured["retrieval"] = (repository, embedding, kwargs)
        return "retrieval-pipeline"

    def generation(retrieval_stage: object, model: object, **kwargs: object) -> _Pipeline:
        captured["generation"] = (retrieval_stage, model, kwargs)
        return pipeline

    monkeypatch.setattr("raglab.evaluation_application.RetrievalPipeline", retrieval)
    monkeypatch.setattr(
        "raglab.evaluation_application.OllamaGenerationModel",
        lambda **kwargs: captured.setdefault("model", kwargs),
    )
    monkeypatch.setattr(
        "raglab.evaluation_application.EvidenceClaimVerifier", lambda: "verifier"
    )
    monkeypatch.setattr("raglab.evaluation_application.GenerationPipeline", generation)
    settings = LiveEvaluationSettings(
        dsn="postgresql://example",
        embedding_model="embed",
        generation_model="generate",
        ollama_base_url="http://ollama",
        keep_alive="10m",
        num_ctx=8192,
    )

    application = create_live_evaluation_application(
        settings,
        retrieval_config=RetrievalConfig(top_k=2),
        generation_config=settings.generation_config(minimum_sources=1),
    )

    assert application.generation_pipeline is pipeline
    assert captured["repository"] == "postgresql://example"
    assert captured["embedding"] == {
        "model": "embed",
        "dimension": 1024,
        "base_url": "http://ollama",
        "num_gpu": None,
        "num_ctx": 4096,
        "keep_alive": "10m",
    }
    assert application.retrieval_config.top_k == 2
    assert application.generation_config.num_ctx == 8192


def test_evaluation_cli_uses_application_and_writes_complete_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pipeline = _Pipeline()
    application = EvaluationApplication(
        pipeline,
        retrieval_config=RetrievalConfig(top_k=1),
        generation_config=GenerationConfig(minimum_sources=1),
    )
    captured: dict[str, object] = {}

    def create(settings: object, **kwargs: object) -> EvaluationApplication:
        captured["settings"] = settings
        captured.update(kwargs)
        return application

    monkeypatch.setattr("raglab.evaluation_cli.load_cases", lambda _path: _dataset())
    monkeypatch.setattr("raglab.evaluation_cli.create_live_evaluation_application", create)
    output = tmp_path / "runs" / "evaluation.json"

    assert (
        main(
            [
                "dataset.json",
                "--collection",
                "manuals",
                "--top-k",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text())
    assert printed["dataset_id"] == "dataset"
    assert printed["case_count"] == 1
    assert printed["output"] == str(output)
    assert persisted["report"]["dataset_id"] == "dataset"
    assert persisted["traces"][0]["response"]["answer"] == "Alpha happens."
    assert pipeline.requests[0].retrieval.collection == "manuals"
    assert captured["retrieval_config"].top_k == 1  # type: ignore[union-attr]
