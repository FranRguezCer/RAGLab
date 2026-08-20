from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from raglab.generation import (
    GenerationMetrics,
    GenerationRequest,
    GenerationResponse,
    GenerationStrategy,
)
from raglab.generation.cli import main
from raglab.retrieval import RetrievalResponse


def test_cli_composes_typed_retrieval_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    retrieval_response = RetrievalResponse("question", None, ("question",), (), ())

    def embedding(**kwargs: object) -> object:
        captured["embedding"] = kwargs
        return object()

    class RetrievalPipeline:
        def __init__(self, repository: object, provider: object, **kwargs: object) -> None:
            captured["retrieval_pipeline"] = (repository, provider, kwargs)

    class GenerationPipeline:
        def __init__(self, retrieval: object, model: object, **kwargs: object) -> None:
            captured["generation_pipeline"] = (retrieval, model, kwargs)

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            captured["request"] = request
            return GenerationResponse(
                "No evidence.",
                True,
                (),
                retrieval_response,
                GenerationStrategy.SINGLE_PASS,
                True,
                5,
                0,
                GenerationMetrics(0, 0, 0, 0),
            )

    monkeypatch.setattr("raglab.generation.cli.OllamaEmbeddingProvider", embedding)
    monkeypatch.setattr("raglab.generation.cli.PostgresRetrievalRepository", lambda dsn: dsn)
    monkeypatch.setattr("raglab.generation.cli.RetrievalPipeline", RetrievalPipeline)
    monkeypatch.setattr("raglab.generation.cli.OllamaGenerationModel", lambda **kwargs: kwargs)
    monkeypatch.setattr("raglab.generation.cli.GenerationPipeline", GenerationPipeline)
    monkeypatch.setattr("raglab.generation.cli.OllamaQueryRewriter", lambda **kwargs: "rewriter")
    history = tmp_path / "history.json"
    history.write_text(json.dumps([{"role": "user", "content": "Fault E17"}]))

    assert (
        main(
            [
                "question",
                "--top-k",
                "2",
                "--no-rerank",
                "--history-file",
                str(history),
            ]
        )
        == 0
    )

    request = cast(GenerationRequest, captured["request"])
    assert request.retrieval.config.top_k == 5
    assert request.retrieval.config.candidate_k >= 5
    assert request.retrieval.history == ("Fault E17",)
    assert request.config.model == "qwen3:4b"
    assert request.config.num_ctx == 12288
    assert cast(dict[str, object], captured["embedding"])["num_gpu"] == 0
    assert cast(dict[str, object], captured["embedding"])["keep_alive"] == "5m"
    assert cast(tuple[object, object, dict[str, object]], captured["retrieval_pipeline"])[2][
        "rewriter"
    ] == "rewriter"
    payload = json.loads(capsys.readouterr().out)
    assert payload["strategy"] == "single_pass"
    assert payload["retrieval"]["query"] == "question"


def test_cli_uses_model_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAGLAB_GENERATION_MODEL", "custom-llm")
    monkeypatch.setenv("RAGLAB_EMBEDDING_MODEL", "custom-embed")
    captured: dict[str, object] = {}

    class Pipeline:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            captured["request"] = request
            retrieval = RetrievalResponse("q", None, ("q",), (), ())
            return GenerationResponse(
                "No evidence.",
                True,
                (),
                retrieval,
                GenerationStrategy.SINGLE_PASS,
                True,
                5,
                0,
                GenerationMetrics(0, 0, 0, 0),
            )

    monkeypatch.setattr("raglab.generation.cli.OllamaEmbeddingProvider", lambda **kwargs: kwargs)
    monkeypatch.setattr("raglab.generation.cli.PostgresRetrievalRepository", lambda _dsn: object())
    monkeypatch.setattr(
        "raglab.generation.cli.RetrievalPipeline", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr("raglab.generation.cli.OllamaGenerationModel", lambda **_kwargs: object())
    monkeypatch.setattr("raglab.generation.cli.GenerationPipeline", Pipeline)

    assert main(["question", "--no-rerank"]) == 0
    request = cast(GenerationRequest, captured["request"])
    assert request.config.model == "custom-llm"
    assert captured["embedding_model"] == "custom-embed"
    assert json.loads(capsys.readouterr().out)["abstained"] is True
