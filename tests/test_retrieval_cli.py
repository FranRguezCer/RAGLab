from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from raglab.retrieval import RetrievalRequest, RetrievalResponse
from raglab.retrieval.cli import load_history, main, parse_filter


def test_parse_filter_supports_json_scalars_and_in_values() -> None:
    assert parse_filter("document.metadata.tenant_id:eq=acme").value == "acme"
    assert parse_filter("chunk.metadata.page:eq=3").value == 3
    assert parse_filter('chunk.metadata.lang:in=["en","es"]').value == ("en", "es")
    assert parse_filter("chunk.metadata.lang:in=en,es").value == ("en", "es")


@pytest.mark.parametrize("value", ["bad", "document.metadata.x:eq=", "document.id:eq=x"])
def test_parse_filter_rejects_malformed_or_unsafe_fields(value: str) -> None:
    with pytest.raises(ValueError):
        parse_filter(value)


def test_load_history_accepts_strings_and_message_objects(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps(["first", {"role": "user", "content": " second "}]))

    assert load_history(path) == ("first", "second")


def test_load_history_rejects_invalid_shape(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"content": "not a list"}))

    with pytest.raises(ValueError, match="must be a list"):
        load_history(path)


def test_cli_builds_request_and_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    history = tmp_path / "history.json"
    history.write_text(json.dumps([{"role": "user", "content": "Fault E17"}]))
    captured: dict[str, object] = {}

    class Pipeline:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured["rewriter"] = kwargs.get("rewriter")

        def retrieve(self, request: object) -> RetrievalResponse:
            captured["request"] = request
            return RetrievalResponse("query", "standalone", ("query",), (), ())

    monkeypatch.setattr("raglab.retrieval.cli.RetrievalPipeline", Pipeline)
    monkeypatch.setattr("raglab.retrieval.cli.PostgresRetrievalRepository", lambda _dsn: object())
    monkeypatch.setattr("raglab.retrieval.cli.OllamaEmbeddingProvider", lambda: object())
    monkeypatch.setattr("raglab.retrieval.cli.OllamaQueryRewriter", lambda: "rewriter")

    result = main(
        [
            "Why?",
            "--history-file",
            str(history),
            "--filter",
            "document.metadata.tenant:eq=acme",
            "--no-rerank",
            "--no-mmr",
            "--no-small-to-big",
        ]
    )

    assert result == 0
    assert captured["rewriter"] == "rewriter"
    request = cast(RetrievalRequest, captured["request"])
    assert request.history == ("Fault E17",)
    assert request.filters[0].value == "acme"
    assert json.loads(capsys.readouterr().out)["rewritten_query"] == "standalone"


def test_cli_reports_runtime_errors_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Pipeline:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def retrieve(self, _request: object) -> RetrievalResponse:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr("raglab.retrieval.cli.RetrievalPipeline", Pipeline)
    monkeypatch.setattr("raglab.retrieval.cli.PostgresRetrievalRepository", lambda _dsn: object())
    monkeypatch.setattr("raglab.retrieval.cli.OllamaEmbeddingProvider", lambda: object())

    with pytest.raises(SystemExit) as raised:
        main(["query", "--no-rerank"])

    assert raised.value.code == 1
    assert "database unavailable" in capsys.readouterr().err
