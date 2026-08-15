from __future__ import annotations

import json
from pathlib import Path

import pytest

from raglab import cli
from raglab.contracts import IngestionReport, ProvenanceStatus, SourceKind


def test_cli_runs_migration_and_ingestion_for_an_absolute_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Guide", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeRepository:
        def __init__(self, dsn: str) -> None:
            captured["migration_dsn"] = dsn

        def migrate(self) -> None:
            captured["migrated"] = True

    def fake_ingest(source, collection, *, dsn, use_jina, chunk_config):
        captured.update(
            source=source,
            collection=collection,
            dsn=dsn,
            use_jina=use_jina,
            chunk_config=chunk_config,
        )
        return IngestionReport(
            source_uri=source.uri,
            collection=collection.name,
            status="indexed",
            document_id="document-id",
            chunk_count=1,
            content_hash="content-hash",
            fingerprint="fingerprint",
            provenance_status=ProvenanceStatus.COMPLETE,
        )

    monkeypatch.setattr(cli, "PostgresRepository", FakeRepository)
    monkeypatch.setattr(cli, "ingest", fake_ingest)

    result = cli.main(
        [
            str(source_path),
            "--collection",
            "knowledge-base",
            "--dsn",
            "postgresql://test",
        ]
    )

    assert result == 0
    assert captured["migrated"] is True
    assert captured["migration_dsn"] == "postgresql://test"
    source = captured["source"]
    assert source.kind is SourceKind.PATH
    assert source.uri == str(source_path.resolve())
    collection = captured["collection"]
    assert collection.name == "knowledge-base"
    assert collection.chunk_config["semantic_percentile"] == 90.0
    assert captured["use_jina"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "indexed"


def test_cli_marks_jina_as_an_explicit_remote_url_opt_in() -> None:
    source = cli._source_input("https://example.org/guide", use_jina=True)

    assert source.kind is SourceKind.URL
    assert source.allow_remote_service is True


def test_cli_rejects_jina_for_a_local_file() -> None:
    with pytest.raises(ValueError, match="public HTTP"):
        cli._source_input("guide.pdf", use_jina=True)
