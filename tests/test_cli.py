from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from raglab import cli
from raglab.chunking import ChunkingConfig
from raglab.contracts import CollectionConfig, IngestionReport, ProvenanceStatus, SourceKind


class TtyInput(StringIO):
    def isatty(self) -> bool:
        return True


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

        def list_collections(self) -> list[CollectionConfig]:
            return []

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
    monkeypatch.setattr(
        cli,
        "_prompt_collection",
        lambda _collections: pytest.fail("explicit --collection must bypass the prompt"),
    )

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


def test_cli_prompts_with_existing_collections_and_reuses_complete_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Guide", encoding="utf-8")
    existing = CollectionConfig(
        name="manuals",
        model="stored-model",
        dimension=1024,
        metric="cosine",
        chunk_config={
            "strategy": "structure_plus_semantics",
            "target_tokens": 240,
            "min_tokens": 80,
            "max_tokens": 360,
            "semantic_percentile": 75.0,
            "overlap_tokens": 0,
            "profile_version": 2,
        },
    )
    captured: dict[str, object] = {}

    class FakeRepository:
        def __init__(self, _dsn: str) -> None:
            pass

        def migrate(self) -> None:
            pass

        def list_collections(self) -> list[CollectionConfig]:
            return [existing]

    def fake_ingest(source, collection, *, dsn, use_jina, chunk_config):
        captured["collection"] = collection
        captured["chunk_config"] = chunk_config
        return IngestionReport(
            source.uri,
            collection.name,
            "indexed",
            "document-id",
            1,
            "hash",
            "fingerprint",
            ProvenanceStatus.COMPLETE,
        )

    monkeypatch.setattr(cli, "PostgresRepository", FakeRepository)
    monkeypatch.setattr(cli, "ingest", fake_ingest)
    monkeypatch.setattr(cli.sys, "stdin", TtyInput("manuals\n"))

    assert cli.main([str(source_path), "--dsn", "postgresql://test"]) == 0

    streams = capsys.readouterr()
    assert "Existing collections:" in streams.err
    assert "  - manuals" in streams.err
    assert json.loads(streams.out)["collection"] == "manuals"
    assert captured["collection"] == existing
    assert captured["chunk_config"] == ChunkingConfig(240, 80, 360, 75.0, 0)


def test_cli_creates_new_collection_from_current_defaults_after_empty_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Guide", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeRepository:
        def __init__(self, _dsn: str) -> None:
            pass

        def migrate(self) -> None:
            pass

        def list_collections(self) -> list[CollectionConfig]:
            return []

    def fake_ingest(source, collection, *, dsn, use_jina, chunk_config):
        captured["collection"] = collection
        captured["chunk_config"] = chunk_config
        return IngestionReport(
            source.uri,
            collection.name,
            "indexed",
            "document-id",
            1,
            "hash",
            "fingerprint",
            ProvenanceStatus.COMPLETE,
        )

    monkeypatch.setattr(cli, "PostgresRepository", FakeRepository)
    monkeypatch.setattr(cli, "ingest", fake_ingest)
    monkeypatch.setattr(cli.sys, "stdin", TtyInput("new-manuals\n"))

    assert cli.main([str(source_path)]) == 0

    streams = capsys.readouterr()
    assert "  (none)" in streams.err
    assert json.loads(streams.out)["collection"] == "new-manuals"
    collection = captured["collection"]
    assert collection.name == "new-manuals"
    assert captured["chunk_config"] == ChunkingConfig()


def test_cli_requires_collection_without_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Guide", encoding="utf-8")
    monkeypatch.setattr(cli.sys, "stdin", StringIO())

    with pytest.raises(SystemExit) as raised:
        cli.main([str(source_path)])

    assert raised.value.code == 2
    assert "--collection is required when stdin is not a TTY" in capsys.readouterr().err


def test_existing_collection_rejects_incompatible_explicit_chunk_override() -> None:
    existing = CollectionConfig(
        name="manuals",
        chunk_config={"target_tokens": 256, "min_tokens": 80, "max_tokens": 384},
    )

    with pytest.raises(ValueError, match="--target-tokens=512 is incompatible"):
        cli._resolve_collection("manuals", [existing], {"target_tokens": 512})


def test_cli_rejects_incompatible_override_before_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Guide", encoding="utf-8")
    existing = CollectionConfig(
        name="manuals",
        chunk_config={"target_tokens": 256, "min_tokens": 80, "max_tokens": 384},
    )

    class FakeRepository:
        def __init__(self, _dsn: str) -> None:
            pass

        def migrate(self) -> None:
            pass

        def list_collections(self) -> list[CollectionConfig]:
            return [existing]

    monkeypatch.setattr(cli, "PostgresRepository", FakeRepository)
    monkeypatch.setattr(
        cli, "ingest", lambda *_args, **_kwargs: pytest.fail("ingest must not be called")
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [str(source_path), "--collection", "manuals", "--target-tokens", "512"]
        )

    assert raised.value.code == 2


@pytest.mark.parametrize("name", ["", "   "])
def test_collection_name_cannot_be_empty(name: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        cli._resolve_collection(name, [], {})


def test_new_collection_cannot_differ_only_by_case() -> None:
    existing = CollectionConfig(name="Attention")

    with pytest.raises(ValueError, match="differs only by case"):
        cli._resolve_collection("attention", [existing], {})


def test_prompt_rejects_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", TtyInput())

    with pytest.raises(ValueError, match="end of input"):
        cli._prompt_collection([])
