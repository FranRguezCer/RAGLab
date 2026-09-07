from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from raglab.contracts import IngestionReport, ProvenanceStatus
from raglab.corpus import CorpusIngestionApplication, load_corpus_manifest, publish_receipt
from raglab.corpus_cli import main

MANIFEST = Path("data/demo/raspberry_pi_v1.json")


def _report(source: object, collection: object, **_: object) -> IngestionReport:
    return IngestionReport(
        source_uri=source.uri,  # type: ignore[union-attr]
        collection=collection.name,  # type: ignore[union-attr]
        status="indexed",
        document_id="document",
        chunk_count=2,
        content_hash="a" * 64,
        fingerprint="b" * 64,
        provenance_status=ProvenanceStatus.COMPLETE,
    )


def test_manifest_validates_nine_unique_pinned_sources() -> None:
    manifest = load_corpus_manifest(MANIFEST)

    assert len(manifest.sources) == 9
    assert set(manifest.collections) == {
        "rpi-computers",
        "rpi-microcontrollers",
        "rpi-camera-ai",
    }
    assert all(source.metadata["upstream_revision"] for source in manifest.sources)
    licenses = {source.id: source.metadata["license"] for source in manifest.sources}
    assert licenses["picamera2"] == "BSD-2-Clause"
    assert set(licenses.values()) == {"CC-BY-SA-4.0", "BSD-2-Clause"}


def test_hash_mismatch_fails_before_any_ingestion(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST.read_text())
    raw["sources"][0]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    raw["sources"][0]["path"] = str(MANIFEST.parent / raw["sources"][0]["path"])
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="escapes|mismatch"):
        load_corpus_manifest(path)


def test_application_preserves_metadata_and_publishes_complete_receipt(tmp_path: Path) -> None:
    manifest = load_corpus_manifest(MANIFEST)
    seen: list[object] = []

    def ingest(source: object, collection: object, **kwargs: object) -> IngestionReport:
        seen.append((source, collection, kwargs))
        return _report(source, collection, **kwargs)

    run = CorpusIngestionApplication(ingest).run(
        manifest,
        dsn="postgresql://test",
        ollama_base_url="http://ollama:11434",
        embedding_num_gpu=1,
        keep_alive="10m",
    )
    receipt = tmp_path / "receipt.json"
    publish_receipt(run, receipt)

    assert run.source_count == run.indexed_count == 9
    assert run.chunk_count == 18
    assert seen[0][0].metadata["source_id"] == "headless-setup"  # type: ignore[index,union-attr]
    assert seen[0][0].uri.startswith("https://github.com/raspberrypi/")  # type: ignore[index,union-attr]
    assert seen[0][2]["ollama_base_url"] == "http://ollama:11434"  # type: ignore[index]
    assert seen[0][2]["embedding_num_gpu"] == 1  # type: ignore[index]
    assert seen[0][2]["keep_alive"] == "10m"  # type: ignore[index]
    assert json.loads(receipt.read_text())["source_count"] == 9


def test_second_idempotent_run_reports_every_source_skipped() -> None:
    manifest = load_corpus_manifest(MANIFEST)

    def skipped(source: object, collection: object, **kwargs: object) -> IngestionReport:
        return replace(_report(source, collection, **kwargs), status="skipped", chunk_count=0)

    run = CorpusIngestionApplication(skipped).run(manifest, dsn="postgresql://test")

    assert run.indexed_count == 0
    assert run.skipped_count == 9
    assert run.chunk_count == 0


def test_corpus_cli_publishes_only_after_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = load_corpus_manifest(MANIFEST)
    monkeypatch.setattr("raglab.corpus_cli.load_corpus_manifest", lambda _path: manifest)
    monkeypatch.setattr("raglab.corpus_cli.PostgresRepository.migrate", lambda _self: None)
    monkeypatch.setattr("raglab.corpus_cli.ingest", _report)
    receipt = tmp_path / "receipt.json"

    assert main([str(MANIFEST), "--receipt", str(receipt)]) == 0

    assert json.loads(capsys.readouterr().out)["source_count"] == 9
    assert receipt.exists()


def test_corpus_cli_does_not_replace_last_good_receipt_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "raglab.corpus_cli.load_corpus_manifest", lambda _path: load_corpus_manifest(MANIFEST)
    )
    monkeypatch.setattr("raglab.corpus_cli.PostgresRepository.migrate", lambda _self: None)
    monkeypatch.setattr(
        "raglab.corpus_cli.ingest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text("last-good")

    with pytest.raises(SystemExit):
        main([str(MANIFEST), "--receipt", str(receipt)])

    assert receipt.read_text() == "last-good"
