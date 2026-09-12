from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import raglab.demo_command as command
from raglab.contracts import CollectionConfig
from raglab.corpus import CorpusManifest, load_corpus_manifest
from raglab.demo_command import _read_tunnel_url, _stop_process, prepare, share
from raglab.demo_evidence import InvalidDemoEvidence


def test_browser_consumes_fragment_token_without_persistent_storage() -> None:
    script = Path("src/raglab/demo_assets/app.js").read_text(encoding="utf-8")
    assert "window.location.hash" in script
    assert "history.replaceState" in script
    assert "localStorage" not in script
    assert "document.cookie" not in script
    assert "Bearer ${sessionToken}" in script


def test_scorecard_explains_bounded_curated_metrics() -> None:
    page = Path("src/raglab/demo_assets/index.html").read_text(encoding="utf-8")
    script = Path("src/raglab/demo_assets/app.js").read_text(encoding="utf-8")

    assert "deliberately small, curated dataset" in page
    assert "does not replay stored scores" in page
    assert "not a general benchmark" in page
    for metric in (
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "fact_coverage",
        "grounded_fact_coverage",
        "citation_precision",
        "abstention_accuracy",
    ):
        assert metric in script
    assert "Each answer case labels one source" in script


def test_launcher_extracts_quick_tunnel_url_and_cleans_up_process() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('https://random.trycloudflare.com', flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        assert _read_tunnel_url(process, timeout=2) == "https://random.trycloudflare.com"
    finally:
        _stop_process(process)
    assert process.poll() is not None


def test_share_rejects_missing_evidence_before_starting_processes(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(command, "_require_clean_tracked_tree", lambda root: None)
        with pytest.raises(InvalidDemoEvidence, match="missing"):
            share(root=tmp_path, cloudflared=tmp_path / "cloudflared")


def test_prepare_canonicalizes_tuple_traces_before_gate_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SimpleNamespace(
        dsn="postgresql://test",
        ollama_base_url="http://ollama",
        embedding_model="embed",
        generation_model="generate",
        embedding_num_gpu=None,
        keep_alive="5m",
        num_ctx=4096,
    )
    evaluation_data = {
        "traces": ({"case": {"id": "case-1"}},),
        "report": {"dataset_id": "demo"},
        "metadata": {"dataset_sha256": "f" * 64},
    }
    evaluation = SimpleNamespace(to_json=lambda: json.dumps(evaluation_data))
    application = SimpleNamespace(run=lambda dataset: evaluation)
    ingestion = SimpleNamespace()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(command, "_require_clean_tracked_tree", lambda root: None)
    monkeypatch.setattr(command.LiveEvaluationSettings, "from_env", lambda: settings)
    monkeypatch.setattr(command, "_run", lambda *args, **kwargs: SimpleNamespace(stdout="GPU"))
    monkeypatch.setattr(command, "PostgresRepository", lambda dsn: object())
    monkeypatch.setattr(command, "_wait_for_postgres", lambda repository: None)
    monkeypatch.setattr(
        command,
        "_ollama_models",
        lambda url: [{"name": "embed", "digest": "1"}, {"name": "generate", "digest": "2"}],
    )
    monkeypatch.setattr(command, "load_corpus_manifest", lambda path: object())
    monkeypatch.setattr(command, "_reconcile_demo_collections", lambda *args: ())
    monkeypatch.setattr(command, "_verify_demo_collections", lambda *args: None)
    monkeypatch.setattr(
        command,
        "CorpusIngestionApplication",
        lambda ingest: SimpleNamespace(run=lambda *args, **kwargs: ingestion),
    )
    monkeypatch.setattr(
        command,
        "publish_receipt",
        lambda run, path: path.parent.mkdir(parents=True) or path.write_text("{}"),
    )
    monkeypatch.setattr(
        command, "create_live_evaluation_application", lambda *args, **kwargs: application
    )
    monkeypatch.setattr(command, "load_cases", lambda path: object())
    monkeypatch.setattr(command, "_read_object", lambda path: {"baseline": True})
    def validate_promotion(run: dict[str, Any], baseline: dict[str, Any]) -> None:
        assert baseline == {"baseline": True}
        assert isinstance(run["traces"], list)
        seen["validated"] = run

    monkeypatch.setattr(command, "validate_promotion", validate_promotion)
    monkeypatch.setattr(command, "sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(command, "current_commit", lambda root: "a" * 40)
    monkeypatch.setattr(
        command,
        "build_evidence",
        lambda evaluation, *args, **kwargs: seen.update(evaluation=evaluation)
        or {"prepared": {"commit": "a" * 40}},
    )
    monkeypatch.setattr(
        command, "write_evidence", lambda path, value: seen.update(path=path, value=value)
    )

    result = prepare(root=tmp_path)

    assert result["prepared"]["commit"] == "a" * 40
    assert seen["path"] == tmp_path / command.EVIDENCE
    assert seen["evaluation"] == seen["validated"]
    assert isinstance(seen["evaluation"]["traces"], list)


def test_share_prints_fragment_url_and_stops_both_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stopped: list[object] = []
    installed: dict[object, object] = {}
    restored: list[tuple[object, object]] = []
    monkeypatch.setattr(command, "_require_clean_tracked_tree", lambda root: None)
    monkeypatch.setattr(command, "load_verified_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(command, "_validate_runtime_identity", lambda *args: None)
    monkeypatch.setattr(command.LiveEvaluationSettings, "from_env", lambda: object())
    monkeypatch.setattr(command, "_verify_cloudflared", lambda path: None)
    monkeypatch.setattr(command, "_wait_for_http", lambda *args: None)
    monkeypatch.setattr(
        command, "_read_tunnel_url", lambda process: "https://demo.trycloudflare.com"
    )
    monkeypatch.setattr(command.secrets, "token_urlsafe", lambda size: "session-token")
    monkeypatch.setattr(command, "_stop_process", lambda process: stopped.append(process))
    monkeypatch.setattr(command.signal, "getsignal", lambda signum: f"previous-{signum}")

    def record_signal(signum: object, handler: object) -> None:
        if callable(handler):
            installed[signum] = handler
        else:
            restored.append((signum, handler))

    monkeypatch.setattr(command.signal, "signal", record_signal)

    def request_shutdown(seconds: float) -> None:
        handler = installed[command.signal.SIGTERM]
        assert callable(handler)
        handler(command.signal.SIGTERM, None)

    monkeypatch.setattr(command.time, "sleep", request_shutdown)
    server = SimpleNamespace(pid=1, poll=lambda: None)
    tunnel = SimpleNamespace(pid=2, poll=lambda: None)
    monkeypatch.setattr(
        command.subprocess, "Popen", lambda *args, **kwargs: server if kwargs.get("env") else tunnel
    )

    assert share(root=tmp_path, cloudflared=tmp_path / "cloudflared") == ""
    assert "https://demo.trycloudflare.com/#token=session-token" in capsys.readouterr().out
    assert stopped == [tunnel, server]
    assert restored == [
        (command.signal.SIGTERM, f"previous-{command.signal.SIGTERM}"),
        (command.signal.SIGHUP, f"previous-{command.signal.SIGHUP}"),
    ]


def test_ollama_inventory_and_command_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"models":[{"name":"qwen3:4b","digest":"abc"}]}'

    monkeypatch.setattr(command, "urlopen", lambda *args, **kwargs: Response())
    assert command._ollama_models("http://ollama")[0]["digest"] == "abc"
    assert (
        command._required_ollama_model([{"name": "qwen3:4b", "digest": "abc"}], "qwen3:4b")[
            "digest"
        ]
        == "abc"
    )
    with pytest.raises(RuntimeError, match="missing"):
        command._required_ollama_model([], "missing")

    monkeypatch.setattr(
        command, "prepare", lambda: {"prepared": {"commit": "a", "created_at": "now"}}
    )
    assert command.main(["prepare"]) == 0


def test_cloudflared_pin_and_json_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "cloudflared"
    binary.write_text("binary")
    binary.chmod(0o755)
    monkeypatch.setattr(command, "sha256_file", lambda path: command.CLOUDFLARED_SHA256)
    command._verify_cloudflared(binary)
    monkeypatch.setattr(command, "sha256_file", lambda path: "wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        command._verify_cloudflared(binary)
    with pytest.raises(RuntimeError, match="not installed"):
        command._verify_cloudflared(tmp_path / "missing")

    payload = tmp_path / "payload.json"
    payload.write_text('{"ok":true}')
    assert command._read_object(payload) == {"ok": True}
    payload.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        command._read_object(payload)


def test_runtime_identity_detects_model_or_configuration_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = command.LiveEvaluationSettings(
        embedding_model="embed", generation_model="generate", num_ctx=4096, keep_alive="5m"
    )
    monkeypatch.setattr(
        command,
        "_ollama_models",
        lambda url: [{"name": "embed", "digest": "one"}, {"name": "generate", "digest": "two"}],
    )
    evidence = {
        "models": {
            "embedding": {"name": "embed", "digest": "one"},
            "generation": {"name": "generate", "digest": "two"},
        },
        "evaluation": {
            "metadata": {
                "retrieval_config": command.asdict(
                    command.RetrievalConfig(candidate_k=50, top_k=3, rerank=True, mmr=True)
                ),
                "generation_config": command.asdict(
                    command.GenerationConfig(
                        model="generate",
                        num_ctx=4096,
                        num_predict=512,
                        keep_alive="5m",
                        minimum_sources=1,
                    )
                ),
            }
        },
    }
    command._validate_runtime_identity(evidence, settings)
    evidence["models"]["generation"]["digest"] = "changed"
    with pytest.raises(InvalidDemoEvidence, match="generation model"):
        command._validate_runtime_identity(evidence, settings)


def test_prepare_and_share_reject_tracked_changes_but_ignore_untracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M src/raglab/demo.py\n"),
    )
    with pytest.raises(InvalidDemoEvidence, match="tracked Git working tree is dirty"):
        command._require_clean_tracked_tree(tmp_path)

    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=""),
    )
    (tmp_path / "ignored-artifact.json").write_text("ignored")
    command._require_clean_tracked_tree(tmp_path)


class _DemoRepository:
    def __init__(
        self,
        configs: dict[str, CollectionConfig],
        counts: dict[str, int],
    ) -> None:
        self.configs = configs
        self.counts = counts
        self.deleted: list[str] = []

    def collection_config(self, name: str) -> CollectionConfig | None:
        return self.configs.get(name)

    def collection_stats(self, name: str) -> dict[str, object] | None:
        return {"document_count": self.counts[name]} if name in self.counts else None

    def delete_collection(self, name: str) -> bool:
        self.deleted.append(name)
        self.configs.pop(name, None)
        self.counts.pop(name, None)
        return True


def _compatible_receipt(path: Path, manifest: CorpusManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus": manifest.corpus,
                "version": manifest.version,
                "manifest_fingerprint": manifest.fingerprint,
                "source_count": len(manifest.sources),
                "collections": list(manifest.collections),
                "configuration": {
                    "embedding_model": CollectionConfig(name="probe").model,
                    "dimension": CollectionConfig(name="probe").dimension,
                },
                "sources": [{"collection": source.collection} for source in manifest.sources],
            }
        )
    )


def test_reconciliation_preserves_compatible_demo_collections(tmp_path: Path) -> None:
    manifest = load_corpus_manifest("data/demo/raspberry_pi_v1.json")
    receipt = tmp_path / "receipt.json"
    _compatible_receipt(receipt, manifest)
    repository = _DemoRepository(
        {name: command._demo_collection_config(name) for name in manifest.collections},
        {name: 3 for name in manifest.collections},
    )

    reset = command._reconcile_demo_collections(repository, manifest, receipt)  # type: ignore[arg-type]

    assert reset == ()
    assert repository.deleted == []


def test_reconciliation_resets_only_legacy_or_polluted_manifest_collections(
    tmp_path: Path,
) -> None:
    manifest = load_corpus_manifest("data/demo/raspberry_pi_v1.json")
    receipt = tmp_path / "receipt.json"
    _compatible_receipt(receipt, manifest)
    computer, microcontroller, camera = manifest.collections
    legacy = command._demo_collection_config(computer)
    legacy = CollectionConfig(
        name=legacy.name,
        chunk_config={**legacy.chunk_config, "semantic_percentile": 85.0},
    )
    unrelated = CollectionConfig(name="unrelated")
    repository = _DemoRepository(
        {
            computer: legacy,
            microcontroller: command._demo_collection_config(microcontroller),
            camera: command._demo_collection_config(camera),
            "unrelated": unrelated,
        },
        {computer: 3, microcontroller: 4, camera: 3, "unrelated": 99},
    )

    reset = command._reconcile_demo_collections(repository, manifest, receipt)  # type: ignore[arg-type]

    assert reset == (computer, microcontroller)
    assert repository.deleted == [computer, microcontroller]
    assert repository.configs["unrelated"] == unrelated


def test_missing_receipt_forces_existing_demo_rebuild_and_postflight_checks_counts(
    tmp_path: Path,
) -> None:
    manifest = load_corpus_manifest("data/demo/raspberry_pi_v1.json")
    repository = _DemoRepository(
        {name: command._demo_collection_config(name) for name in manifest.collections},
        {name: 3 for name in manifest.collections},
    )
    assert (
        command._reconcile_demo_collections(  # type: ignore[arg-type]
            repository, manifest, tmp_path / "missing.json"
        )
        == manifest.collections
    )

    rebuilt = _DemoRepository(
        {name: command._demo_collection_config(name) for name in manifest.collections},
        {name: 3 for name in manifest.collections},
    )
    command._verify_demo_collections(rebuilt, manifest)  # type: ignore[arg-type]
    rebuilt.counts[manifest.collections[0]] = 2
    with pytest.raises(RuntimeError, match="exactly 3 documents"):
        command._verify_demo_collections(rebuilt, manifest)  # type: ignore[arg-type]
