"""Operational command for preparing and temporarily sharing the local demo."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import selectors
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.error import URLError
from urllib.request import urlopen

from raglab.chunking import ChunkingConfig
from raglab.config import load_project_env
from raglab.contracts import CollectionConfig
from raglab.corpus import (
    CorpusIngestionApplication,
    CorpusManifest,
    load_corpus_manifest,
    publish_receipt,
)
from raglab.demo_evidence import (
    InvalidDemoEvidence,
    build_evidence,
    current_commit,
    load_verified_evidence,
    sha256_file,
    write_evidence,
)
from raglab.errors import RagLabError
from raglab.evaluation import load_cases
from raglab.evaluation_application import LiveEvaluationSettings, create_live_evaluation_application
from raglab.evaluation_gate import validate_promotion
from raglab.generation import GenerationConfig
from raglab.nli import NLI_MODEL, NLI_REVISION
from raglab.pipeline import ingest
from raglab.retrieval import RetrievalConfig
from raglab.retrieval.reranking import BGE_MODEL, BGE_REVISION
from raglab.storage import PostgresRepository

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path("data/demo/raspberry_pi_v1.json")
DATASET = Path("data/evaluation/raspberry_pi_demo_v1.json")
BASELINE = Path("data/evaluation/baselines/raspberry_pi_demo_v1.json")
EVIDENCE = Path("artifacts/demo/evidence.json")
INGESTION_RECEIPT = Path("artifacts/demo/ingestion.json")
CLOUDFLARED_SHA256 = "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def prepare(*, root: Path = ROOT) -> dict[str, Any]:
    """Start PostgreSQL, validate native GPU inference, and publish gated evidence."""

    _require_clean_tracked_tree(root)
    settings = LiveEvaluationSettings.from_env()
    _run(["docker", "compose", "up", "-d", "postgres"], cwd=root)
    repository = PostgresRepository(settings.dsn)
    _wait_for_postgres(repository)
    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        cwd=root,
    ).stdout.strip()
    if not gpu:
        raise RuntimeError("CUDA GPU was not detected by nvidia-smi")
    ollama_models = _ollama_models(settings.ollama_base_url)
    models = {
        "embedding": _required_ollama_model(ollama_models, settings.embedding_model),
        "generation": _required_ollama_model(ollama_models, settings.generation_model),
        "reranker": {"name": BGE_MODEL, "revision": BGE_REVISION},
        "nli": {"name": NLI_MODEL, "revision": NLI_REVISION},
    }

    corpus_path = root / CORPUS
    receipt_path = root / INGESTION_RECEIPT
    manifest = load_corpus_manifest(corpus_path)
    _reconcile_demo_collections(repository, manifest, receipt_path)
    ingestion = CorpusIngestionApplication(ingest).run(
        manifest,
        dsn=settings.dsn,
        ollama_base_url=settings.ollama_base_url,
        embedding_num_gpu=settings.embedding_num_gpu,
        keep_alive=settings.keep_alive,
    )
    _verify_demo_collections(repository, manifest)
    publish_receipt(ingestion, receipt_path)

    retrieval = RetrievalConfig(candidate_k=50, top_k=3, rerank=True, mmr=True)
    generation = GenerationConfig(
        model=settings.generation_model,
        num_ctx=settings.num_ctx,
        num_predict=512,
        keep_alive=settings.keep_alive,
        minimum_sources=1,
    )
    evaluation = create_live_evaluation_application(
        settings,
        retrieval_config=retrieval,
        generation_config=generation,
    ).run(load_cases(root / DATASET))
    baseline = _read_object(root / BASELINE)
    evaluation_payload = evaluation.to_dict()
    validate_promotion(evaluation_payload, baseline)
    inputs = {str(path): sha256_file(root / path) for path in (CORPUS, DATASET, BASELINE)}
    evidence = build_evidence(
        evaluation_payload,
        baseline,
        cast(Mapping[str, Any], json.loads(receipt_path.read_text(encoding="utf-8"))),
        models,
        commit=current_commit(root),
        inputs=inputs,
    )
    write_evidence(root / EVIDENCE, evidence)
    return evidence


def share(
    *,
    root: Path = ROOT,
    cloudflared: Path | None = None,
    port: int = 8000,
) -> str:
    """Run Uvicorn and a Quick Tunnel together until interrupted or either fails."""

    _require_clean_tracked_tree(root)
    evidence_path = root / EVIDENCE
    verified = load_verified_evidence(evidence_path, project_root=root)
    _validate_runtime_identity(verified, LiveEvaluationSettings.from_env())
    executable = cloudflared or Path.home() / ".local/bin/cloudflared"
    _verify_cloudflared(executable)
    token = secrets.token_urlsafe(32)
    environment = {
        **os.environ,
        "RAGLAB_DEMO_TOKEN": token,
        "RAGLAB_DEMO_EVIDENCE": str(evidence_path),
        "RAGLAB_PROJECT_ROOT": str(root),
    }
    previous_handlers = _install_shutdown_handlers()
    server: subprocess.Popen[Any] | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    try:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "raglab.demo_cli:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=root,
            env=environment,
            start_new_session=True,
        )
        _wait_for_http(f"http://127.0.0.1:{port}/health/ready", server)
        tunnel = subprocess.Popen(
            [
                str(executable),
                "tunnel",
                "--url",
                f"http://127.0.0.1:{port}",
                "--no-autoupdate",
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        public = _read_tunnel_url(tunnel)
        shared = f"{public}/#token={token}"
        print(f"Share this URL for this session only:\n{shared}", flush=True)
        while server.poll() is None and tunnel.poll() is None:
            time.sleep(0.2)
        raise RuntimeError("demo server or Cloudflare tunnel stopped unexpectedly")
    except KeyboardInterrupt:
        return ""
    finally:
        _stop_process(tunnel)
        _stop_process(server)
        _restore_shutdown_handlers(previous_handlers)


def _ollama_models(base_url: str) -> list[dict[str, Any]]:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as response:  # noqa: S310
            payload = json.load(response)
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError(f"Ollama is unavailable at {base_url}") from exc
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("Ollama returned an invalid model inventory")
    return [cast(dict[str, Any], model) for model in models if isinstance(model, dict)]


def _required_ollama_model(models: list[dict[str, Any]], required: str) -> dict[str, str]:
    match = next((model for model in models if model.get("name") == required), None)
    digest = match.get("digest") if match else None
    if not isinstance(digest, str) or not digest:
        raise RuntimeError(f"required Ollama model is missing or unresolved: {required}")
    return {"name": required, "digest": digest}


def _validate_runtime_identity(
    evidence: Mapping[str, Any], settings: LiveEvaluationSettings
) -> None:
    """Ensure the current models and query configuration still match the gated run."""

    inventory = _ollama_models(settings.ollama_base_url)
    prepared_models = cast(Mapping[str, Any], evidence["models"])
    for role, name in (
        ("embedding", settings.embedding_model),
        ("generation", settings.generation_model),
    ):
        if prepared_models.get(role) != _required_ollama_model(inventory, name):
            raise InvalidDemoEvidence(f"prepared {role} model no longer matches Ollama")
    metadata = cast(Mapping[str, Any], cast(Mapping[str, Any], evidence["evaluation"])["metadata"])
    expected_retrieval = asdict(RetrievalConfig(candidate_k=50, top_k=3, rerank=True, mmr=True))
    expected_generation = asdict(
        GenerationConfig(
            model=settings.generation_model,
            num_ctx=settings.num_ctx,
            num_predict=512,
            keep_alive=settings.keep_alive,
            minimum_sources=1,
        )
    )
    if metadata.get("retrieval_config") != expected_retrieval:
        raise InvalidDemoEvidence("prepared retrieval configuration is outdated")
    if metadata.get("generation_config") != expected_generation:
        raise InvalidDemoEvidence("prepared generation configuration is outdated")


def _reconcile_demo_collections(
    repository: PostgresRepository, manifest: CorpusManifest, receipt_path: Path
) -> tuple[str, ...]:
    """Reset only manifest-owned collections whose last known state cannot be trusted."""

    expected_counts = Counter(source.collection for source in manifest.sources)
    expected_configs = {name: _demo_collection_config(name) for name in manifest.collections}
    receipt = _read_optional_object(receipt_path)
    receipt_compatible = _receipt_matches_manifest(receipt, manifest, expected_counts)
    reset: list[str] = []
    for name in manifest.collections:
        current = repository.collection_config(name)
        if current is None:
            continue
        stats = repository.collection_stats(name)
        compatible = (
            receipt_compatible
            and current == expected_configs[name]
            and stats is not None
            and stats.get("document_count") == expected_counts[name]
        )
        if not compatible:
            repository.delete_collection(name)
            reset.append(name)
    return tuple(reset)


def _verify_demo_collections(repository: PostgresRepository, manifest: CorpusManifest) -> None:
    expected_counts = Counter(source.collection for source in manifest.sources)
    for name in manifest.collections:
        if repository.collection_config(name) != _demo_collection_config(name):
            raise RuntimeError(f"demo collection {name!r} has an unexpected configuration")
        stats = repository.collection_stats(name)
        if stats is None or stats.get("document_count") != expected_counts[name]:
            raise RuntimeError(
                f"demo collection {name!r} must contain exactly {expected_counts[name]} documents"
            )


def _demo_collection_config(name: str) -> CollectionConfig:
    config = ChunkingConfig()
    return CollectionConfig(
        name=name,
        chunk_config={
            "strategy": "structure_plus_semantics",
            "target_tokens": config.target_tokens,
            "min_tokens": config.min_tokens,
            "max_tokens": config.max_tokens,
            "semantic_percentile": config.semantic_percentile,
            "overlap_tokens": config.overlap_tokens,
        },
    )


def _read_optional_object(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else None


def _receipt_matches_manifest(
    receipt: Mapping[str, Any] | None,
    manifest: CorpusManifest,
    expected_counts: Mapping[str, int],
) -> bool:
    if receipt is None:
        return False
    sources = receipt.get("sources")
    configuration = receipt.get("configuration")
    if not isinstance(sources, list) or not isinstance(configuration, dict):
        return False
    recorded_counts = Counter(
        source.get("collection")
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("collection"), str)
    )
    return (
        receipt.get("schema_version") == 1
        and receipt.get("corpus") == manifest.corpus
        and receipt.get("version") == manifest.version
        and receipt.get("manifest_fingerprint") == manifest.fingerprint
        and receipt.get("source_count") == len(manifest.sources)
        and tuple(receipt.get("collections", ())) == manifest.collections
        and recorded_counts == Counter(expected_counts)
        and configuration.get("embedding_model") == CollectionConfig(name="probe").model
        and configuration.get("dimension") == CollectionConfig(name="probe").dimension
    )


def _require_clean_tracked_tree(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise InvalidDemoEvidence(
            "tracked Git working tree is dirty; commit or restore tracked changes "
            "before prepare/share"
        )


def _install_shutdown_handlers() -> dict[signal.Signals, Any]:
    previous: dict[signal.Signals, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, stop)
    return previous


def _restore_shutdown_handlers(previous: Mapping[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _wait_for_postgres(repository: PostgresRepository, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            repository.migrate()
            repository.healthcheck()
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("PostgreSQL did not become ready") from exc
            time.sleep(1)


def _wait_for_http(url: str, process: subprocess.Popen[Any], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("local demo server failed during startup")
        try:
            with urlopen(url, timeout=1) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.2)
    raise RuntimeError("local demo server did not become ready")


def _read_tunnel_url(process: subprocess.Popen[bytes], timeout: float = 30) -> str:
    if process.stdout is None:
        raise RuntimeError("cloudflared output is unavailable")
    selector = selectors.DefaultSelector()
    selector.register(cast(BinaryIO, process.stdout), selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    captured = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        for key, _ in selector.select(timeout=0.2):
            line = cast(BinaryIO, key.fileobj).readline().decode(errors="replace")
            captured += line
            match = TUNNEL_URL.search(line)
            if match:
                return match.group(0)
    raise RuntimeError(f"Cloudflare Quick Tunnel URL was not produced: {captured[-500:]}")


def _verify_cloudflared(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"cloudflared is not installed and executable at {path}")
    if sha256_file(path) != CLOUDFLARED_SHA256:
        raise RuntimeError("cloudflared SHA-256 does not match the pinned 2026.9.0 binary")


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(Mapping[str, Any], value)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raglab-demo")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build and gate reusable local demo evidence")
    share_parser = commands.add_parser("share", help="Open the local UI through a temporary tunnel")
    share_parser.add_argument("--cloudflared", type=Path)
    share_parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            evidence = prepare()
            print(
                json.dumps(
                    {
                        "evidence": str(EVIDENCE),
                        "commit": evidence["prepared"]["commit"],
                        "evaluated_at": evidence["prepared"]["created_at"],
                    },
                    indent=2,
                )
            )
        else:
            share(cloudflared=args.cloudflared, port=args.port)
    except (
        InvalidDemoEvidence,
        OSError,
        RagLabError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as exc:
        parser.exit(1, f"raglab-demo: error: {exc}\n")
    return 0


def entrypoint() -> int:
    load_project_env()
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
