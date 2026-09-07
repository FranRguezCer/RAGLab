"""Validated, manifest-driven corpus ingestion and atomic receipts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from raglab.chunking import ChunkingConfig
from raglab.contracts import CollectionConfig, IngestionReport, SourceInput

CORPUS_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CorpusSource:
    id: str
    path: Path
    sha256: str
    collection: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    schema_version: int
    corpus: str
    version: str
    collections: tuple[str, ...]
    sources: tuple[CorpusSource, ...]
    path: Path

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusIngestionRun:
    schema_version: int
    corpus: str
    version: str
    manifest_fingerprint: str
    started_at: str
    duration_seconds: float
    source_count: int
    indexed_count: int
    skipped_count: int
    chunk_count: int
    collections: tuple[str, ...]
    configuration: Mapping[str, Any]
    sources: tuple[IngestionReport, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    manifest_path = Path(path).resolve()
    raw = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    expected = {"schema_version", "corpus", "version", "collections", "sources"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("manifest has invalid fields")
    if raw["schema_version"] != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"unsupported corpus schema version: {raw['schema_version']}")
    corpus = _text(raw["corpus"], "corpus")
    version = _text(raw["version"], "version")
    collections = tuple(_text(item, "collection") for item in _list(raw["collections"]))
    if not collections or len(collections) != len(set(collections)):
        raise ValueError("collections must be non-empty and unique")
    source_rows = _list(raw["sources"])
    if not source_rows:
        raise ValueError("sources cannot be empty")
    sources: list[CorpusSource] = []
    for index, value in enumerate(source_rows):
        if not isinstance(value, dict):
            raise ValueError(f"sources[{index}] must be an object")
        row = cast(dict[str, object], value)
        if set(row) != {"id", "path", "sha256", "collection", "metadata"}:
            raise ValueError(f"sources[{index}] has invalid fields")
        source_path = (manifest_path.parent / _text(row["path"], "path")).resolve()
        try:
            source_path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ValueError(f"sources[{index}].path escapes the manifest directory") from exc
        digest = _text(row["sha256"], "sha256").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"sources[{index}].sha256 must be a SHA-256 hex digest")
        collection = _text(row["collection"], "collection")
        if collection not in collections:
            raise ValueError(f"sources[{index}] references undeclared collection {collection!r}")
        metadata = row["metadata"]
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise ValueError(f"sources[{index}].metadata must be an object")
        sources.append(
            CorpusSource(_text(row["id"], "id"), source_path, digest, collection, metadata)
        )
    if len({source.id for source in sources}) != len(sources):
        raise ValueError("source IDs must be unique")
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("source paths must be unique")
    for source in sources:
        try:
            actual = hashlib.sha256(source.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot read source {source.id!r}: {exc}") from exc
        if actual != source.sha256:
            raise ValueError(f"SHA-256 mismatch for source {source.id!r}")
    return CorpusManifest(
        CORPUS_SCHEMA_VERSION, corpus, version, collections, tuple(sources), manifest_path
    )


class CorpusIngestionApplication:
    """Validate all inputs first, then reuse the single-source ingestion boundary."""

    def __init__(self, ingest_source: Callable[..., IngestionReport]) -> None:
        self._ingest_source = ingest_source

    def run(
        self,
        manifest: CorpusManifest,
        *,
        dsn: str,
        ollama_base_url: str = "http://127.0.0.1:11434",
        embedding_num_gpu: int | None = None,
        keep_alive: str | None = None,
    ) -> CorpusIngestionRun:
        started = time.time()
        reports: list[IngestionReport] = []
        config = ChunkingConfig()
        for item in manifest.sources:
            collection = CollectionConfig(
                name=item.collection,
                chunk_config={
                    "strategy": "structure_plus_semantics",
                    "target_tokens": config.target_tokens,
                    "min_tokens": config.min_tokens,
                    "max_tokens": config.max_tokens,
                    "semantic_percentile": config.semantic_percentile,
                    "overlap_tokens": config.overlap_tokens,
                },
            )
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "corpus": manifest.corpus,
                    "corpus_version": manifest.version,
                    "source_id": item.id,
                }
            )
            original_url = metadata.get("original_url")
            source = (
                SourceInput.text(
                    item.path.read_text(encoding="utf-8"),
                    uri=original_url,
                    **metadata,
                )
                if isinstance(original_url, str) and original_url.startswith(("http://", "https://"))
                else SourceInput.path(item.path, **metadata)
            )
            reports.append(
                self._ingest_source(
                    source,
                    collection,
                    dsn=dsn,
                    chunk_config=config,
                    ollama_base_url=ollama_base_url,
                    embedding_num_gpu=embedding_num_gpu,
                    keep_alive=keep_alive,
                )
            )
        return CorpusIngestionRun(
            RECEIPT_SCHEMA_VERSION,
            manifest.corpus,
            manifest.version,
            manifest.fingerprint,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            round(time.time() - started, 6),
            len(reports),
            sum(report.status == "indexed" for report in reports),
            sum(report.status == "skipped" for report in reports),
            sum(report.chunk_count for report in reports),
            manifest.collections,
            {
                "embedding_model": "qwen3-embedding:0.6b",
                "dimension": 1024,
                "ollama_base_url": ollama_base_url,
                "embedding_num_gpu": embedding_num_gpu,
            },
            tuple(reports),
        )


def publish_receipt(run: CorpusIngestionRun, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(run.to_json(), encoding="utf-8")
    os.replace(temporary, destination)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("manifest arrays must be arrays")
    return value
