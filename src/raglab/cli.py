"""Small command-line adapter for the end-to-end ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from raglab.chunking import ChunkingConfig
from raglab.contracts import CollectionConfig, SourceInput
from raglab.errors import RagLabError
from raglab.pipeline import ingest
from raglab.storage import PostgresRepository

DEFAULT_DSN = "postgresql://raglab:raglab@127.0.0.1:5432/raglab"
DEFAULT_CHUNK_CONFIG = ChunkingConfig()
CHUNK_ARGUMENTS = {
    "target_tokens": "--target-tokens",
    "min_tokens": "--min-tokens",
    "max_tokens": "--max-tokens",
    "semantic_percentile": "--semantic-percentile",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab-ingest",
        description=(
            "Convert a local file or HTTP(S) URL to canonical Markdown, chunk it, "
            "embed it with Ollama, and index it in PostgreSQL + pgvector."
        ),
    )
    parser.add_argument("source", help="Local file path or public HTTP(S) URL")
    parser.add_argument(
        "--collection",
        help=(
            "Collection name. When omitted in a terminal, existing collections are listed "
            "before prompting"
        ),
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN (defaults to RAGLAB_DSN or the Compose database)",
    )
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--semantic-percentile", type=float)
    parser.add_argument(
        "--use-jina",
        action="store_true",
        help="Explicitly send a public URL to Jina Reader instead of converting it locally",
    )
    return parser


def _source_input(value: str, *, use_jina: bool) -> SourceInput:
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"}:
        return SourceInput.url(value, allow_remote_service=use_jina)
    if parsed.scheme:
        raise ValueError("source must be a local path or an HTTP(S) URL")
    if use_jina:
        raise ValueError("--use-jina accepts only a public HTTP(S) URL")
    return SourceInput.path(Path(value).expanduser().resolve())


def _collection(name: str, config: ChunkingConfig) -> CollectionConfig:
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


def _chunk_overrides(args: argparse.Namespace) -> dict[str, int | float]:
    return {
        name: value
        for name in CHUNK_ARGUMENTS
        if (value := getattr(args, name)) is not None
    }


def _chunk_config(values: dict[str, Any] | None = None) -> ChunkingConfig:
    stored = values or {}
    return ChunkingConfig(
        target_tokens=int(stored.get("target_tokens", DEFAULT_CHUNK_CONFIG.target_tokens)),
        min_tokens=int(stored.get("min_tokens", DEFAULT_CHUNK_CONFIG.min_tokens)),
        max_tokens=int(stored.get("max_tokens", DEFAULT_CHUNK_CONFIG.max_tokens)),
        semantic_percentile=float(
            stored.get("semantic_percentile", DEFAULT_CHUNK_CONFIG.semantic_percentile)
        ),
        overlap_tokens=int(stored.get("overlap_tokens", DEFAULT_CHUNK_CONFIG.overlap_tokens)),
    )


def _prompt_collection(collections: Sequence[CollectionConfig]) -> str:
    print("Existing collections:", file=sys.stderr)
    if collections:
        for collection in collections:
            print(f"  - {collection.name}", file=sys.stderr)
    else:
        print("  (none)", file=sys.stderr)
    print("Collection name (existing or new): ", end="", file=sys.stderr, flush=True)
    value = sys.stdin.readline()
    if value == "":
        raise ValueError("no collection name received (end of input)")
    return value


def _resolve_collection(
    name: str,
    collections: Sequence[CollectionConfig],
    overrides: dict[str, int | float],
) -> tuple[CollectionConfig, ChunkingConfig]:
    normalized = name.strip()
    if not normalized:
        raise ValueError("collection name cannot be empty")

    existing = next((item for item in collections if item.name == normalized), None)
    if existing is not None:
        stored_chunk_config = dict(existing.chunk_config)
        effective_config = _chunk_config(stored_chunk_config)
        for field, value in overrides.items():
            if getattr(effective_config, field) != value:
                raise ValueError(
                    f"{CHUNK_ARGUMENTS[field]}={value} is incompatible with existing "
                    f"collection {normalized!r} ({getattr(effective_config, field)})"
                )
        return existing, effective_config

    collision = next(
        (item.name for item in collections if item.name.casefold() == normalized.casefold()), None
    )
    if collision is not None:
        raise ValueError(
            f"collection {normalized!r} differs only by case from existing collection {collision!r}"
        )

    chunk_config = _chunk_config(dict(overrides))
    return _collection(normalized, chunk_config), chunk_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.collection is None and not sys.stdin.isatty():
        parser.error("--collection is required when stdin is not a TTY")
    try:
        source = _source_input(args.source, use_jina=args.use_jina)
    except ValueError as exc:
        parser.error(str(exc))

    repository = PostgresRepository(args.dsn)
    try:
        repository.migrate()
        collections = repository.list_collections()
    except (RagLabError, RuntimeError) as exc:
        parser.exit(1, f"raglab-ingest: error: {exc}\n")

    try:
        collection_name = (
            args.collection if args.collection is not None else _prompt_collection(collections)
        )
        collection, chunk_config = _resolve_collection(
            collection_name, collections, _chunk_overrides(args)
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        report = ingest(
            source,
            collection,
            dsn=args.dsn,
            use_jina=args.use_jina,
            chunk_config=chunk_config,
        )
    except (RagLabError, RuntimeError) as exc:
        parser.exit(1, f"raglab-ingest: error: {exc}\n")

    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
