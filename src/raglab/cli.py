"""Small command-line adapter for the end-to-end ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from raglab.chunking import ChunkingConfig
from raglab.contracts import CollectionConfig, SourceInput
from raglab.errors import RagLabError
from raglab.pipeline import ingest
from raglab.storage import PostgresRepository

DEFAULT_DSN = "postgresql://raglab:raglab@127.0.0.1:5432/raglab"


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
        default="documents",
        help="Stable collection name for sources that share one chunking profile",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN),
        help="PostgreSQL DSN (defaults to RAGLAB_DSN or the Compose database)",
    )
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--semantic-percentile", type=float, default=90.0)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        source = _source_input(args.source, use_jina=args.use_jina)
        chunk_config = ChunkingConfig(
            target_tokens=args.target_tokens,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            semantic_percentile=args.semantic_percentile,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        PostgresRepository(args.dsn).migrate()
        report = ingest(
            source,
            _collection(args.collection, chunk_config),
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
