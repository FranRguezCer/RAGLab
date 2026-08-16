"""Command-line adapter for independent hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from raglab.embeddings import OllamaEmbeddingProvider
from raglab.errors import RagLabError
from raglab.retrieval.models import MetadataFilter, RetrievalConfig, RetrievalRequest
from raglab.retrieval.pipeline import RetrievalPipeline
from raglab.retrieval.repository import PostgresRetrievalRepository
from raglab.retrieval.rewriting import OllamaQueryRewriter

DEFAULT_DSN = "postgresql://raglab:raglab@127.0.0.1:5432/raglab"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab-retrieve",
        description="Run inspectable ANN + BM25 hybrid retrieval with RRF.",
    )
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--dsn", default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN))
    parser.add_argument("--filter", action="append", default=[], dest="filters")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--expansions", type=int, default=0)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--no-small-to-big", action="store_true")
    parser.add_argument("--parent-max-tokens", type=int, default=1500)
    return parser


def parse_filter(value: str) -> MetadataFilter:
    try:
        left, raw_value = value.split("=", 1)
        field, operator = left.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("filter must use field:operator=value") from exc
    if not field or not operator or raw_value == "":
        raise ValueError("filter must use non-empty field:operator=value")
    if operator == "in":
        parsed = _json_or_string(raw_value)
        if isinstance(parsed, list):
            filter_value = tuple(_scalar(item) for item in parsed)
        else:
            filter_value = tuple(
                _scalar(_json_or_string(item.strip())) for item in raw_value.split(",")
            )
        return MetadataFilter(field, operator, filter_value)
    return MetadataFilter(field, operator, _scalar(_json_or_string(raw_value)))


def load_history(path: Path) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read history JSON from {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("history JSON must be a list")
    history: list[str] = []
    for item in payload:
        if isinstance(item, str):
            content = item
        elif isinstance(item, dict) and isinstance(item.get("content"), str):
            content = item["content"]
        else:
            raise ValueError("history entries must be strings or objects with string content")
        if content.strip():
            history.append(content.strip())
    return tuple(history)


def _json_or_string(value: str) -> str | int | float | bool | None | list[Any]:
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (str, int, float, bool, list)) or parsed is None:
        return parsed
    raise ValueError("filter values must be JSON scalars or lists")


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("'in' filter values must be scalar")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        filters = tuple(parse_filter(value) for value in args.filters)
        history = load_history(args.history_file) if args.history_file else ()
        config = RetrievalConfig(
            candidate_k=args.candidate_k,
            top_k=args.top_k,
            ef_search=args.ef_search,
            exact=args.exact,
            rewrite=args.rewrite,
            expansions=args.expansions,
            rerank=not args.no_rerank,
            mmr=not args.no_mmr,
            mmr_lambda=args.mmr_lambda,
            small_to_big=not args.no_small_to_big,
            parent_max_tokens=args.parent_max_tokens,
        )
        request = RetrievalRequest(
            query=args.query,
            collection=args.collection,
            filters=filters,
            history=history,
            config=config,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        pipeline = RetrievalPipeline(
            PostgresRetrievalRepository(args.dsn),
            OllamaEmbeddingProvider(),
            rewriter=OllamaQueryRewriter() if config.rewrite or history else None,
        )
        response = pipeline.retrieve(request)
    except (RagLabError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"raglab-retrieve: error: {exc}\n")
    print(json.dumps(asdict(response), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
