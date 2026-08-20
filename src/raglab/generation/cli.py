"""Command-line adapter for one-shot, strict RAG generation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from raglab.embeddings import OllamaEmbeddingProvider
from raglab.errors import RagLabError
from raglab.generation.models import GenerationConfig, GenerationRequest
from raglab.generation.ollama import OllamaGenerationModel
from raglab.generation.pipeline import GenerationPipeline
from raglab.retrieval import RetrievalConfig, RetrievalPipeline, RetrievalRequest
from raglab.retrieval.cli import DEFAULT_DSN, load_history, parse_filter
from raglab.retrieval.repository import PostgresRetrievalRepository
from raglab.retrieval.rewriting import OllamaQueryRewriter


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab-generate",
        description="Retrieve evidence and generate one strict, cited JSON answer.",
    )
    parser.add_argument("query", help="Question to answer")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--dsn", default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN))
    parser.add_argument("--filter", action="append", default=[], dest="filters")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-sources", type=int, default=5)
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
    parser.add_argument(
        "--model", default=os.environ.get("RAGLAB_GENERATION_MODEL", "qwen3:4b")
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("RAGLAB_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("RAGLAB_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--num-ctx", type=int, default=_env_int("RAGLAB_NUM_CTX", 12_288))
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument(
        "--keep-alive", default=os.environ.get("RAGLAB_KEEP_ALIVE", "5m")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = _parser()
    except ValueError as exc:
        raise SystemExit(f"raglab-generate: error: {exc}") from exc
    args = parser.parse_args(argv)
    try:
        minimum = args.minimum_sources
        top_k = max(args.top_k, minimum)
        retrieval_config = RetrievalConfig(
            candidate_k=max(args.candidate_k, top_k),
            top_k=top_k,
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
        retrieval_request = RetrievalRequest(
            args.query,
            collection=args.collection,
            filters=tuple(parse_filter(value) for value in args.filters),
            history=load_history(args.history_file) if args.history_file else (),
            config=retrieval_config,
        )
        generation_config = GenerationConfig(
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            parallelism=1,
            keep_alive=args.keep_alive,
            minimum_sources=minimum,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        embedding = OllamaEmbeddingProvider(
            model=args.embedding_model,
            dimension=1024,
            base_url=args.ollama_base_url,
            num_gpu=0,
            num_ctx=4096,
            keep_alive=args.keep_alive,
        )
        retrieval = RetrievalPipeline(
            PostgresRetrievalRepository(args.dsn),
            embedding,
            rewriter=(
                OllamaQueryRewriter(model=args.model, base_url=args.ollama_base_url)
                if args.rewrite or retrieval_request.history
                else None
            ),
        )
        pipeline = GenerationPipeline(
            retrieval,
            OllamaGenerationModel(base_url=args.ollama_base_url),
            embedding_model=args.embedding_model,
        )
        response = pipeline.generate(GenerationRequest(retrieval_request, generation_config))
    except (RagLabError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"raglab-generate: error: {exc}\n")
    print(json.dumps(asdict(response), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
