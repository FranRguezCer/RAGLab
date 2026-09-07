"""Command-line adapter for live dataset evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from raglab.config import load_project_env
from raglab.errors import RagLabError
from raglab.evaluation import load_cases
from raglab.evaluation_application import (
    DEFAULT_EVALUATION_DSN,
    LiveEvaluationSettings,
    create_live_evaluation_application,
)
from raglab.generation import GenerationConfig
from raglab.retrieval import RetrievalConfig

DEFAULT_DATASET = Path("data/evaluation/aster_greenhouse_controller_v1.json")


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
        prog="raglab-evaluate",
        description="Run every dataset case through PostgreSQL and Ollama, then score it.",
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--output", type=Path, help="Also write the complete JSON run to this path")
    parser.add_argument(
        "--dsn", default=os.environ.get("RAGLAB_DSN", DEFAULT_EVALUATION_DSN)
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-sources", type=int, default=1)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
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
        raise SystemExit(f"raglab-evaluate: error: {exc}") from exc
    args = parser.parse_args(argv)
    try:
        retrieval_config = RetrievalConfig(
            candidate_k=max(args.candidate_k, args.top_k),
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
        generation_config = GenerationConfig(
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            keep_alive=args.keep_alive,
            minimum_sources=args.minimum_sources,
        )
        settings = LiveEvaluationSettings(
            dsn=args.dsn,
            embedding_model=args.embedding_model,
            generation_model=args.model,
            ollama_base_url=args.ollama_base_url,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
        )
        dataset = load_cases(args.dataset)
        application = create_live_evaluation_application(
            settings,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )
        run = application.run(dataset, collection=args.collection)
        serialized = run.to_json()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(f".{args.output.name}.tmp")
            temporary.write_text(serialized, encoding="utf-8")
            os.replace(temporary, args.output)
    except (OSError, RagLabError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"raglab-evaluate: error: {exc}\n")
    summary = {
        "dataset_id": run.report.dataset_id,
        "case_count": len(run.report.case_ids),
        "retrieval_summary": run.report.retrieval_summary,
        "generation_summary": run.report.generation_summary,
        "duration_seconds": run.metadata["duration_seconds"],
        "output": str(args.output) if args.output is not None else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def entrypoint() -> int:
    """Load project configuration before running the installed CLI."""

    load_project_env()
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
