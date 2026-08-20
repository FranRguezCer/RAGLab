"""Thin command-line adapter for reproducible RAG evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from raglab.config import load_project_env
from raglab.errors import RagLabError
from raglab.evaluation.application import EvaluationApplication
from raglab.evaluation.manifest import load_manifest
from raglab.evaluation.models import EvaluationExecutor
from raglab.evaluation.runtime import LiveEvaluationExecutor, OllamaEvaluationJudge
from raglab.retrieval.cli import DEFAULT_DSN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raglab-evaluate",
        description="Run, compare, and explicitly promote reproducible RAG evaluations.",
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/evaluation")
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Build an evaluation candidate")
    run.add_argument("--profile", choices=("core", "live"), default="core")
    run.add_argument("--manifest", type=Path)
    run.add_argument("--reuse-index", action="store_true")
    run.add_argument("--judge-model")
    run.add_argument("--dsn", default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN))
    run.add_argument(
        "--model", default=os.environ.get("RAGLAB_GENERATION_MODEL", "qwen3:4b")
    )
    run.add_argument(
        "--embedding-model",
        default=os.environ.get("RAGLAB_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
    )
    run.add_argument(
        "--ollama-base-url",
        default=os.environ.get("RAGLAB_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )

    compare = subcommands.add_parser("compare", help="Compare compatible candidate and baseline")
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--baseline", type=Path)

    baseline = subcommands.add_parser("baseline", help="Manage the approved baseline")
    baseline_subcommands = baseline.add_subparsers(dest="baseline_command", required=True)
    promote = baseline_subcommands.add_parser("promote", help="Promote one clean complete run")
    promote.add_argument("run", type=Path)
    promote.add_argument("--destination", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            manifest = load_manifest(args.manifest, profile=args.profile)
            judge = (
                OllamaEvaluationJudge(args.judge_model, base_url=args.ollama_base_url)
                if args.judge_model
                else None
            )
            executor = LiveEvaluationExecutor(
                dsn=args.dsn,
                generation_model=args.model,
                embedding_model=args.embedding_model,
                ollama_base_url=args.ollama_base_url,
            )
            application = EvaluationApplication(
                executor,
                artifact_dir=args.artifact_dir,
                judge=judge,
                generation_model=args.model,
                embedding_model=args.embedding_model,
            )
            result = application.run(manifest, reuse_index=args.reuse_index)
        else:
            application = EvaluationApplication(
                cast(EvaluationExecutor, _UnavailableExecutor()),
                artifact_dir=args.artifact_dir,
            )
            if args.command == "compare":
                baseline = args.baseline or args.artifact_dir / "baseline.json"
                result = application.compare(args.candidate, baseline)
            else:
                destination = application.promote(
                    args.run, destination=args.destination
                )
                result = {"baseline": str(destination)}
    except (RagLabError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"raglab-evaluate: error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


class _UnavailableExecutor:
    """Compare and promotion do not cross the execution boundary."""

    def prepare(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution is unavailable")

    def retrieve(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution is unavailable")

    def generate(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution is unavailable")

    def release_generator(self) -> None:
        raise AssertionError("execution is unavailable")


def entrypoint() -> int:
    load_project_env()
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
