"""Administrative CLI for versioned corpus ingestion."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from raglab.cli import DEFAULT_DSN
from raglab.config import load_project_env
from raglab.corpus import CorpusIngestionApplication, load_corpus_manifest, publish_receipt
from raglab.errors import RagLabError
from raglab.pipeline import ingest
from raglab.storage import PostgresRepository


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raglab-ingest-corpus",
        description="Ingest a validated local corpus manifest and publish an atomic receipt.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dsn", default=os.environ.get("RAGLAB_DSN", DEFAULT_DSN))
    args = parser.parse_args(argv)
    try:
        manifest = load_corpus_manifest(args.manifest)
        PostgresRepository(args.dsn).migrate()
        num_gpu_value = os.environ.get("RAGLAB_EMBEDDING_NUM_GPU")
        run = CorpusIngestionApplication(ingest).run(
            manifest,
            dsn=args.dsn,
            ollama_base_url=os.environ.get(
                "RAGLAB_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ),
            embedding_num_gpu=int(num_gpu_value) if num_gpu_value is not None else None,
            keep_alive=os.environ.get("RAGLAB_KEEP_ALIVE"),
        )
        publish_receipt(run, args.receipt)
    except (OSError, RagLabError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"raglab-ingest-corpus: error: {exc}\n")
    print(run.to_json(), end="")
    return 0


def entrypoint() -> int:
    load_project_env()
    return main()


if __name__ == "__main__":
    raise SystemExit(entrypoint())
