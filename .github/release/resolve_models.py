"""Resolve release model identities from Ollama and pinned source revisions."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

OLLAMA_MODELS = {
    "embedding": "qwen3-embedding:0.6b",
    "generation": "qwen3:4b",
}
PINNED_MODELS = {
    "reranker": {
        "name": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    },
    "nli": {
        "name": "tasksource/deberta-small-long-nli",
        "revision": "9a77395d4d3751be9e2a69c4ae318491d9b3fffb",
    },
}


def main() -> None:
    destination = Path(sys.argv[1])
    base_url = os.environ.get("RAGLAB_OLLAMA_BASE_URL", "http://ollama:11434")
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=30) as response:  # noqa: S310
        payload: dict[str, Any] = json.load(response)
    available = {str(item["name"]): str(item["digest"]) for item in payload["models"]}
    resolved: dict[str, dict[str, str]] = {}
    for role, name in OLLAMA_MODELS.items():
        digest = available.get(name)
        if digest is None or re.fullmatch(r"sha256:[0-9a-f]{64}|[0-9a-f]{64}", digest) is None:
            raise SystemExit(f"Ollama did not return a full digest for {name}")
        resolved[role] = {"name": name, "digest": digest.removeprefix("sha256:")}
    resolved.update(PINNED_MODELS)
    destination.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
