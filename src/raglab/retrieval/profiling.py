"""Opt-in, privacy-preserving retrieval stage profiling."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


_STAGES = ("rewrite", "embedding", "search", "fusion", "rerank", "expansion", "mmr")


@dataclass(slots=True)
class RetrievalProfile:
    """Accumulate one retrieval profile and append it as a single JSONL record."""

    query: str
    exact: bool
    started: float = field(default_factory=time.perf_counter)
    timings_ms: dict[str, float] = field(
        default_factory=lambda: {stage: 0.0 for stage in _STAGES}
    )
    candidate_count: int = 0
    document_count: int = 0
    rewrite_failures: int = 0
    postgres_calls: int = 0
    device: str = "none"

    def measure(self, stage: str, started: float) -> None:
        self.timings_ms[stage] += (time.perf_counter() - started) * 1000

    def emit(self) -> None:
        path = os.getenv("RAGLAB_RETRIEVAL_PROFILE")
        if not path:
            return
        timings = {key: round(value, 3) for key, value in self.timings_ms.items()}
        timings["total"] = round((time.perf_counter() - self.started) * 1000, 3)
        record = {
            "query_sha256": hashlib.sha256(self.query.encode()).hexdigest(),
            "exact": self.exact,
            "candidate_count": self.candidate_count,
            "document_count": self.document_count,
            "device": self.device,
            "rewrite_failures": self.rewrite_failures,
            "postgres_calls": self.postgres_calls,
            "timings_ms": timings,
        }
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
