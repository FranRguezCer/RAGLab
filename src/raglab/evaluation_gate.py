"""Fail-closed promotion gate for persisted evaluation runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def validate_promotion(
    current: Mapping[str, Any], baseline: Mapping[str, Any], *, tolerance: float = 0.05
) -> None:
    report = cast(Mapping[str, Any], current["report"])
    if report["dataset_id"] != baseline["dataset_id"]:
        raise ValueError("baseline and current run use different datasets")
    failures: list[str] = []
    for trace in cast(list[dict[str, Any]], current["traces"]):
        generation = trace["generation"]
        if generation["forbidden_phrase_hits"]:
            failures.append(f"{generation['case_id']}: forbidden phrase")
        if generation["invalid_citation_source_ids"]:
            failures.append(f"{generation['case_id']}: invalid citation")
        if not generation["abstention_correct"]:
            failures.append(f"{generation['case_id']}: wrong abstention outcome")
    for group in ("retrieval_summary", "generation_summary"):
        current_metrics = cast(Mapping[str, float], report[group])
        baseline_metrics = cast(Mapping[str, float], baseline[group])
        for name, previous in baseline_metrics.items():
            if current_metrics[name] < previous - tolerance:
                failures.append(f"{group}.{name}: regression exceeds {tolerance:.2f}")
    if failures:
        raise ValueError("promotion blocked: " + "; ".join(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raglab-evaluation-gate")
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        current = json.loads(args.run.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        validate_promotion(current, baseline, tolerance=args.tolerance)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"raglab-evaluation-gate: error: {exc}\n")
    print("promotion gate: passed")
    return 0
