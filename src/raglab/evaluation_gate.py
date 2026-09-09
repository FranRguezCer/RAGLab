"""Fail-closed promotion gate for persisted evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from raglab.evaluation import REPORT_SCHEMA_VERSION

BASELINE_SCHEMA_VERSION = 1
RETRIEVAL_METRICS = ("precision_at_k", "recall_at_k", "mrr_at_k")
GENERATION_METRICS = (
    "fact_coverage",
    "grounded_fact_coverage",
    "citation_precision",
    "abstention_accuracy",
)


def validate_promotion(
    current: Mapping[str, Any], baseline: Mapping[str, Any], *, tolerance: float = 0.05
) -> None:
    """Validate that a run is complete, internally consistent, and non-regressing."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    _require_keys(
        baseline,
        {
            "schema_version", "report_schema_version", "dataset_id", "dataset_sha256",
            "case_ids", "top_k", "retrieval_summary", "generation_summary",
        },
        "baseline",
    )
    if baseline["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported baseline schema version: {baseline['schema_version']}")
    if baseline["report_schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("baseline targets an unsupported report schema version")

    report = _mapping(current.get("report"), "run.report")
    metadata = _mapping(current.get("metadata"), "run.metadata")
    traces = _mapping_list(current.get("traces"), "run.traces")
    if not traces:
        raise ValueError("run.traces cannot be empty")

    expected_ids = _string_list(baseline["case_ids"], "baseline.case_ids")
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("baseline.case_ids must be non-empty and unique")
    dataset_sha256 = _sha256(baseline["dataset_sha256"], "baseline.dataset_sha256")
    if metadata.get("dataset_sha256") != dataset_sha256:
        raise ValueError("baseline and current run use different dataset content")
    if report.get("schema_version") != baseline["report_schema_version"]:
        raise ValueError("baseline and current run use different report schemas")
    if report.get("dataset_id") != baseline["dataset_id"]:
        raise ValueError("baseline and current run use different datasets")
    if _string_list(report.get("case_ids"), "run.report.case_ids") != expected_ids:
        raise ValueError("run report contains missing, duplicate, or reordered cases")
    top_k = _positive_integer(baseline["top_k"], "baseline.top_k")
    if report.get("top_k") != top_k:
        raise ValueError("baseline and current run use different top_k values")
    if len(traces) != len(expected_ids):
        raise ValueError("run traces do not contain exactly the expected cases")

    failures: list[str] = []
    trace_retrieval: list[Mapping[str, Any]] = []
    trace_generation: list[Mapping[str, Any]] = []
    trace_ids: list[str] = []
    answer_ids: set[str] = set()
    for index, trace in enumerate(traces):
        case = _mapping(trace.get("case"), f"run.traces[{index}].case")
        retrieval = _mapping(trace.get("retrieval"), f"run.traces[{index}].retrieval")
        generation = _mapping(trace.get("generation"), f"run.traces[{index}].generation")
        case_id = _string(case.get("id"), f"run.traces[{index}].case.id")
        if retrieval.get("case_id") != case_id or generation.get("case_id") != case_id:
            raise ValueError(f"trace {case_id!r} has mismatched case identities")
        if retrieval.get("top_k") != top_k:
            raise ValueError(f"trace {case_id!r} uses a different top_k value")
        expected_outcome = case.get("expected_outcome")
        if expected_outcome not in {"answer", "abstain"}:
            raise ValueError(f"trace {case_id!r} has an invalid expected outcome")
        if generation.get("expected_outcome") != expected_outcome:
            raise ValueError(f"trace {case_id!r} has mismatched expected outcomes")
        if expected_outcome == "answer":
            answer_ids.add(case_id)
        trace_ids.append(case_id)
        trace_retrieval.append(retrieval)
        trace_generation.append(generation)
        if _nonempty_list(generation.get("forbidden_phrase_hits"), "forbidden_phrase_hits"):
            failures.append(f"{case_id}: forbidden phrase")
        if _nonempty_list(
            generation.get("invalid_citation_source_ids"), "invalid_citation_source_ids"
        ):
            failures.append(f"{case_id}: invalid citation")
        if generation.get("abstention_correct") is not True:
            failures.append(f"{case_id}: wrong abstention outcome")

    if trace_ids != expected_ids:
        raise ValueError("run traces contain missing, duplicate, or reordered cases")
    report_retrieval = _mapping_list(report.get("retrieval"), "run.report.retrieval")
    report_generation = _mapping_list(report.get("generation"), "run.report.generation")
    if report_retrieval != trace_retrieval or report_generation != trace_generation:
        raise ValueError("run report case results do not match its traces")

    calculated = {
        "retrieval_summary": {
            name: _mean(
                _metric(item, name)
                for item in trace_retrieval
                if cast(str, item["case_id"]) in answer_ids
            )
            for name in RETRIEVAL_METRICS
        },
        "generation_summary": {
            "fact_coverage": _mean(_metric(item, "fact_coverage") for item in trace_generation),
            "grounded_fact_coverage": _mean(
                _metric(item, "grounded_fact_coverage") for item in trace_generation
            ),
            "citation_precision": _mean(
                _metric(item, "citation_precision") for item in trace_generation
            ),
            "abstention_accuracy": _mean(
                float(item.get("abstention_correct") is True) for item in trace_generation
            ),
        },
    }
    for group, metric_names in (
        ("retrieval_summary", RETRIEVAL_METRICS),
        ("generation_summary", GENERATION_METRICS),
    ):
        reported = _summary(report.get(group), metric_names, f"run.report.{group}")
        previous = _summary(baseline[group], metric_names, f"baseline.{group}")
        for name in metric_names:
            actual = calculated[group][name]
            if not math.isclose(reported[name], actual, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"run.report.{group}.{name} does not match trace results")
            if actual < previous[name] - tolerance:
                failures.append(f"{group}.{name}: regression exceeds {tolerance:.2f}")
    if failures:
        raise ValueError("promotion blocked: " + "; ".join(failures))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _mapping_list(value: object, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be an array of objects")
    return cast(list[Mapping[str, Any]], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be an array of non-empty strings")
    return cast(list[str], value)


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return digest


def _nonempty_list(value: object, label: str) -> bool:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return bool(value)


def _metric(item: Mapping[str, Any], name: str) -> float:
    value = item.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"trace metric {name} must be a finite number")
    return float(value)


def _mean(values: Iterable[float]) -> float:
    items = tuple(values)
    return sum(items) / len(items) if items else 0.0


def _summary(value: object, names: Sequence[str], label: str) -> dict[str, float]:
    summary = _mapping(value, label)
    if set(summary) != set(names):
        raise ValueError(f"{label} has invalid metric keys")
    return {name: _metric(summary, name) for name in names}


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has invalid fields")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raglab-evaluation-gate")
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        current = _mapping(json.loads(args.run.read_text(encoding="utf-8")), "run")
        baseline = _mapping(json.loads(args.baseline.read_text(encoding="utf-8")), "baseline")
        validate_promotion(current, baseline, tolerance=args.tolerance)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"raglab-evaluation-gate: error: {exc}\n")
    print("promotion gate: passed")
    return 0
