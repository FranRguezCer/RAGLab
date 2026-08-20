"""Deterministic metrics and conservative comparison rules."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from statistics import median


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def retrieval_metrics(ranked: Sequence[str], relevant: Sequence[str]) -> dict[str, float]:
    expected = set(relevant)
    if not expected:
        return {"hit_at_1": 1.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "recall_at_5": 1.0,
                "mrr": 1.0, "ndcg_at_5": 1.0}
    ranks = [index + 1 for index, source in enumerate(ranked) if source in expected]
    gains = [1.0 if source in expected else 0.0 for source in ranked[:5]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(5, len(expected))))
    return {
        "hit_at_1": float(any(rank <= 1 for rank in ranks)),
        "hit_at_3": float(any(rank <= 3 for rank in ranks)),
        "hit_at_5": float(any(rank <= 5 for rank in ranks)),
        "recall_at_5": len({source for source in ranked[:5] if source in expected}) / len(expected),
        "mrr": 1.0 / min(ranks) if ranks else 0.0,
        "ndcg_at_5": dcg / ideal if ideal else 0.0,
    }


def percentile(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * value / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {"p50_ms": median(values) if values else 0.0, "p95_ms": percentile(values, 95)}


def conservative_verdict(
    baseline_axes: dict[str, float], candidate_axes: dict[str, float]
) -> str:
    directions = {
        1 if candidate_axes[key] > baseline_axes[key] else -1
        if candidate_axes[key] < baseline_axes[key] else 0
        for key in baseline_axes.keys() & candidate_axes.keys()
    }
    directions.discard(0)
    if directions == {1}:
        return "improved"
    if directions == {-1}:
        return "regressed"
    if directions == {1, -1}:
        return "mixed"
    return "no_clear_change"
