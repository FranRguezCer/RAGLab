"""Read-only quality and latency benchmark for the preserved core corpus."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.embeddings import OllamaEmbeddingProvider
from raglab.evaluation import load_manifest
from raglab.evaluation.models import EvaluationCase
from raglab.retrieval import RetrievalConfig, RetrievalPipeline, RetrievalRequest
from raglab.retrieval.models import RetrievalResponse
from raglab.retrieval.repository import PostgresRetrievalRepository
from raglab.retrieval.rewriting import OllamaQueryRewriter

DEFAULT_DSN = "postgresql://raglab:raglab@127.0.0.1:5432/raglab"
EVIDENCE_RANK_ONE_CASES = {
    "cache-key-components",
    "authorization-header-distractor",
    "canary-failure-follow-up",
}


@dataclass(frozen=True, slots=True)
class Observation:
    elapsed_ms: float
    response: RetrievalResponse


def _retrieve(
    pipeline: RetrievalPipeline,
    case: EvaluationCase,
    *,
    collection: str,
    exact: bool,
) -> Observation:
    started = time.perf_counter()
    response = pipeline.retrieve(
        RetrievalRequest(
            case.query,
            collection,
            history=case.history,
            config=RetrievalConfig(candidate_k=50, top_k=5, ef_search=100, exact=exact),
        )
    )
    return Observation((time.perf_counter() - started) * 1000, response)


def _fact_ranks(case: EvaluationCase, response: RetrievalResponse) -> tuple[int | None, ...]:
    contents = tuple(result.content.casefold() for result in response.results)
    return tuple(
        next(
            (
                rank
                for rank, content in enumerate(contents, start=1)
                if any(anchor.casefold() in content for anchor in fact.evidence_anchors)
            ),
            None,
        )
        for fact in case.required_facts
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.getenv("RAGLAB_TEST_DSN", DEFAULT_DSN))
    parser.add_argument("--collection", default="raglab-eval-core")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--baseline-p50-ms", type=float, default=7420.0)
    args = parser.parse_args(argv)
    if args.runs < 5:
        parser.error("--runs must be at least 5")

    manifest = load_manifest(profile="core")
    cases = tuple(case for case in manifest.cases if case.required_facts)
    pipeline = RetrievalPipeline(
        PostgresRetrievalRepository(args.dsn),
        OllamaEmbeddingProvider(num_gpu=0, num_ctx=4096, keep_alive="5m"),
        rewriter=OllamaQueryRewriter(),
    )

    cold = _retrieve(pipeline, cases[0], collection=args.collection, exact=False)
    warm: list[float] = []
    runs: list[dict[str, Observation]] = []
    for _ in range(args.runs):
        run = {
            case.id: _retrieve(pipeline, case, collection=args.collection, exact=False)
            for case in cases
        }
        runs.append(run)
        warm.extend(item.elapsed_ms for item in run.values())

    approximate = runs[0]
    exact = {
        case.id: _retrieve(pipeline, case, collection=args.collection, exact=True) for case in cases
    }
    orders = {
        case.id: tuple(item.id for item in approximate[case.id].response.results) for case in cases
    }
    deterministic = all(
        tuple(item.id for item in run[case.id].response.results) == orders[case.id]
        for run in runs[1:]
        for case in cases
    )
    exact_match = all(
        tuple(item.id for item in exact[case.id].response.results) == orders[case.id]
        for case in cases
    )
    fact_ranks = {
        case.id: _fact_ranks(case, approximate[case.id].response) for case in cases
    }
    recall = statistics.fmean(
        all(rank is not None and rank <= 5 for rank in ranks) for ranks in fact_ranks.values()
    )
    first_ranks = {
        case_id: min((rank for rank in ranks if rank is not None), default=None)
        for case_id, ranks in fact_ranks.items()
    }
    mrr = statistics.fmean(
        0.0 if rank is None else 1.0 / rank for rank in first_ranks.values()
    )
    p50 = statistics.median(warm)
    p95 = _percentile(warm, 0.95)
    gain = 1 - p50 / args.baseline_p50_ms
    named_ranks = {case_id: first_ranks[case_id] for case_id in EVIDENCE_RANK_ONE_CASES}
    report = {
        "collection": args.collection,
        "case_count": len(cases),
        "recall_at_5": recall,
        "mrr": mrr,
        "evidence_ranks": named_ranks,
        "exact_approx_top5_identical": exact_match,
        "deterministic_warm_runs": deterministic,
        "warm_runs": args.runs,
        "cold_ms": cold.elapsed_ms,
        "warm_p50_ms": p50,
        "warm_p95_ms": p95,
        "p50_gain": gain,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    accepted = (
        recall == 1
        and mrr == 1
        and all(rank == 1 for rank in named_ranks.values())
        and exact_match
        and deterministic
        and p50 <= 2500
        and p95 <= 4000
        and gain >= 0.60
        and cold.elapsed_ms <= 5000
    )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
