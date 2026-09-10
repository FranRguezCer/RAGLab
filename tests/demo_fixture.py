from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from raglab.demo_evidence import build_evidence, sha256_file, write_evidence

COMMIT = "a" * 40
MODELS = {
    "embedding": {"name": "qwen3-embedding:0.6b", "digest": "e" * 64},
    "generation": {"name": "qwen3:4b", "digest": "f" * 64},
    "reranker": {"name": "bge", "revision": "revision-1"},
    "nli": {"name": "nli", "revision": "revision-2"},
}


def create_evidence(root: Path, *, created_at: str = "2026-09-10T12:00:00Z") -> Path:
    dataset_source = Path("data/evaluation/raspberry_pi_demo_v1.json")
    baseline_source = Path("data/evaluation/baselines/raspberry_pi_demo_v1.json")
    (root / "data").mkdir(parents=True)
    dataset = root / "data" / "dataset.json"
    baseline_path = root / "data" / "baseline.json"
    dataset.write_bytes(dataset_source.read_bytes())
    baseline_path.write_bytes(baseline_source.read_bytes())
    baseline = json.loads(baseline_path.read_text())
    cases = json.loads(dataset.read_text())["cases"]
    retrieval_summary = baseline["retrieval_summary"]
    generation_summary = baseline["generation_summary"]
    retrieval: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for case in cases:
        retrieval_result = {
            "case_id": case["id"], "top_k": 3,
            "precision_at_k": retrieval_summary["precision_at_k"],
            "recall_at_k": retrieval_summary["recall_at_k"],
            "mrr_at_k": retrieval_summary["mrr_at_k"],
        }
        generation_result = {
            "case_id": case["id"], "expected_outcome": case["expected_outcome"],
            "fact_coverage": generation_summary["fact_coverage"],
            "grounded_fact_coverage": generation_summary["grounded_fact_coverage"],
            "citation_precision": generation_summary["citation_precision"],
            "forbidden_phrase_hits": [], "invalid_citation_source_ids": [],
            "abstention_correct": True,
        }
        response = {
            "answer": (
                "Grounded answer."
                if case["expected_outcome"] == "answer"
                else "I cannot answer from this corpus."
            ),
            "abstained": case["expected_outcome"] == "abstain", "sources": [],
            "retrieval": {"query": case["question"], "rewritten_query": None,
                          "query_variants": [case["question"]], "results": []},
            "strategy": "single_pass",
            "metrics": {"model_calls": 1, "prompt_tokens": 10, "generated_tokens": 8},
        }
        retrieval.append(retrieval_result)
        generation.append(generation_result)
        traces.append({"case": case, "response": response,
                       "retrieval": retrieval_result, "generation": generation_result})
    run = {
        "traces": traces,
        "report": {"schema_version": 1, "dataset_id": baseline["dataset_id"],
                   "case_ids": baseline["case_ids"], "top_k": 3,
                   "retrieval": retrieval, "generation": generation,
                   "retrieval_summary": retrieval_summary,
                   "generation_summary": generation_summary},
        "metadata": {"dataset_sha256": baseline["dataset_sha256"],
                     "retrieval_config": {"top_k": 3}, "duration_seconds": 12.5},
    }
    evidence = build_evidence(
        run, baseline, {"source_count": 9}, MODELS, commit=COMMIT,
        inputs={"data/dataset.json": sha256_file(dataset),
                "data/baseline.json": sha256_file(baseline_path)}, created_at=created_at,
    )
    output = root / "evidence.json"
    write_evidence(output, evidence)
    return output
