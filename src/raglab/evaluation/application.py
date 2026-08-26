"""Application service shared by the evaluation CLI, notebook, and tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, cast

from raglab.errors import EvaluationError, GenerationContractError
from raglab.evaluation.manifest import corpus_fingerprint, definition_fingerprint
from raglab.evaluation.metrics import (
    conservative_verdict,
    evidence_retrieval_metrics,
    latency_summary,
    normalize,
    percentile,
    retrieval_metrics,
)
from raglab.evaluation.models import (
    PROTECTED_COLLECTION_PREFIX,
    RUN_SCHEMA_VERSION,
    EvaluationCase,
    EvaluationExecutor,
    EvaluationJudge,
    EvaluationManifest,
    GenerationObservation,
    IngestionCheckObservation,
    IngestionObservation,
    RetrievalObservation,
    SemanticConfig,
)
from raglab.evaluation.semantic import (
    NLIScores,
    SemanticScorer,
    TransformersNLIScorer,
    render_pair,
    semantic_fingerprint,
)

MetadataProvider = Callable[[], dict[str, Any]]


class EvaluationApplication:
    """Own evaluation execution, comparison, persistence, and baseline promotion."""

    def __init__(
        self,
        executor: EvaluationExecutor,
        *,
        artifact_dir: str | Path = "artifacts/evaluation",
        judge: EvaluationJudge | None = None,
        generation_model: str = "qwen3:4b",
        embedding_model: str | None = None,
        metadata_provider: MetadataProvider | None = None,
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        if judge is not None and judge.model == generation_model:
            raise EvaluationError("The evaluation judge must differ from the generation model")
        self.executor = executor
        self.artifact_dir = Path(artifact_dir)
        self.judge = judge
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.metadata_provider = metadata_provider or environment_metadata
        self.semantic_scorer = semantic_scorer

    def run(
        self,
        manifest: EvaluationManifest,
        *,
        reuse_index: bool = False,
        persist: bool = True,
    ) -> dict[str, Any]:
        self._active_manifest = manifest
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        collection = f"{PROTECTED_COLLECTION_PREFIX}{manifest.profile}"
        corpus_hash, source_hashes = corpus_fingerprint(manifest)
        config_hash = _hash_json(manifest.config)
        benchmark_hash = definition_fingerprint(manifest)
        metadata = self.metadata_provider()
        metadata["generation_model"] = self.generation_model
        if self.embedding_model is not None:
            metadata["embedding_model"] = self.embedding_model
        errors: dict[str, list[str]] = {"hard": [], "advisory": []}
        try:
            ingestion = self.executor.prepare(
                manifest, collection=collection, reuse_index=reuse_index
            )
            if ingestion.separation_passed != ingestion.separation_total:
                errors["hard"].append("One or more must-separate chunk checks failed")
            if ingestion.cohesion_passed != ingestion.cohesion_total:
                errors["hard"].append("One or more must-keep chunk checks failed")
            cases = [self._run_case(case, collection, errors) for case in manifest.cases]
        except Exception as exc:
            failed = self._base_run(
                run_id,
                manifest,
                reuse_index,
                metadata,
                corpus_hash,
                source_hashes,
                config_hash,
                benchmark_hash,
            )
            failed.update(
                status="failed",
                ingestion={},
                cases=[],
                summary={},
                errors={"hard": [str(exc)], "advisory": []},
            )
            if persist:
                self._write_artifacts(failed)
            raise

        summary = self._summary(ingestion, cases)
        run = self._base_run(
            run_id,
            manifest,
            reuse_index,
            metadata,
            corpus_hash,
            source_hashes,
            config_hash,
            benchmark_hash,
        )
        run.update(
            status="complete",
            ingestion=asdict(ingestion),
            cases=cases,
            summary=summary,
            errors=errors,
            judge={"model": self.judge.model, "authority": "none", "status": "pending"}
            if self.judge is not None
            else None,
        )
        if persist:
            self._write_artifacts(run)
        try:
            self.executor.release_generator()
        except Exception as exc:
            errors["advisory"].append(f"generator unload: {exc}")
        if self.judge is not None:
            self._run_judge(run, manifest, errors)
        if persist:
            self._write_artifacts(run)
        return run

    def compare(
        self, candidate: str | Path | Mapping[str, Any], baseline: str | Path | Mapping[str, Any]
    ) -> dict[str, Any]:
        candidate_run = _load_run(candidate)
        baseline_run = _load_run(baseline)
        self._validate_comparison(candidate_run, baseline_run)
        candidate_axes = _quality_axes(candidate_run)
        baseline_axes = _quality_axes(baseline_run)
        hardware_compatible = candidate_run["metadata"].get("hardware_fingerprint") == baseline_run[
            "metadata"
        ].get("hardware_fingerprint")
        result: dict[str, Any] = {
            "baseline_run_id": baseline_run["run_id"],
            "candidate_run_id": candidate_run["run_id"],
            "quality_compatible": True,
            "latency_compatible": hardware_compatible,
            "axes": {
                key: {
                    "baseline": baseline_axes[key],
                    "candidate": candidate_axes[key],
                    "delta": candidate_axes[key] - baseline_axes[key],
                }
                for key in baseline_axes
            },
            "verdict": conservative_verdict(baseline_axes, candidate_axes),
        }
        if hardware_compatible:
            result["latency"] = {
                "baseline": baseline_run["summary"].get("latency", {}),
                "candidate": candidate_run["summary"].get("latency", {}),
            }
        return result

    def promote(
        self,
        run: str | Path | Mapping[str, Any],
        *,
        destination: str | Path | None = None,
    ) -> Path:
        payload = _load_run(run)
        target = Path(destination) if destination else self.artifact_dir / "baseline.json"
        reasons: list[str] = []
        if payload.get("status") != "complete":
            reasons.append("run is not complete")
        if payload.get("schema_version") != RUN_SCHEMA_VERSION:
            reasons.append(f"run must use schema v{RUN_SCHEMA_VERSION}")
        if not payload.get("definition", {}).get("fingerprint"):
            reasons.append("run has no benchmark definition fingerprint")
        if payload.get("partial") is True:
            reasons.append("runs created with --reuse-index are not promotable")
        if payload.get("metadata", {}).get("dirty") is not False:
            reasons.append("Git worktree was dirty or could not be verified")
        if payload.get("errors", {}).get("hard"):
            reasons.append("run contains hard failures")
        if target.exists():
            baseline = _load_run(target)
            if baseline.get("schema_version") != RUN_SCHEMA_VERSION:
                reasons.append(f"existing baseline must use schema v{RUN_SCHEMA_VERSION}")
            elif baseline.get("definition", {}).get("fingerprint") != payload.get(
                "definition", {}
            ).get("fingerprint"):
                reasons.append("benchmark definition fingerprint differs from existing baseline")
        if reasons:
            raise EvaluationError("Cannot promote baseline: " + "; ".join(reasons))
        _atomic_write(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def _run_case(
        self, case: EvaluationCase, collection: str, errors: dict[str, list[str]]
    ) -> dict[str, Any]:
        approximate = self.executor.retrieve(case, collection=collection, exact=False)
        exact = self.executor.retrieve(case, collection=collection, exact=True)
        source_metrics = retrieval_metrics(approximate.source_ids, case.expected_source_ids)
        metrics = evidence_retrieval_metrics(
            approximate.fact_ids_by_rank, tuple(fact.id for fact in case.required_facts)
        )
        exact_agreement = len(
            set(approximate.stable_references[:5]) & set(exact.stable_references[:5])
        ) / max(1, len(set(exact.stable_references[:5])))
        aggregate_agreement = (
            len(set(approximate.source_ids[:5]) & set(approximate.aggregate_source_ids[:5]))
            / max(1, len(set(approximate.aggregate_source_ids[:5])))
            if approximate.aggregate_source_ids
            else None
        )
        repetitions: list[dict[str, Any]] = []
        checks: list[dict[str, bool | None]] = []
        fact_grading: list[list[dict[str, Any]]] = []
        for repetition_number in range(1, 4):
            started = time.perf_counter()
            try:
                observation = self.executor.generate(case, collection=collection)
            except GenerationContractError as exc:
                attempt: dict[str, Any] = {
                    "status": "failed",
                    "error": str(exc),
                    "raw_output": exc.raw_output,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
                for key in (
                    "answer",
                    "abstained",
                    "cited_source_ids",
                    "prompt_tokens",
                    "generated_tokens",
                    "model_calls",
                ):
                    value = getattr(exc, key)
                    if value is not None:
                        attempt[key] = list(value) if isinstance(value, tuple) else value
                repetitions.append(attempt)
                checks.append(self._contract_failure_checks(case, exc))
                fact_grading.append(self._contract_fact_grading(case, exc))
                errors["hard"].append(f"{case.id} repetition {repetition_number}: {exc}")
                continue
            repetition_checks, repetition_grading = self._generation_checks(
                case, observation, errors
            )
            checks.append(repetition_checks)
            fact_grading.append(repetition_grading)
            failed_checks = [name for name, passed in repetition_checks.items() if passed is False]
            if failed_checks:
                reason = "Generation checks failed: " + ", ".join(failed_checks)
                repetitions.append({"status": "failed", **asdict(observation), "error": reason})
                errors["hard"].append(f"{case.id} repetition {repetition_number}: {reason}")
            else:
                repetitions.append({"status": "passed", **asdict(observation)})
        valid_repetitions = sum(all(value is True for value in check.values()) for check in checks)
        if metrics["recall_at_5"] is not None and metrics["recall_at_5"] < 1.0:
            errors["hard"].append(f"{case.id}: expected evidence was not retrieved")
        found_facts = sorted(
            {fact_id for row in approximate.fact_ids_by_rank[:5] for fact_id in row}
        )
        required_fact_ids = [fact.id for fact in case.required_facts]
        return {
            "id": case.id,
            "query": case.query,
            "history": list(case.history),
            "retrieval": {
                "source_ids": list(approximate.source_ids),
                "ranges": [
                    {
                        "reference": reference,
                        "source_id": source_id,
                        "fact_ids": list(fact_ids),
                    }
                    for reference, source_id, fact_ids in zip(
                        approximate.stable_references,
                        approximate.source_ids,
                        approximate.fact_ids_by_rank,
                        strict=False,
                    )
                ],
                "exact_source_ids": list(exact.source_ids),
                "exact_references": list(exact.stable_references),
                "aggregate_source_ids": list(approximate.aggregate_source_ids),
                "specialized_vs_aggregate_agreement_at_5": aggregate_agreement,
                "metrics": metrics,
                "source_metrics": source_metrics,
                "facts_found": found_facts,
                "facts_missing": [
                    fact_id for fact_id in required_fact_ids if fact_id not in found_facts
                ],
                "exact_agreement_at_5": exact_agreement,
                "latency_ms": approximate.latency_ms,
                "exact_latency_ms": exact.latency_ms,
            },
            "generation": {
                "repetitions": repetitions,
                "checks": checks,
                "fact_grading": fact_grading,
                "stability": f"{valid_repetitions}/3",
                "valid_repetitions": valid_repetitions,
            },
        }

    def _generation_checks(
        self,
        case: EvaluationCase,
        observation: GenerationObservation,
        errors: dict[str, list[str]],
    ) -> tuple[dict[str, bool | None], list[dict[str, Any]]]:
        answer = normalize(observation.answer)
        lexical = [
            any(normalize(variant) in answer for variant in fact.answer_variants)
            for fact in case.required_facts
        ]
        semantic_scores: list[NLIScores | None] = [None] * len(case.required_facts)
        semantic = self._semantic_config
        eligible = [
            index
            for index, _passed in enumerate(lexical)
            if not case.should_abstain
            and not observation.abstained
            and case.required_facts[index].semantic_claim is not None
            and semantic is not None
            and semantic.enabled
            and semantic.calibration.threshold is not None
            and semantic.calibration.contradiction_threshold is not None
        ]
        if eligible:
            assert semantic is not None
            scorer = self.semantic_scorer
            try:
                if scorer is None:
                    scorer = TransformersNLIScorer(semantic)
                    self.semantic_scorer = scorer
                pairs = [
                    render_pair(
                        semantic,
                        answer=observation.answer,
                        claim=case.required_facts[index].semantic_claim or "",
                    )
                    for index in eligible
                ]
                scores = scorer.score(pairs)
                if len(scores) != len(eligible):
                    raise EvaluationError("Semantic scorer returned an invalid score count")
                for index, score in zip(eligible, scores, strict=True):
                    if (
                        score is not None
                        and math.isfinite(score.entailment)
                        and math.isfinite(score.contradiction)
                        and 0.0 <= score.entailment <= 1.0
                        and 0.0 <= score.contradiction <= 1.0
                    ):
                        semantic_scores[index] = score
            except Exception as exc:
                errors["advisory"].append(f"semantic {case.id}: {exc}")
        grading: list[dict[str, Any]] = []
        fact_checks: dict[str, bool] = {}
        fingerprint = semantic_fingerprint(semantic) if semantic is not None else None
        for fact, lexical_passed, grade_scores in zip(
            case.required_facts, lexical, semantic_scores, strict=True
        ):
            semantic_passed = (
                grade_scores.entailment >= semantic.calibration.threshold
                if grade_scores is not None
                and semantic is not None
                and semantic.calibration.threshold is not None
                else None
            )
            contradiction_veto = (
                grade_scores.contradiction >= semantic.calibration.contradiction_threshold
                if grade_scores is not None
                and semantic is not None
                and semantic.calibration.contradiction_threshold is not None
                else None
            )
            semantic_active = semantic is not None and semantic.enabled
            if semantic_active and grade_scores is None:
                final = False
            elif lexical_passed:
                final = contradiction_veto is not True
            else:
                final = semantic_passed is True
            fact_checks[fact.id] = final
            grading.append(
                {
                    "fact_id": fact.id,
                    "lexical": lexical_passed,
                    "semantic": semantic_passed,
                    "final": final,
                    "semantic_score": grade_scores.entailment if grade_scores is not None else None,
                    "contradiction_score": (
                        grade_scores.contradiction if grade_scores is not None else None
                    ),
                    "contradiction_veto": contradiction_veto,
                    "model": semantic.model.name if semantic is not None else None,
                    "revision": semantic.model.revision if semantic is not None else None,
                    "fingerprint": fingerprint,
                }
            )
        cited = set(observation.cited_source_ids)
        citation_ok = not cited if case.should_abstain else set(case.expected_source_ids) <= cited
        return (
            {
                "contract": True,
                **fact_checks,
                "abstention": observation.abstained is case.should_abstain,
                "citations": citation_ok,
            },
            grading,
        )

    @property
    def _semantic_config(self) -> SemanticConfig | None:
        return self._active_manifest.semantic if hasattr(self, "_active_manifest") else None

    def _contract_fact_grading(
        self, case: EvaluationCase, error: GenerationContractError
    ) -> list[dict[str, Any]]:
        semantic = self._semantic_config
        fingerprint = semantic_fingerprint(semantic) if semantic is not None else None
        answer = normalize(error.answer) if error.answer is not None else None
        return [
            {
                "fact_id": fact.id,
                "lexical": (
                    None
                    if answer is None
                    else any(normalize(variant) in answer for variant in fact.answer_variants)
                ),
                "semantic": None,
                "final": (
                    None
                    if answer is None
                    else any(normalize(variant) in answer for variant in fact.answer_variants)
                ),
                "semantic_score": None,
                "contradiction_score": None,
                "contradiction_veto": None,
                "model": semantic.model.name if semantic is not None else None,
                "revision": semantic.model.revision if semantic is not None else None,
                "fingerprint": fingerprint,
            }
            for fact in case.required_facts
        ]

    @staticmethod
    def _contract_failure_checks(
        case: EvaluationCase, error: GenerationContractError
    ) -> dict[str, bool | None]:
        answer = normalize(error.answer) if error.answer is not None else None
        cited = set(error.cited_source_ids) if error.cited_source_ids is not None else None
        checks: dict[str, bool | None] = {"contract": False}
        checks.update(
            {
                fact.id: (
                    None
                    if answer is None
                    else any(normalize(variant) in answer for variant in fact.answer_variants)
                )
                for fact in case.required_facts
            }
        )
        checks["abstention"] = (
            None if error.abstained is None else error.abstained is case.should_abstain
        )
        checks["citations"] = (
            None
            if cited is None
            else (not cited if case.should_abstain else set(case.expected_source_ids) <= cited)
        )
        return checks

    @staticmethod
    def _summary(ingestion: IngestionObservation, cases: list[dict[str, Any]]) -> dict[str, Any]:
        retrieval = [
            cast(dict[str, float], case["retrieval"]["metrics"])
            for case in cases
            if case["retrieval"]["metrics"]["recall_at_5"] is not None
        ]
        generation_valid = [int(case["generation"]["valid_repetitions"]) / 3 for case in cases]
        retrieval_latencies = [float(case["retrieval"]["latency_ms"]) for case in cases]
        exact_latencies = [float(case["retrieval"]["exact_latency_ms"]) for case in cases]
        generation_latencies = [
            float(repetition["latency_ms"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
        ]
        prompt_tokens = [
            int(repetition["prompt_tokens"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
            if repetition.get("prompt_tokens") is not None
        ]
        generated_tokens = [
            int(repetition["generated_tokens"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
            if repetition.get("generated_tokens") is not None
        ]
        model_calls = sum(
            int(repetition["model_calls"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
            if repetition.get("model_calls") is not None
        )
        checks_total = ingestion.separation_total + ingestion.cohesion_total
        checks_passed = ingestion.separation_passed + ingestion.cohesion_passed
        all_checks = [check for case in cases for check in case["generation"]["checks"]]
        all_fact_grading = [
            grade
            for case in cases
            for repetition in case["generation"].get("fact_grading", [])
            for grade in repetition
        ]
        fact_ids = {
            fact_id
            for case in cases
            for fact_id in case["retrieval"]["facts_found"] + case["retrieval"]["facts_missing"]
        }

        def check_rate(names: set[str]) -> float:
            values = [
                value
                for check in all_checks
                for name, value in check.items()
                if name in names and value is not None
            ]
            return sum(value is True for value in values) / len(values) if values else 1.0

        return {
            "quality": {
                "ingestion_checks": checks_passed / checks_total if checks_total else 1.0,
                "ingestion_separation_rate": (
                    ingestion.separation_passed / ingestion.separation_total
                    if ingestion.separation_total
                    else 1.0
                ),
                "ingestion_cohesion_rate": (
                    ingestion.cohesion_passed / ingestion.cohesion_total
                    if ingestion.cohesion_total
                    else 1.0
                ),
                "retrieval_recall_at_5": (
                    mean(item["recall_at_5"] for item in retrieval) if retrieval else 1.0
                ),
                "retrieval_mrr": mean(item["mrr"] for item in retrieval) if retrieval else 1.0,
                "generation_pass_rate": mean(generation_valid),
                "generation_contract_rate": check_rate({"contract"}),
                "generation_facts_rate": check_rate(fact_ids),
                "generation_facts_lexical_rate": _grade_rate(all_fact_grading, "lexical"),
                "generation_facts_final_rate": _grade_rate(all_fact_grading, "final"),
                "generation_semantic_rescue_rate": _semantic_rate(all_fact_grading, rescued=True),
                "generation_semantic_unresolved_rate": _semantic_rate(
                    all_fact_grading, rescued=False
                ),
                "generation_semantic_contradiction_veto_rate": _grade_rate(
                    all_fact_grading, "contradiction_veto"
                ),
                "generation_abstention_rate": check_rate({"abstention"}),
                "generation_citations_rate": check_rate({"citations"}),
            },
            "latency": {
                "ingestion_ms": ingestion.latency_ms,
                "retrieval": latency_summary(retrieval_latencies),
                "exact_retrieval": latency_summary(exact_latencies),
                "generation": latency_summary(generation_latencies),
            },
            "tokens": {
                "ingestion": {
                    "p50": percentile([float(value) for value in ingestion.token_counts], 50),
                    "p95": percentile([float(value) for value in ingestion.token_counts], 95),
                },
                "prompt_total": sum(prompt_tokens),
                "generated_total": sum(generated_tokens),
            },
            "model_calls": model_calls,
        }

    @staticmethod
    def _base_run(
        run_id: str,
        manifest: EvaluationManifest,
        partial: bool,
        metadata: dict[str, Any],
        corpus_hash: str,
        source_hashes: dict[str, str],
        config_hash: str,
        definition_hash: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "profile": manifest.profile,
            "status": "running",
            "partial": partial,
            "metadata": metadata,
            "corpus": {"fingerprint": corpus_hash, "sources": source_hashes},
            "config": {"fingerprint": config_hash, "values": manifest.config},
            "definition": {
                "fingerprint": definition_hash,
                "manifest": {
                    key: value
                    for key, value in asdict(manifest).items()
                    if key not in {"base_path", "semantic"}
                },
            },
        }

    def _run_judge(
        self,
        run: dict[str, Any],
        manifest: EvaluationManifest,
        errors: dict[str, list[str]],
    ) -> None:
        assert self.judge is not None
        by_id = {case.id: case for case in manifest.cases}
        judgments: list[dict[str, Any]] = []
        judge_failed = False
        for result in cast(list[dict[str, Any]], run["cases"]):
            case = by_id[str(result["id"])]
            for repetition in result["generation"]["repetitions"]:
                if repetition["status"] == "failed":
                    continue
                try:
                    judgments.append(self.judge.evaluate(case, str(repetition["answer"])))
                except Exception as exc:
                    judge_failed = True
                    errors["advisory"].append(f"judge {case.id}: {exc}")
        run["judge"] = {
            "model": self.judge.model,
            "authority": "none",
            "status": "partial" if judge_failed else "complete",
            "judgments": judgments,
        }

    @staticmethod
    def _validate_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> None:
        if (
            candidate.get("schema_version") != RUN_SCHEMA_VERSION
            or baseline.get("schema_version") != RUN_SCHEMA_VERSION
        ):
            raise EvaluationError(f"Only schema v{RUN_SCHEMA_VERSION} runs can be compared")
        if candidate.get("status") != "complete" or baseline.get("status") != "complete":
            raise EvaluationError("Only complete evaluation runs can be compared")
        for label, run in (("candidate", candidate), ("baseline", baseline)):
            if run.get("partial") is True:
                raise EvaluationError(f"The {label} run is partial")
            if run.get("metadata", {}).get("dirty") is not False:
                raise EvaluationError(f"The {label} run is dirty or unverifiable")
            if run.get("errors", {}).get("hard"):
                raise EvaluationError(f"The {label} run contains hard failures")
        if candidate.get("profile") != baseline.get("profile"):
            raise EvaluationError("Evaluation profiles are incompatible")
        if candidate.get("corpus", {}).get("fingerprint") != baseline.get("corpus", {}).get(
            "fingerprint"
        ):
            raise EvaluationError("Corpus fingerprints differ; quality cannot be compared")
        if candidate.get("definition", {}).get("fingerprint") != baseline.get("definition", {}).get(
            "fingerprint"
        ):
            raise EvaluationError("Benchmark definition fingerprints differ")

    def _write_artifacts(self, run: dict[str, Any]) -> None:
        stem = str(run["run_id"])
        _atomic_write(
            self.artifact_dir / f"{stem}.json",
            json.dumps(run, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(self.artifact_dir / f"{stem}.md", render_markdown(run))


class HermeticEvaluationExecutor:
    """Deterministic teaching executor for notebook and application contract tests."""

    def __init__(self, *, latency_scale: float = 1.0) -> None:
        self.latency_scale = latency_scale
        self.profile = "core"

    def prepare(
        self, manifest: EvaluationManifest, *, collection: str, reuse_index: bool
    ) -> IngestionObservation:
        del collection, reuse_index
        self.profile = manifest.profile
        return IngestionObservation(
            len(manifest.sources),
            max(len(manifest.sources), 1) * 3,
            tuple(120 + index * 10 for index in range(max(len(manifest.sources), 1) * 3)),
            len(manifest.must_separate),
            len(manifest.must_separate),
            len(manifest.must_keep),
            len(manifest.must_keep),
            10.0 * self.latency_scale,
            tuple(
                IngestionCheckObservation(
                    check.id,
                    check_type,
                    True,
                    check.reason,
                    (),
                )
                for check_type, configured in (
                    ("must_separate", manifest.must_separate),
                    ("must_keep", manifest.must_keep),
                )
                for check in configured
            ),
        )

    def retrieve(
        self, case: EvaluationCase, *, collection: str, exact: bool
    ) -> RetrievalObservation:
        del collection
        return RetrievalObservation(
            case.expected_source_ids,
            tuple(
                tuple(anchor for fact in case.required_facts for anchor in fact.evidence_anchors)
                if index == 0
                else ()
                for index, _ in enumerate(case.expected_source_ids)
            ),
            (2.0 if exact else 1.0) * self.latency_scale,
            case.expected_source_ids if self.profile == "live" else (),
            tuple(f"{source_id}:0-0" for source_id in case.expected_source_ids),
            tuple(
                tuple(fact.id for fact in case.required_facts) if index == 0 else ()
                for index, _ in enumerate(case.expected_source_ids)
            ),
        )

    def generate(self, case: EvaluationCase, *, collection: str) -> GenerationObservation:
        del collection
        answer = (
            "I cannot answer from the available evidence."
            if case.should_abstain
            else "; ".join(
                fact.semantic_claim or fact.answer_variants[0]
                for fact in case.required_facts
            )
        )
        return GenerationObservation(
            answer,
            case.should_abstain,
            () if case.should_abstain else case.expected_source_ids,
            100,
            20,
            1,
            3.0 * self.latency_scale,
        )

    def release_generator(self) -> None:
        return None


def environment_metadata() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty_result = _git("status", "--porcelain")
    hardware = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "gpu": _gpu(),
    }
    return {
        "commit": commit,
        "dirty": None if dirty_result is None else bool(dirty_result),
        "generation_model": os.environ.get("RAGLAB_GENERATION_MODEL", "qwen3:4b"),
        "embedding_model": os.environ.get("RAGLAB_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        "hardware": hardware,
        "hardware_fingerprint": _hash_json(hardware),
    }


def render_markdown(run: Mapping[str, Any]) -> str:
    quality = cast(Mapping[str, Any], run.get("summary", {})).get("quality", {})
    lines = [
        f"# RAG evaluation {run['run_id']}",
        "",
        f"- Profile: `{run['profile']}`",
        f"- Status: `{run['status']}`",
        f"- Partial: `{str(run['partial']).lower()}`",
        f"- Hard failures: `{len(run.get('errors', {}).get('hard', []))}`",
        "",
        "## Quality axes",
        "",
    ]
    lines.extend(f"- {key}: `{float(value):.4f}`" for key, value in quality.items())
    ingestion = cast(Mapping[str, Any], run.get("ingestion", {}))
    failed_ingestion = [
        check for check in ingestion.get("checks", []) if not check.get("passed", False)
    ]
    lines.extend(["", "## Ingestion checks", ""])
    if failed_ingestion:
        for check in failed_ingestion:
            references = ", ".join(check.get("chunk_references", [])) or "none"
            lines.append(
                f"- `{check['id']}` ({check['type']}): {check['reason']} — chunks `{references}`"
            )
    else:
        lines.append("- All named chunk checks passed.")
    lines.extend(["", "## Cases", ""])
    for case in cast(list[dict[str, Any]], run.get("cases", [])):
        metrics = case["retrieval"]["metrics"]
        recall = metrics["recall_at_5"]
        mrr = metrics["mrr"]
        metric_text = (
            "evidence metrics `n/a`"
            if recall is None
            else f"Recall@5 `{recall:.3f}`, MRR `{mrr:.3f}`"
        )
        lines.append(
            f"- `{case['id']}`: {metric_text}, stability `{case['generation']['stability']}`"
        )
        found = ", ".join(case["retrieval"].get("facts_found", [])) or "none"
        missing = ", ".join(case["retrieval"].get("facts_missing", [])) or "none"
        lines.append(f"  - Facts found: `{found}`; missing: `{missing}`")
        for number, repetition in enumerate(case["generation"]["repetitions"], 1):
            if repetition["status"] == "failed":
                lines.append(f"  - Repetition {number} failed: {repetition['error']}")
    return "\n".join(lines) + "\n"


def _load_run(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        payload = json.loads(Path(value).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not load evaluation run {value}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Evaluation run JSON must be an object")
    return cast(dict[str, Any], payload)


def _quality_axes(run: Mapping[str, Any]) -> dict[str, float]:
    quality = cast(Mapping[str, Any], cast(Mapping[str, Any], run["summary"])["quality"])
    return {str(key): float(value) for key, value in quality.items()}


def _grade_rate(grades: list[dict[str, Any]], field: str) -> float:
    values = [grade[field] for grade in grades if grade.get(field) is not None]
    return sum(value is True for value in values) / len(values) if values else 1.0


def _semantic_rate(grades: list[dict[str, Any]], *, rescued: bool) -> float:
    misses = [grade for grade in grades if grade.get("lexical") is False]
    if not misses:
        return 0.0
    if rescued:
        return sum(grade.get("semantic") is True for grade in misses) / len(misses)
    return sum(grade.get("semantic") is None for grade in misses) / len(misses)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, check=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _memory_bytes() -> int | None:
    try:
        first = next(
            line
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
        return int(first.split()[1]) * 1024
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _gpu() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
