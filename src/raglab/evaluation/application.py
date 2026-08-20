"""Application service shared by the evaluation CLI, notebook, and tests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, cast

from raglab.errors import EvaluationError
from raglab.evaluation.manifest import corpus_fingerprint
from raglab.evaluation.metrics import (
    conservative_verdict,
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
    IngestionObservation,
    RetrievalObservation,
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
    ) -> None:
        if judge is not None and judge.model == generation_model:
            raise EvaluationError("The evaluation judge must differ from the generation model")
        self.executor = executor
        self.artifact_dir = Path(artifact_dir)
        self.judge = judge
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.metadata_provider = metadata_provider or environment_metadata

    def run(
        self,
        manifest: EvaluationManifest,
        *,
        reuse_index: bool = False,
        persist: bool = True,
    ) -> dict[str, Any]:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        collection = f"{PROTECTED_COLLECTION_PREFIX}{manifest.profile}"
        corpus_hash, source_hashes = corpus_fingerprint(manifest)
        config_hash = _hash_json(manifest.config)
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
                run_id, manifest, reuse_index, metadata, corpus_hash, source_hashes, config_hash
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
            run_id, manifest, reuse_index, metadata, corpus_hash, source_hashes, config_hash
        )
        run.update(
            status="complete",
            ingestion=asdict(ingestion),
            cases=cases,
            summary=summary,
            errors=errors,
            judge={"model": self.judge.model, "status": "pending"}
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
        hardware_compatible = (
            candidate_run["metadata"].get("hardware_fingerprint")
            == baseline_run["metadata"].get("hardware_fingerprint")
        )
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
        reasons: list[str] = []
        if payload.get("status") != "complete":
            reasons.append("run is not complete")
        if payload.get("partial") is True:
            reasons.append("runs created with --reuse-index are not promotable")
        if payload.get("metadata", {}).get("dirty") is not False:
            reasons.append("Git worktree was dirty or could not be verified")
        if payload.get("errors", {}).get("hard"):
            reasons.append("run contains hard failures")
        if reasons:
            raise EvaluationError("Cannot promote baseline: " + "; ".join(reasons))
        target = Path(destination) if destination else self.artifact_dir / "baseline.json"
        _atomic_write(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def _run_case(
        self, case: EvaluationCase, collection: str, errors: dict[str, list[str]]
    ) -> dict[str, Any]:
        approximate = self.executor.retrieve(case, collection=collection, exact=False)
        exact = self.executor.retrieve(case, collection=collection, exact=True)
        metrics = retrieval_metrics(approximate.source_ids, case.expected_source_ids)
        exact_agreement = len(
            set(approximate.source_ids[:5]) & set(exact.source_ids[:5])
        ) / max(1, len(set(exact.source_ids[:5])))
        aggregate_agreement = (
            len(
                set(approximate.source_ids[:5])
                & set(approximate.aggregate_source_ids[:5])
            )
            / max(1, len(set(approximate.aggregate_source_ids[:5])))
            if approximate.aggregate_source_ids
            else None
        )
        repetitions = [self.executor.generate(case, collection=collection) for _ in range(3)]
        checks = [self._generation_checks(case, repetition) for repetition in repetitions]
        valid_repetitions = sum(all(check.values()) for check in checks)
        if metrics["recall_at_5"] < 1.0 and not case.should_abstain:
            errors["hard"].append(f"{case.id}: expected evidence was not retrieved")
        if valid_repetitions != 3:
            errors["hard"].append(
                f"{case.id}: deterministic generation checks passed {valid_repetitions}/3"
            )
        return {
            "id": case.id,
            "query": case.query,
            "history": list(case.history),
            "retrieval": {
                "source_ids": list(approximate.source_ids),
                "exact_source_ids": list(exact.source_ids),
                "aggregate_source_ids": list(approximate.aggregate_source_ids),
                "specialized_vs_aggregate_agreement_at_5": aggregate_agreement,
                "metrics": metrics,
                "exact_agreement_at_5": exact_agreement,
                "latency_ms": approximate.latency_ms,
                "exact_latency_ms": exact.latency_ms,
            },
            "generation": {
                "repetitions": [asdict(item) for item in repetitions],
                "checks": checks,
                "stability": f"{valid_repetitions}/3",
                "valid_repetitions": valid_repetitions,
            },
        }

    @staticmethod
    def _generation_checks(
        case: EvaluationCase, observation: GenerationObservation
    ) -> dict[str, bool]:
        answer = normalize(observation.answer)
        facts = all(normalize(fact) in answer for fact in case.required_facts)
        cited = set(observation.cited_source_ids)
        citation_ok = (
            not cited if case.should_abstain else set(case.expected_source_ids) <= cited
        )
        return {
            "required_facts": facts,
            "abstention": observation.abstained is case.should_abstain,
            "citations": citation_ok,
        }

    @staticmethod
    def _summary(
        ingestion: IngestionObservation, cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        retrieval = [cast(dict[str, float], case["retrieval"]["metrics"]) for case in cases]
        generation_valid = [
            int(case["generation"]["valid_repetitions"]) / 3 for case in cases
        ]
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
            if repetition["prompt_tokens"] is not None
        ]
        generated_tokens = [
            int(repetition["generated_tokens"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
            if repetition["generated_tokens"] is not None
        ]
        model_calls = sum(
            int(repetition["model_calls"])
            for case in cases
            for repetition in case["generation"]["repetitions"]
        )
        checks_total = ingestion.separation_total + ingestion.cohesion_total
        checks_passed = ingestion.separation_passed + ingestion.cohesion_passed
        return {
            "quality": {
                "ingestion_checks": checks_passed / checks_total if checks_total else 1.0,
                "retrieval_recall_at_5": mean(item["recall_at_5"] for item in retrieval),
                "retrieval_mrr": mean(item["mrr"] for item in retrieval),
                "generation_pass_rate": mean(generation_valid),
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
        for result in cast(list[dict[str, Any]], run["cases"]):
            case = by_id[str(result["id"])]
            for repetition in result["generation"]["repetitions"]:
                try:
                    judgments.append(self.judge.evaluate(case, str(repetition["answer"])))
                except Exception as exc:
                    errors["advisory"].append(f"judge {case.id}: {exc}")
        run["judge"] = {
            "model": self.judge.model,
            "status": "complete" if not errors["advisory"] else "partial",
            "judgments": judgments,
        }

    @staticmethod
    def _validate_comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> None:
        if candidate.get("status") != "complete" or baseline.get("status") != "complete":
            raise EvaluationError("Only complete evaluation runs can be compared")
        if candidate.get("schema_version") != baseline.get("schema_version"):
            raise EvaluationError("Run schema versions are incompatible")
        if candidate.get("profile") != baseline.get("profile"):
            raise EvaluationError("Evaluation profiles are incompatible")
        if candidate.get("corpus", {}).get("fingerprint") != baseline.get("corpus", {}).get(
            "fingerprint"
        ):
            raise EvaluationError("Corpus fingerprints differ; quality cannot be compared")
        if candidate.get("config", {}).get("fingerprint") != baseline.get("config", {}).get(
            "fingerprint"
        ):
            raise EvaluationError("Evaluation configurations differ; quality cannot be compared")

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
        )

    def retrieve(
        self, case: EvaluationCase, *, collection: str, exact: bool
    ) -> RetrievalObservation:
        del collection
        return RetrievalObservation(
            case.expected_source_ids,
            tuple(() for _ in case.expected_source_ids),
            (2.0 if exact else 1.0) * self.latency_scale,
            case.expected_source_ids if self.profile == "live" else (),
        )

    def generate(
        self, case: EvaluationCase, *, collection: str
    ) -> GenerationObservation:
        del collection
        answer = (
            "I cannot answer from the available evidence."
            if case.should_abstain
            else "; ".join(case.required_facts)
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
    lines.extend(["", "## Cases", ""])
    for case in cast(list[dict[str, Any]], run.get("cases", [])):
        metrics = case["retrieval"]["metrics"]
        lines.append(
            f"- `{case['id']}`: Recall@5 `{metrics['recall_at_5']:.3f}`, "
            f"MRR `{metrics['mrr']:.3f}`, stability `{case['generation']['stability']}`"
        )
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
