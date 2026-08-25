"""Manifest loading, validation, and corpus fingerprinting."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from raglab.errors import EvaluationError
from raglab.evaluation.models import (
    ChunkCheck,
    EvaluationCase,
    EvaluationManifest,
    EvaluationSource,
    FactExpectation,
    SemanticCalibrationConfig,
    SemanticConfig,
    SemanticModelConfig,
    SemanticTemplateConfig,
)


def default_manifest_path(profile: str = "core") -> Path:
    if profile not in {"core", "live"}:
        raise EvaluationError(f"Unknown evaluation profile: {profile!r}")
    resource = files("raglab.evaluation.fixtures").joinpath(f"{profile}_manifest.json")
    return Path(str(resource))


def load_manifest(path: str | Path | None = None, *, profile: str = "core") -> EvaluationManifest:
    manifest_path = Path(path) if path is not None else default_manifest_path(profile)
    try:
        raw = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Could not load evaluation manifest {manifest_path}: {exc}") from exc
    try:
        version = int(raw["schema_version"])
        manifest_profile = str(raw["profile"])
        source_rows = cast(list[dict[str, Any]], raw["sources"])
        case_rows = cast(list[dict[str, Any]], raw["cases"])
        checks = cast(dict[str, list[dict[str, Any]]], raw.get("chunk_checks", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Evaluation manifest has an invalid top-level shape") from exc
    if version not in {1, 2, 3}:
        raise EvaluationError(f"Unsupported evaluation manifest schema: {version}")
    try:
        sources = tuple(
            EvaluationSource(
                str(row["id"]),
                str(row["path"]),
                _optional_str(row.get("sha256")),
                _optional_str(row.get("domain")),
            )
            for row in source_rows
        )
        cases = tuple(
            EvaluationCase(
                id=str(row["id"]),
                query=str(row["query"]),
                expected_source_ids=_strings(row.get("expected_source_ids", [])),
                required_facts=_facts(
                    str(row["id"]), row.get("required_facts", []), version=version
                ),
                should_abstain=bool(row.get("should_abstain", False)),
                history=_strings(row.get("history", [])),
                domain=_optional_str(row.get("domain")),
            )
            for row in case_rows
        )
        separate = tuple(_chunk_check(row) for row in checks.get("must_separate", []))
        keep = tuple(_chunk_check(row) for row in checks.get("must_keep", []))
        config = cast(dict[str, Any], raw.get("config", {}))
        semantic = _semantic_config(config.get("semantic"), manifest_path.parent, version)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Evaluation manifest contains an invalid source or case") from exc
    manifest = EvaluationManifest(
        version,
        manifest_profile,
        sources,
        cases,
        separate,
        keep,
        config,
        str(manifest_path.parent.resolve()),
        semantic,
    )
    _validate(manifest, requested_profile=profile)
    return manifest


def resolved_source(manifest: EvaluationManifest, source: EvaluationSource) -> str:
    location = os.path.expandvars(source.location)
    if "$" in location:
        raise EvaluationError(f"Evaluation source {source.id!r} uses an unset environment variable")
    if urlsplit(location).scheme in {"http", "https"}:
        return location
    if source.location.startswith("$"):
        return str(Path(location).expanduser().resolve())
    return str((Path(manifest.base_path) / location).resolve())


def corpus_fingerprint(manifest: EvaluationManifest) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for source in manifest.sources:
        location = resolved_source(manifest, source)
        if urlsplit(location).scheme in {"http", "https"}:
            try:
                request = urllib.request.Request(
                    location, headers={"User-Agent": "RAGLab evaluation/1"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                    digest = hashlib.sha256(response.read()).hexdigest()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise EvaluationError(f"Could not hash live source {source.id!r}: {exc}") from exc
            if source.sha256 is not None and digest != source.sha256:
                raise EvaluationError(f"Live source hash changed for {source.id!r}")
        else:
            try:
                digest = hashlib.sha256(Path(location).read_bytes()).hexdigest()
            except OSError as exc:
                raise EvaluationError(
                    f"Could not hash evaluation source {location}: {exc}"
                ) from exc
            if source.sha256 is not None and digest != source.sha256:
                raise EvaluationError(f"Evaluation source hash changed for {source.id!r}")
        hashes[source.id] = digest
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), hashes


def definition_fingerprint(manifest: EvaluationManifest) -> str:
    """Hash the complete semantic benchmark definition, excluding its disk location."""

    payload = {
        "schema_version": manifest.schema_version,
        "profile": manifest.profile,
        "sources": [
            {
                "id": source.id,
                "location": source.location,
                "sha256": source.sha256,
                "domain": source.domain,
            }
            for source in manifest.sources
        ],
        "cases": [
            {
                "id": case.id,
                "query": case.query,
                "expected_source_ids": list(case.expected_source_ids),
                "required_facts": [
                    {
                        "id": fact.id,
                        "evidence_anchors": list(fact.evidence_anchors),
                        "answer_variants": list(fact.answer_variants),
                        "semantic_claim": fact.semantic_claim,
                    }
                    for fact in case.required_facts
                ],
                "should_abstain": case.should_abstain,
                "history": list(case.history),
                "domain": case.domain,
            }
            for case in manifest.cases
        ],
        "chunk_checks": {
            "must_separate": [_chunk_check_payload(check) for check in manifest.must_separate],
            "must_keep": [_chunk_check_payload(check) for check in manifest.must_keep],
        },
        "config": manifest.config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate(manifest: EvaluationManifest, *, requested_profile: str) -> None:
    if manifest.profile != requested_profile:
        raise EvaluationError(
            f"Manifest profile {manifest.profile!r} does not match {requested_profile!r}"
        )
    source_ids = [source.id for source in manifest.sources]
    case_ids = [case.id for case in manifest.cases]
    if not source_ids or any(not value.strip() for value in source_ids) or len(source_ids) != len(
        set(source_ids)
    ):
        raise EvaluationError("Evaluation source ids must be non-empty and unique")
    if not case_ids or any(not value.strip() for value in case_ids) or len(case_ids) != len(
        set(case_ids)
    ):
        raise EvaluationError("Evaluation case ids must be non-empty and unique")
    known = set(source_ids)
    for case in manifest.cases:
        if not case.query.strip():
            raise EvaluationError(f"Evaluation case {case.id!r} has an empty query")
        if not set(case.expected_source_ids) <= known:
            raise EvaluationError(f"Evaluation case {case.id!r} references an unknown source")
        if manifest.profile == "live" and not case.domain:
            raise EvaluationError(f"Live evaluation case {case.id!r} needs a domain")
        if case.should_abstain and (case.expected_source_ids or case.required_facts):
            raise EvaluationError(f"Abstention case {case.id!r} cannot require evidence")
        for fact in case.required_facts:
            if not fact.id.strip():
                raise EvaluationError("Required fact ids must be non-empty and unique")
            if not fact.evidence_anchors or any(
                not anchor.strip() for anchor in fact.evidence_anchors
            ):
                raise EvaluationError(
                    f"Required fact {fact.id!r} needs non-empty evidence anchors"
                )
            if not fact.answer_variants or any(
                not variant.strip() for variant in fact.answer_variants
            ):
                raise EvaluationError(
                    f"Required fact {fact.id!r} needs non-empty answer variants"
                )
            if manifest.schema_version >= 3 and (
                fact.semantic_claim is None or not fact.semantic_claim.strip()
            ):
                raise EvaluationError(
                    f"Required fact {fact.id!r} needs a non-empty semantic_claim"
                )
    fact_ids = [fact.id for case in manifest.cases for fact in case.required_facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise EvaluationError("Required fact ids must be non-empty and unique")
    for check in (*manifest.must_separate, *manifest.must_keep):
        if check.source_id not in known:
            raise EvaluationError(f"Chunk check {check.id!r} references an unknown source")
    check_ids = [check.id for check in (*manifest.must_separate, *manifest.must_keep)]
    if any(not value.strip() for value in check_ids) or len(check_ids) != len(set(check_ids)):
        raise EvaluationError("Chunk check ids must be non-empty and unique")
    if manifest.profile == "live" and any(not source.domain for source in manifest.sources):
        raise EvaluationError("Every live evaluation source needs a domain")


def _chunk_check(row: dict[str, Any]) -> ChunkCheck:
    try:
        return ChunkCheck(
            str(row["id"]),
            str(row["source_id"]),
            str(row["left_anchor"]),
            str(row["right_anchor"]),
            str(row["reason"]),
        )
    except KeyError as exc:
        raise EvaluationError("Chunk check is missing a required field") from exc


def _chunk_check_payload(check: ChunkCheck) -> dict[str, str]:
    return {
        "id": check.id,
        "source_id": check.source_id,
        "left_anchor": check.left_anchor,
        "right_anchor": check.right_anchor,
        "reason": check.reason,
    }


def _facts(case_id: str, value: object, *, version: int) -> tuple[FactExpectation, ...]:
    if not isinstance(value, list):
        raise EvaluationError("Expected required_facts to be a list")
    if version == 1:
        if not all(isinstance(item, str) for item in value):
            raise EvaluationError("Manifest v1 required_facts must be strings")
        return tuple(
            FactExpectation(
                id=f"{case_id}-fact-{index}",
                evidence_anchors=(item,),
                answer_variants=(item,),
            )
            for index, item in enumerate(value, start=1)
        )
    facts: list[FactExpectation] = []
    for item in value:
        if not isinstance(item, dict):
            raise EvaluationError("Manifest v2+ required_facts must be objects")
        try:
            facts.append(
                FactExpectation(
                    id=str(item["id"]),
                    evidence_anchors=_strings(item["evidence_anchors"]),
                    answer_variants=_strings(item["answer_variants"]),
                    semantic_claim=(
                        str(item["semantic_claim"]) if version >= 3 else None
                    ),
                )
            )
        except KeyError as exc:
            raise EvaluationError("Required fact is missing a required field") from exc
    return tuple(facts)


def _semantic_config(value: object, base_path: Path, version: int) -> SemanticConfig | None:
    if version < 3:
        return None
    if not isinstance(value, dict):
        raise EvaluationError("Manifest v3 config.semantic must be an object")
    try:
        model = cast(dict[str, Any], value["model"])
        calibration = cast(dict[str, Any], value["calibration"])
        template = cast(dict[str, Any], value["template"])
        fixture = Path(str(calibration["fixture"]))
        if not fixture.is_absolute():
            fixture = (base_path / fixture).resolve()
        result = SemanticConfig(
            enabled=bool(value["enabled"]),
            model=SemanticModelConfig(
                name=str(model["name"]),
                revision=str(model["revision"]),
                device=str(model["device"]),
                batch_size=int(model["batch_size"]),
                max_tokens=int(model["max_tokens"]),
            ),
            calibration=SemanticCalibrationConfig(
                fixture=str(fixture),
                threshold=(
                    float(calibration["threshold"])
                    if calibration.get("threshold") is not None
                    else None
                ),
                fingerprint=_optional_str(calibration.get("fingerprint")),
            ),
            template=SemanticTemplateConfig(
                premise=str(template["premise"]),
                hypothesis=str(template["hypothesis"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Manifest v3 semantic config is invalid") from exc
    from raglab.evaluation.semantic import (
        MAX_BATCH_SIZE,
        MAX_TOKENS,
        SEMANTIC_MODEL,
        SEMANTIC_REVISION,
        calibration_fixture_fingerprint,
    )

    if (
        result.model.name != SEMANTIC_MODEL
        or result.model.revision != SEMANTIC_REVISION
        or result.model.device != "cpu"
        or not 1 <= result.model.batch_size <= MAX_BATCH_SIZE
        or result.model.max_tokens != MAX_TOKENS
    ):
        raise EvaluationError("Manifest v3 semantic model configuration is not bounded")
    if result.template.premise != "{answer}" or result.template.hypothesis != "{semantic_claim}":
        raise EvaluationError("Manifest v3 semantic template must compare answer to semantic_claim")
    if result.enabled and (
        result.calibration.threshold is None or result.calibration.fingerprint is None
    ):
        raise EvaluationError(
            "Enabled semantic rescue requires calibrated threshold and fingerprint"
        )
    if result.enabled and result.calibration.fingerprint != calibration_fixture_fingerprint(
        result.calibration.fixture
    ):
        raise EvaluationError("Semantic calibration fixture fingerprint does not match")
    if result.calibration.threshold is not None and not 0.0 <= result.calibration.threshold <= 1.0:
        raise EvaluationError("Semantic calibration threshold must be between zero and one")
    return result


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError("Expected a list of strings in evaluation manifest")
    return tuple(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
