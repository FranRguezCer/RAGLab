"""Build and verify local evidence consumed by the supervised demo."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from raglab.evaluation_gate import validate_promotion

SCHEMA_VERSION = 1
DEFAULT_MAX_AGE = timedelta(days=7)


class InvalidDemoEvidence(ValueError):
    """Raised when prepared demo evidence cannot be trusted."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_evidence(
    evaluation: Mapping[str, Any],
    baseline: Mapping[str, Any],
    ingestion_receipt: Mapping[str, Any],
    models: Mapping[str, Any],
    *,
    commit: str,
    inputs: Mapping[str, str],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create a self-authenticating local bundle after recomputing the evaluation gate."""

    validate_promotion(evaluation, baseline)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise InvalidDemoEvidence("commit must be a lowercase 40-character Git SHA")
    if not ingestion_receipt:
        raise InvalidDemoEvidence("ingestion receipt cannot be empty")
    _validate_models(models)
    timestamp = created_at or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    _parse_timestamp(timestamp)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prepared": {"commit": commit, "created_at": timestamp, "inputs": dict(inputs)},
        "models": dict(models),
        "ingestion_receipt": dict(ingestion_receipt),
        "evaluation": dict(evaluation),
        "gate": {"passed": True, "baseline": dict(baseline)},
    }
    payload["integrity"] = {"sha256": _payload_digest(payload)}
    return payload


def write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_verified_evidence(
    path: Path,
    *,
    project_root: Path,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed if evidence is missing, changed, old, or no longer matches repository inputs."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidDemoEvidence("prepared evidence is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise InvalidDemoEvidence("prepared evidence must contain an object")
    evidence = cast(dict[str, Any], value)
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise InvalidDemoEvidence("unsupported evidence schema")
    integrity = evidence.get("integrity")
    claimed = integrity.get("sha256") if isinstance(integrity, dict) else None
    if not isinstance(claimed, str) or not hmac.compare_digest(claimed, _payload_digest(evidence)):
        raise InvalidDemoEvidence("prepared evidence integrity check failed")
    prepared = evidence.get("prepared")
    if not isinstance(prepared, dict):
        raise InvalidDemoEvidence("prepared metadata is missing")
    actual_commit = expected_commit or current_commit(project_root)
    if prepared.get("commit") != actual_commit:
        raise InvalidDemoEvidence("prepared evidence belongs to a different commit")
    created = _parse_timestamp(prepared.get("created_at"))
    clock = now or datetime.now(UTC)
    if created > clock + timedelta(minutes=5) or clock - created > max_age:
        raise InvalidDemoEvidence("prepared evidence is stale")
    inputs = prepared.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise InvalidDemoEvidence("prepared input fingerprints are missing")
    for relative, expected in inputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise InvalidDemoEvidence("prepared input fingerprint is invalid")
        candidate = (project_root / relative).resolve()
        try:
            candidate.relative_to(project_root.resolve())
        except ValueError as exc:
            raise InvalidDemoEvidence("prepared input escapes the project") from exc
        try:
            actual = sha256_file(candidate)
        except OSError as exc:
            raise InvalidDemoEvidence(f"prepared input is missing: {relative}") from exc
        if not hmac.compare_digest(actual, expected):
            raise InvalidDemoEvidence(f"prepared input changed: {relative}")
    try:
        validate_promotion(evidence["evaluation"], evidence["gate"]["baseline"])
        _validate_models(cast(Mapping[str, Any], evidence["models"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidDemoEvidence(f"evaluation evidence is invalid: {exc}") from exc
    return evidence


def _payload_digest(evidence: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in evidence.items() if key != "integrity"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _parse_timestamp(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise InvalidDemoEvidence("created_at must be an ISO-8601 timestamp")
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidDemoEvidence("created_at must be an ISO-8601 timestamp") from exc
    if value.tzinfo is None:
        raise InvalidDemoEvidence("created_at must include a timezone")
    return value.astimezone(UTC)


def _validate_models(models: Mapping[str, Any]) -> None:
    required = {"embedding", "generation", "reranker", "nli"}
    if set(models) != required:
        raise InvalidDemoEvidence(
            "model identities must cover embedding, generation, reranker, and nli"
        )
    for role, raw in models.items():
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise InvalidDemoEvidence(f"models.{role} is invalid")
        identity = raw.get("digest", raw.get("revision"))
        if not isinstance(identity, str) or not identity:
            raise InvalidDemoEvidence(f"models.{role} requires a digest or revision")


__all__ = [
    "DEFAULT_MAX_AGE",
    "InvalidDemoEvidence",
    "build_evidence",
    "current_commit",
    "load_verified_evidence",
    "sha256_file",
    "write_evidence",
]
