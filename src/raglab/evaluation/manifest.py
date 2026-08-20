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
    if version != 1:
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
                required_facts=_strings(row.get("required_facts", [])),
                should_abstain=bool(row.get("should_abstain", False)),
                history=_strings(row.get("history", [])),
                domain=_optional_str(row.get("domain")),
            )
            for row in case_rows
        )
        separate = tuple(_chunk_check(row) for row in checks.get("must_separate", []))
        keep = tuple(_chunk_check(row) for row in checks.get("must_keep", []))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("Evaluation manifest contains an invalid source or case") from exc
    manifest = EvaluationManifest(
        version,
        manifest_profile,
        sources,
        cases,
        separate,
        keep,
        cast(dict[str, Any], raw.get("config", {})),
        str(manifest_path.parent.resolve()),
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


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError("Expected a list of strings in evaluation manifest")
    return tuple(value)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
