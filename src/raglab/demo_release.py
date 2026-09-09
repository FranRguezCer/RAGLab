"""Build the immutable evidence bundle served by the public portfolio application."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from raglab.evaluation_gate import validate_promotion

DEMO_RELEASE_SCHEMA_VERSION = 1


def build_demo_release(
    evaluation: Mapping[str, Any],
    baseline: Mapping[str, Any],
    ingestion_receipt: Mapping[str, Any],
    models: Mapping[str, Any],
    *,
    build_sha: str,
    image_digest: str,
    workflow_url: str,
    created_at: str,
) -> dict[str, Any]:
    """Return a release bundle only after its evaluation evidence passes the gate."""
    validate_promotion(evaluation, baseline)
    if not re.fullmatch(r"[0-9a-f]{40}", build_sha):
        raise ValueError("build_sha must be a lowercase 40-character Git SHA")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ValueError("image_digest must be an immutable sha256 digest")
    evaluation_metadata = evaluation.get("metadata")
    if not isinstance(evaluation_metadata, dict):
        raise ValueError("evaluation.metadata must be an object")
    if evaluation_metadata.get("build") != build_sha:
        raise ValueError("evaluation build does not match build_sha")
    if evaluation_metadata.get("image_digest") != image_digest:
        raise ValueError("evaluation image does not match image_digest")
    parsed_url = urlparse(workflow_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("workflow_url must be an absolute HTTPS URL")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    if not ingestion_receipt:
        raise ValueError("ingestion_receipt cannot be empty")
    _validate_models(models)
    return {
        "schema_version": DEMO_RELEASE_SCHEMA_VERSION,
        "release": {
            "build_sha": build_sha,
            "image_digest": image_digest,
            "workflow_url": workflow_url,
            "created_at": created_at,
        },
        "models": dict(models),
        "ingestion_receipt": dict(ingestion_receipt),
        "evaluation": dict(evaluation),
        "gate": {"passed": True, "baseline": dict(baseline)},
    }


def _validate_models(models: Mapping[str, Any]) -> None:
    if not models:
        raise ValueError("models cannot be empty")
    for role, raw in models.items():
        if not isinstance(role, str) or not role or not isinstance(raw, dict):
            raise ValueError("models must map roles to model objects")
        model = cast(dict[str, object], raw)
        if not isinstance(model.get("name"), str) or not model["name"]:
            raise ValueError(f"models.{role}.name must be a non-empty string")
        identity = model.get("digest", model.get("revision"))
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"models.{role} requires a resolved digest or revision")


def _read_object(path: Path, label: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return cast(Mapping[str, Any], value)


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raglab-build-demo-release")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ingestion-receipt", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--build-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_demo_release(
            _read_object(args.evaluation, "evaluation"),
            _read_object(args.baseline, "baseline"),
            _read_object(args.ingestion_receipt, "ingestion receipt"),
            _read_object(args.models, "models"),
            build_sha=args.build_sha,
            image_digest=args.image_digest,
            workflow_url=args.workflow_url,
            created_at=args.created_at
            or datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        _write_atomic(args.output, artifact)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"raglab-build-demo-release: error: {exc}\n")
    print(json.dumps({"output": str(args.output), "build_sha": args.build_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
