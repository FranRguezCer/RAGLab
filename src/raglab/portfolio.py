"""Stateless public application for serving verified release evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from raglab.evaluation_gate import validate_promotion

_ASSETS = Path(__file__).with_name("portfolio_assets")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_BUILD_SHA = re.compile(r"^[0-9a-f]{40}$")


class InvalidDemoRelease(ValueError):
    """Raised when public evidence cannot be trusted or served."""


def create_portfolio_app(
    *,
    release_path: Path | None = None,
    build_sha: str | None = None,
    release_sha256: str | None = None,
) -> FastAPI:
    """Create the CPU-only app without composing database or model infrastructure."""

    evidence_path = release_path or Path(
        os.environ.get("RAGLAB_DEMO_RELEASE", "/app/demo-release.json")
    )
    expected_build = build_sha if build_sha is not None else os.environ.get("RAGLAB_BUILD_SHA", "")
    expected_digest = (
        release_sha256
        if release_sha256 is not None
        else os.environ.get("RAGLAB_DEMO_RELEASE_SHA256")
    )
    app = FastAPI(
        title="RAGLab verified portfolio",
        version="1.0.0",
        docs_url="/docs",
        description="Read-only evidence from a gated RAG release.",
    )
    def verified_release() -> dict[str, Any]:
        return load_verified_release(
            evidence_path,
            expected_build=expected_build,
            expected_sha256=expected_digest,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return (_ASSETS / "index.html").read_text(encoding="utf-8")

    @app.get("/assets/styles.css", response_class=Response, include_in_schema=False)
    async def styles() -> Response:
        return Response(
            (_ASSETS / "styles.css").read_text(encoding="utf-8"), media_type="text/css"
        )

    @app.get("/assets/app.js", response_class=Response, include_in_schema=False)
    async def script() -> Response:
        return Response(
            (_ASSETS / "app.js").read_text(encoding="utf-8"),
            media_type="text/javascript",
        )

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        try:
            evidence = verified_release()
        except InvalidDemoRelease as exc:
            raise HTTPException(503, f"Release evidence is not ready: {exc}") from exc
        return {"status": "ok", "build_sha": evidence["release"]["build_sha"]}

    @app.get("/v1/demo/status")
    async def status() -> dict[str, Any]:
        evidence = _http_release(verified_release)
        evaluation = cast(dict[str, Any], evidence["evaluation"])
        report = cast(dict[str, Any], evaluation["report"])
        metadata = cast(dict[str, Any], evaluation["metadata"])
        baseline = cast(dict[str, Any], cast(dict[str, Any], evidence["gate"])["baseline"])
        return {
            "label": "Verified deployment snapshot",
            "release": evidence["release"],
            "models": evidence["models"],
            "dataset": {
                "id": report["dataset_id"],
                "sha256": metadata["dataset_sha256"],
                "case_count": len(evaluation["traces"]),
                "top_k": metadata["retrieval_config"]["top_k"],
            },
            "scorecard": {
                "retrieval": report["retrieval_summary"],
                "generation": report["generation_summary"],
            },
            "gate": {
                "passed": True,
                "baseline_schema_version": baseline["schema_version"],
            },
            "ingestion": evidence["ingestion_receipt"],
        }

    @app.get("/v1/demo/cases")
    async def cases() -> dict[str, list[dict[str, Any]]]:
        evidence = _http_release(verified_release)
        traces = cast(list[dict[str, Any]], cast(dict[str, Any], evidence["evaluation"])["traces"])
        return {
            "cases": [
                {
                    "id": trace["case"]["id"],
                    "question": trace["case"]["question"],
                    "collection": trace["case"].get("collection"),
                    "outcome": "abstained" if trace["response"]["abstained"] else "answered",
                }
                for trace in traces
            ]
        }

    @app.get("/v1/demo/cases/{case_id}")
    async def case(case_id: str) -> dict[str, Any]:
        evidence = _http_release(verified_release)
        evaluation = cast(dict[str, Any], evidence["evaluation"])
        trace = next(
            (
                item
                for item in cast(list[dict[str, Any]], evaluation["traces"])
                if item["case"]["id"] == case_id
            ),
            None,
        )
        if trace is None:
            raise HTTPException(404, "Unknown verified demo case")
        response = cast(dict[str, Any], trace["response"])
        retrieval_response = cast(dict[str, Any], response["retrieval"])
        return {
            "id": trace["case"]["id"],
            "question": trace["case"]["question"],
            "collection": trace["case"].get("collection"),
            "answer": response["answer"],
            "abstained": response["abstained"],
            "citations": response["sources"],
            "retrieval": {
                "query": retrieval_response["query"],
                "rewritten_query": retrieval_response["rewritten_query"],
                "query_variants": retrieval_response["query_variants"],
                "ranking": retrieval_response["results"],
                "validation": trace["retrieval"],
            },
            "generation": {
                "strategy": response["strategy"],
                "metrics": response["metrics"],
                "validation": trace["generation"],
            },
            "recorded_release_duration_seconds": evaluation["metadata"]["duration_seconds"],
        }

    return app


def load_verified_release(
    path: Path,
    *,
    expected_build: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and validate the complete evidence on every readiness/API request."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InvalidDemoRelease("demo-release.json is missing or unreadable") from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        configured_digest = expected_sha256.removeprefix("sha256:").lower()
        if re.fullmatch(r"[0-9a-f]{64}", configured_digest) is None:
            raise InvalidDemoRelease("RAGLAB_DEMO_RELEASE_SHA256 must be a SHA-256 digest")
        if not _constant_time_equal(actual, configured_digest):
            raise InvalidDemoRelease("demo-release.json SHA-256 does not match")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidDemoRelease("demo-release.json is not valid JSON") from exc
    if not isinstance(value, dict):
        raise InvalidDemoRelease("demo-release.json must contain an object")

    evidence = cast(dict[str, Any], value)
    _validate_release_shape(evidence, expected_build=expected_build)
    try:
        validate_promotion(evidence["evaluation"], evidence["gate"]["baseline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidDemoRelease(f"evaluation gate rejected evidence: {exc}") from exc
    return evidence


def _validate_release_shape(evidence: dict[str, Any], *, expected_build: str) -> None:
    if evidence.get("schema_version") != 1:
        raise InvalidDemoRelease("unsupported release schema_version")
    release = _mapping(evidence.get("release"), "release")
    evaluation = _mapping(evidence.get("evaluation"), "evaluation")
    gate = _mapping(evidence.get("gate"), "gate")
    metadata = _mapping(evaluation.get("metadata"), "evaluation.metadata")
    _mapping(evaluation.get("report"), "evaluation.report")
    traces = evaluation.get("traces")
    if not isinstance(traces, list) or not traces:
        raise InvalidDemoRelease("evaluation.traces must be a non-empty array")
    if not _mapping(evidence.get("models"), "models"):
        raise InvalidDemoRelease("models cannot be empty")
    if not _mapping(evidence.get("ingestion_receipt"), "ingestion_receipt"):
        raise InvalidDemoRelease("ingestion_receipt cannot be empty")
    if gate.get("passed") is not True:
        raise InvalidDemoRelease("release gate did not pass")
    _mapping(gate.get("baseline"), "gate.baseline")

    build = release.get("build_sha")
    if not isinstance(build, str) or _BUILD_SHA.fullmatch(build) is None:
        raise InvalidDemoRelease("release.build_sha must be a full Git SHA")
    if not expected_build:
        raise InvalidDemoRelease("RAGLAB_BUILD_SHA is required")
    if build != expected_build or metadata.get("build") != build:
        raise InvalidDemoRelease("release, evaluation, and runtime builds do not match")
    image_digest = release.get("image_digest")
    if not isinstance(image_digest, str) or _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise InvalidDemoRelease("release.image_digest must be an immutable SHA-256 digest")
    if metadata.get("image_digest") != image_digest:
        raise InvalidDemoRelease("release and evaluation image digests do not match")
    workflow_url = release.get("workflow_url")
    if not isinstance(workflow_url, str) or urlsplit(workflow_url).scheme != "https":
        raise InvalidDemoRelease("release.workflow_url must be HTTPS")
    created_at = release.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise InvalidDemoRelease("release.created_at must be non-empty")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidDemoRelease("release.created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise InvalidDemoRelease("release.created_at must include a timezone")

    for index, raw_trace in enumerate(traces):
        trace = _mapping(raw_trace, f"evaluation.traces[{index}]")
        case = _mapping(trace.get("case"), f"evaluation.traces[{index}].case")
        response = _mapping(trace.get("response"), f"evaluation.traces[{index}].response")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise InvalidDemoRelease(f"evaluation.traces[{index}].case.id must be non-empty")
        if not isinstance(case.get("question"), str) or not case["question"]:
            raise InvalidDemoRelease(f"evaluation.traces[{index}].case.question must be non-empty")
        if not isinstance(response.get("answer"), str):
            raise InvalidDemoRelease(f"evaluation trace {case_id!r} has no answer")
        if not isinstance(response.get("abstained"), bool):
            raise InvalidDemoRelease(f"evaluation trace {case_id!r} has invalid abstention")
        if not isinstance(response.get("sources"), list):
            raise InvalidDemoRelease(f"evaluation trace {case_id!r} has invalid sources")
        retrieval_response = _mapping(
            response.get("retrieval"), f"evaluation trace {case_id!r} response.retrieval"
        )
        if retrieval_response.get("query") != case.get("question"):
            raise InvalidDemoRelease(f"evaluation trace {case_id!r} has a mismatched query")
        if not isinstance(retrieval_response.get("results"), list):
            raise InvalidDemoRelease(f"evaluation trace {case_id!r} has invalid ranking")
        _mapping(response.get("metrics"), f"evaluation trace {case_id!r} response.metrics")


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidDemoRelease(f"{field} must be an object")
    return dict(value)


def _constant_time_equal(actual: str, expected: str) -> bool:
    import hmac

    return hmac.compare_digest(actual, expected.lower())


def _http_release(loader: Any) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], loader())
    except InvalidDemoRelease as exc:
        raise HTTPException(503, f"Release evidence is not ready: {exc}") from exc


__all__ = ["InvalidDemoRelease", "create_portfolio_app", "load_verified_release"]
