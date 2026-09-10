"""Token-protected, evidence-backed application for supervised local demonstrations."""

from __future__ import annotations

import asyncio
import hmac
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from raglab.demo_evidence import InvalidDemoEvidence, load_verified_evidence
from raglab.evaluation_application import (
    GenerationStage,
    LiveEvaluationSettings,
    create_live_evaluation_application,
)
from raglab.generation import GenerationConfig, GenerationRequest, GenerationResponse
from raglab.retrieval import RetrievalConfig, RetrievalRequest
from raglab.storage import PostgresRepository

DEMO_COLLECTIONS = ("rpi-computers", "rpi-microcontrollers", "rpi-camera-ai")
_ASSETS = Path(__file__).with_name("demo_assets")
_SECURITY_HEADERS = {
    b"cache-control": b"no-store",
    b"referrer-policy": b"no-referrer",
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    ),
}


class DemoRepository(Protocol):
    def healthcheck(self) -> dict[str, Any]: ...

    def collection_stats(self, collection: str) -> dict[str, Any]: ...


class QueryBody(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    collection: str
    history: list[str] = Field(default_factory=list, max_length=8)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def secured_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(_SECURITY_HEADERS.items())
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, secured_send)


def create_app(
    *,
    pipeline: GenerationStage | None = None,
    repository: DemoRepository | None = None,
    token: str | None = None,
    evidence_path: Path | None = None,
    project_root: Path | None = None,
    expected_commit: str | None = None,
) -> FastAPI:
    settings = LiveEvaluationSettings.from_env()
    app = FastAPI(
        title="RAGLab local demo",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    repo = repository or PostgresRepository(settings.dsn)
    generation = pipeline or create_live_evaluation_application(settings).generation_pipeline
    expected_token = token if token is not None else os.environ.get("RAGLAB_DEMO_TOKEN", "")
    evidence_file = evidence_path or Path(
        os.environ.get("RAGLAB_DEMO_EVIDENCE", "artifacts/demo/evidence.json")
    )
    root = (project_root or Path(os.environ.get("RAGLAB_PROJECT_ROOT", "."))).resolve()
    inference_lock = asyncio.Lock()

    def evidence() -> dict[str, Any]:
        return load_verified_evidence(
            evidence_file,
            project_root=root,
            expected_commit=expected_commit,
        )

    async def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not expected_token or not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        return Response((_ASSETS / "index.html").read_bytes(), media_type="text/html")

    @app.get("/assets/styles.css", include_in_schema=False)
    async def styles() -> Response:
        return Response((_ASSETS / "styles.css").read_bytes(), media_type="text/css")

    @app.get("/assets/app.js", include_in_schema=False)
    async def script() -> Response:
        return Response((_ASSETS / "app.js").read_bytes(), media_type="text/javascript")

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready() -> dict[str, Any]:
        try:
            verified = evidence()
            database = repo.healthcheck()
        except (InvalidDemoEvidence, OSError, RuntimeError) as exc:
            raise HTTPException(503, f"Demo is not ready: {exc}") from exc
        return {"status": "ok", "commit": verified["prepared"]["commit"], "database": database}

    @app.get("/v1/demo/status")
    async def status() -> dict[str, Any]:
        verified = _http_evidence(evidence)
        evaluation = cast(dict[str, Any], verified["evaluation"])
        report = cast(dict[str, Any], evaluation["report"])
        metadata = cast(dict[str, Any], evaluation["metadata"])
        prepared = cast(dict[str, Any], verified["prepared"])
        return {
            "readiness": "ready",
            "collections": {name: repo.collection_stats(name) for name in DEMO_COLLECTIONS},
            "models": verified["models"],
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
            "provenance": {
                "commit": prepared["commit"],
                "evaluated_at": prepared["created_at"],
                "duration_seconds": metadata["duration_seconds"],
                "integrity_sha256": verified["integrity"]["sha256"],
            },
            "gate": {"passed": True},
            "ingestion": verified["ingestion_receipt"],
        }

    @app.get("/v1/demo/cases")
    async def cases() -> dict[str, list[dict[str, Any]]]:
        traces = cast(
            list[dict[str, Any]], _http_evidence(evidence)["evaluation"]["traces"]
        )
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
        evaluation = cast(dict[str, Any], _http_evidence(evidence)["evaluation"])
        trace = next(
            (item for item in evaluation["traces"] if item["case"]["id"] == case_id),
            None,
        )
        if trace is None:
            raise HTTPException(404, "Unknown evaluated demo case")
        response = trace["response"]
        return {
            "id": trace["case"]["id"],
            "question": trace["case"]["question"],
            "collection": trace["case"].get("collection"),
            "answer": response["answer"],
            "abstained": response["abstained"],
            "citations": response["sources"],
            "retrieval": {
                "query": response["retrieval"]["query"],
                "rewritten_query": response["retrieval"]["rewritten_query"],
                "query_variants": response["retrieval"]["query_variants"],
                "ranking": response["retrieval"]["results"],
                "validation": trace["retrieval"],
            },
            "generation": {
                "strategy": response["strategy"],
                "metrics": response["metrics"],
                "validation": trace["generation"],
            },
            "evaluation_duration_seconds": evaluation["metadata"]["duration_seconds"],
        }

    @app.post("/v1/query", dependencies=[Depends(authorize)])
    async def query(body: QueryBody) -> dict[str, Any]:
        if body.collection not in DEMO_COLLECTIONS:
            raise HTTPException(422, "Unknown demo collection")
        if inference_lock.locked():
            raise HTTPException(429, "GPU is busy; wait for the active query to finish")
        request = GenerationRequest(
            RetrievalRequest(
                body.query,
                collection=body.collection,
                history=tuple(body.history),
                config=RetrievalConfig(top_k=5, candidate_k=50, rewrite=bool(body.history)),
            ),
            GenerationConfig(model=settings.generation_model, minimum_sources=1),
        )
        started = time.perf_counter()
        async with inference_lock:
            result = await _generate_in_thread(generation, request)
        duration = round((time.perf_counter() - started) * 1000, 2)
        response = asdict(result)
        response["telemetry"] = {
            "kind": "live_query",
            "has_ground_truth": False,
            "duration_ms": duration,
            "metrics": response.get("metrics", {}),
        }
        return response

    return app


def _http_evidence(loader: Any) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], loader())
    except InvalidDemoEvidence as exc:
        raise HTTPException(503, f"Prepared evidence is unavailable: {exc}") from exc


async def _generate_in_thread(
    generation: GenerationStage, request: GenerationRequest
) -> GenerationResponse:
    """Offload blocking inference without tying process shutdown to an executor pool."""

    completed = threading.Event()
    outcome: list[GenerationResponse | BaseException] = []

    def run() -> None:
        try:
            outcome.append(generation.generate(request))
        except BaseException as exc:
            outcome.append(exc)
        finally:
            completed.set()

    threading.Thread(target=run, name="raglab-inference", daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.01)
    result = outcome[0]
    if isinstance(result, BaseException):
        raise result
    return result


__all__ = ["DEMO_COLLECTIONS", "QueryBody", "create_app"]
