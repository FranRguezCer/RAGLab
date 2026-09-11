from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from demo_fixture import COMMIT, create_evidence

from raglab.demo import create_app
from raglab.generation import GenerationMetrics, GenerationResponse, GenerationStrategy
from raglab.retrieval import RetrievalResponse


class _Repository:
    def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    def collection_stats(self, collection: str) -> dict[str, object]:
        return {"name": collection, "document_count": 3, "chunk_count": 6}


class _Pipeline:
    def __init__(self, block: threading.Event | None = None) -> None:
        self.block = block
        self.started = threading.Event()

    def generate(self, request: object) -> GenerationResponse:
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=2)
        return GenerationResponse(
            answer="Grounded answer.",
            abstained=False,
            sources=(),
            retrieval=RetrievalResponse("Question", None, ("Question",), (), ()),
            strategy=GenerationStrategy.SINGLE_PASS,
            source_shortfall=False,
            minimum_sources=1,
            source_count=0,
            metrics=GenerationMetrics(1, 10, 8, 2),
        )


def _app(tmp_path: Path, pipeline: _Pipeline | None = None) -> object:
    evidence = create_evidence(tmp_path)
    return create_app(
        pipeline=pipeline or _Pipeline(),
        repository=_Repository(),
        token="secret",
        evidence_path=evidence,
        project_root=tmp_path,
        expected_commit=COMMIT,
    )


def test_ui_security_status_and_six_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("raglab.demo_evidence.datetime", _FixedDatetime)

    async def exercise() -> None:
        app = _app(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",  # type: ignore[arg-type]
        ) as client:
            page = await client.get("/")
            assert '<script src="/assets/app.js" defer>' in page.text
            assert page.headers["cache-control"] == "no-store"
            assert page.headers["x-frame-options"] == "DENY"
            assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
            assert (await client.get("/openapi.json")).status_code == 404
            assert (await client.get("/assets/styles.css")).status_code == 200
            assert (await client.get("/assets/app.js")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 200
            status = (await client.get("/v1/demo/status")).json()
            assert status["dataset"]["case_count"] == 6
            assert status["scorecard"]["retrieval"]["recall_at_k"] == 1.0
            cases = (await client.get("/v1/demo/cases")).json()["cases"]
            assert len(cases) == 6
            selected = (await client.get(f"/v1/demo/cases/{cases[0]['id']}")).json()
            assert "validation" in selected["retrieval"]
            assert (await client.get("/v1/demo/cases/missing")).status_code == 404
            unauthorized = await client.post(
                "/v1/query", json={"query": "Q", "collection": "rpi-computers"}
            )
            assert unauthorized.status_code == 401

    asyncio.run(exercise())


def test_authorized_live_query_is_not_presented_as_evaluated(tmp_path: Path) -> None:
    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app(tmp_path)),
            base_url="http://test",  # type: ignore[arg-type]
        ) as client:
            response = await client.post(
                "/v1/query",
                headers={"Authorization": "Bearer secret"},
                json={"query": "Question", "collection": "rpi-computers"},
            )
            assert response.status_code == 200
            assert response.json()["telemetry"]["has_ground_truth"] is False
            assert response.json()["telemetry"]["kind"] == "live_query"
            unknown = await client.post(
                "/v1/query",
                headers={"Authorization": "Bearer secret"},
                json={"query": "Question", "collection": "unknown"},
            )
            assert unknown.status_code == 422

    asyncio.run(exercise())


def test_second_simultaneous_query_is_rejected(tmp_path: Path) -> None:
    release = threading.Event()
    pipeline = _Pipeline(release)

    async def exercise() -> None:
        app = _app(tmp_path, pipeline)
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",  # type: ignore[arg-type]
            ) as client,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",  # type: ignore[arg-type]
            ) as concurrent_client,
        ):
            first = asyncio.create_task(
                client.post(
                    "/v1/query",
                    headers={"Authorization": "Bearer secret"},
                    json={"query": "First", "collection": "rpi-computers"},
                )
            )
            for _ in range(100):
                if pipeline.started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert pipeline.started.is_set()
            second = await concurrent_client.post(
                "/v1/query",
                headers={"Authorization": "Bearer secret"},
                json={"query": "Second", "collection": "rpi-computers"},
            )
            release.set()
            assert second.status_code == 429
            assert (await first).status_code == 200

    asyncio.run(exercise())


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


def test_demo_cli_disables_schema() -> None:
    from raglab.demo_cli import app

    assert app.openapi_url is None
