from __future__ import annotations

import asyncio

import httpx

from raglab.demo import create_app
from raglab.generation import GenerationMetrics, GenerationResponse, GenerationStrategy
from raglab.retrieval import RetrievalResponse


class _Repository:
    def healthcheck(self) -> dict[str, str]:
        return {"status": "ok"}

    def collection_stats(self, collection: str) -> dict[str, object]:
        return {"name": collection, "document_count": 3, "chunk_count": 6}


class _Pipeline:
    def generate(self, request: object) -> object:
        raise AssertionError("not called by authentication tests")


class _WorkingPipeline:
    def generate(self, request: object) -> GenerationResponse:
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


def _app() -> object:
    return create_app(
        pipeline=_Pipeline(),  # type: ignore[arg-type]
        repository=_Repository(),  # type: ignore[arg-type]
        token="secret",
    )


def test_public_status_is_read_only_and_query_requires_bearer() -> None:
    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app()),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            assert (await client.get("/health/live")).json() == {"status": "ok"}
            assert (await client.get("/health/ready")).status_code == 200
            assert 'id="citations"' in (await client.get("/")).text
            status = await client.get("/v1/demo/status")
            assert status.status_code == 200
            assert status.json()["evaluation"]["approved"] is False
            assert status.json()["evaluation"]["baseline"]["dataset_id"] == (
                "raspberry-pi-demo-v1"
            )
            response = await client.post(
                "/v1/query", json={"query": "Question", "collection": "rpi-computers"}
            )
            assert response.status_code == 401
            assert (await client.post("/v1/demo/status")).status_code == 405

    asyncio.run(exercise())


def test_unknown_collection_is_rejected_before_pipeline() -> None:
    async def exercise() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app()),  # type: ignore[arg-type]
            base_url="http://test",
        ) as client:
            return await client.post(
                "/v1/query",
                headers={"Authorization": "Bearer secret"},
                json={"query": "Question", "collection": "other"},
            )

    assert asyncio.run(exercise()).status_code == 422


def test_authorized_query_returns_inspectable_response() -> None:
    app = create_app(
        pipeline=_WorkingPipeline(),
        repository=_Repository(),  # type: ignore[arg-type]
        token="secret",
    )

    async def exercise() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/v1/query",
                headers={"Authorization": "Bearer secret"},
                json={"query": "Question", "collection": "rpi-computers", "history": []},
            )

    payload = asyncio.run(exercise()).json()
    assert payload["answer"] == "Grounded answer."
    assert payload["latency_ms"] >= 0
