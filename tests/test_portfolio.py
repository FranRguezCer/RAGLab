from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from raglab.portfolio import InvalidDemoRelease, create_portfolio_app, load_verified_release

BUILD_SHA = "a" * 40
DATASET_SHA = "b" * 64
IMAGE_DIGEST = "sha256:" + "c" * 64


def _release() -> dict[str, Any]:
    ids = [f"case-{number}" for number in range(1, 7)]
    retrieval = [
        {
            "case_id": case_id,
            "top_k": 1,
            "ranked_source_ids": [f"source-{case_id}"],
            "relevant_source_ids": [f"source-{case_id}"],
            "relevant_positions": {f"source-{case_id}": 1},
            "retrieved_relevant_source_ids": [f"source-{case_id}"],
            "missed_relevant_source_ids": [],
            "precision_at_k": 1.0,
            "recall_at_k": 1.0,
            "mrr_at_k": 1.0,
        }
        for case_id in ids
    ]
    generation = [
        {
            "case_id": case_id,
            "answer": f"Grounded answer for {case_id}.",
            "cited_source_ids": [f"source-{case_id}"],
            "found_fact_ids": ["fact"],
            "missed_fact_ids": [],
            "grounded_fact_ids": ["fact"],
            "ungrounded_fact_ids": [],
            "valid_citation_source_ids": [f"source-{case_id}"],
            "invalid_citation_source_ids": [],
            "forbidden_phrase_hits": [],
            "fact_coverage": 1.0,
            "grounded_fact_coverage": 1.0,
            "citation_precision": 1.0,
            "expected_outcome": "answer",
            "abstention_correct": True,
        }
        for case_id in ids
    ]
    traces = []
    for index, case_id in enumerate(ids):
        citation = {
            "source_uri": f"https://example.com/{case_id}",
            "source_name": f"Source {index + 1}",
            "title": f"Raspberry Pi source {index + 1}",
            "heading_path": ["Guide", case_id],
        }
        traces.append(
            {
                "case": {
                    "id": case_id,
                    "question": f"Verified question {index + 1}?",
                    "expected_outcome": "answer",
                    "collection": "rpi-computers",
                },
                "response": {
                    "answer": f"Grounded answer for {case_id}.",
                    "abstained": False,
                    "sources": [
                        {
                            "id": "S1",
                            "retrieval_result_id": f"chunk-{case_id}",
                            "document_id": f"source-{case_id}",
                            "citation": citation,
                        }
                    ],
                    "retrieval": {
                        "query": f"Verified question {index + 1}?",
                        "rewritten_query": None,
                        "query_variants": [f"Verified question {index + 1}?"],
                        "results": [
                            {
                                "id": f"chunk-{case_id}",
                                "document_id": f"source-{case_id}",
                                "content": "Retrieved evidence.",
                                "citation": citation,
                                "trace": {"rrf_score": 0.5, "reranker_score": 0.9},
                            }
                        ],
                    },
                    "strategy": "single_pass",
                    "metrics": {"model_calls": 1, "prompt_tokens": 20, "generated_tokens": 8},
                },
                "retrieval": retrieval[index],
                "generation": generation[index],
            }
        )
    report = {
        "schema_version": 1,
        "dataset_id": "raspberry-pi-demo-v1",
        "case_ids": ids,
        "top_k": 1,
        "retrieval": retrieval,
        "generation": generation,
        "retrieval_summary": {"precision_at_k": 1.0, "recall_at_k": 1.0, "mrr_at_k": 1.0},
        "generation_summary": {
            "fact_coverage": 1.0,
            "grounded_fact_coverage": 1.0,
            "citation_precision": 1.0,
            "abstention_accuracy": 1.0,
        },
    }
    baseline = {
        "schema_version": 1,
        "report_schema_version": 1,
        "dataset_id": "raspberry-pi-demo-v1",
        "dataset_sha256": DATASET_SHA,
        "case_ids": ids,
        "top_k": 1,
        "retrieval_summary": report["retrieval_summary"],
        "generation_summary": report["generation_summary"],
    }
    return {
        "schema_version": 1,
        "release": {
            "build_sha": BUILD_SHA,
            "image_digest": IMAGE_DIGEST,
            "workflow_url": "https://github.com/example/raglab/actions/runs/1",
            "created_at": "2026-09-08T12:00:00Z",
        },
        "models": {"embedding": {"name": "embed", "digest": "sha256:1"}},
        "ingestion_receipt": {"corpus": "raspberry-pi-official-docs"},
        "evaluation": {
            "traces": traces,
            "report": report,
            "metadata": {
                "build": BUILD_SHA,
                "image_digest": IMAGE_DIGEST,
                "dataset_sha256": DATASET_SHA,
                "retrieval_config": {"top_k": 1},
                "duration_seconds": 12.5,
            },
        },
        "gate": {"passed": True, "baseline": baseline},
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


async def _request(app: object, method: str, path: str) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as client:
        return await client.request(method, path)


def test_public_app_serves_six_verified_cases_without_live_query(tmp_path: Path) -> None:
    artifact = tmp_path / "demo-release.json"
    _write(artifact, _release())
    app = create_portfolio_app(release_path=artifact, build_sha=BUILD_SHA)

    live = asyncio.run(_request(app, "GET", "/health/live"))
    ready = asyncio.run(_request(app, "GET", "/health/ready"))
    status = asyncio.run(_request(app, "GET", "/v1/demo/status"))
    cases = asyncio.run(_request(app, "GET", "/v1/demo/cases"))
    selected = asyncio.run(_request(app, "GET", "/v1/demo/cases/case-1"))

    assert live.json() == {"status": "ok"}
    assert ready.json() == {"status": "ok", "build_sha": BUILD_SHA}
    assert status.json()["label"] == "Verified deployment snapshot"
    assert status.json()["dataset"]["case_count"] == 6
    assert len(cases.json()["cases"]) == 6
    assert selected.json()["answer"] == "Grounded answer for case-1."
    assert selected.json()["retrieval"]["ranking"][0]["trace"]["reranker_score"] == 0.9
    assert selected.json()["recorded_release_duration_seconds"] == 12.5
    assert asyncio.run(_request(app, "POST", "/v1/query")).status_code == 404
    assert "/v1/query" not in app.openapi()["paths"]


def test_portfolio_page_uses_static_assets_and_snapshot_language(tmp_path: Path) -> None:
    artifact = tmp_path / "demo-release.json"
    _write(artifact, _release())
    app = create_portfolio_app(release_path=artifact, build_sha=BUILD_SHA)

    page = asyncio.run(_request(app, "GET", "/"))
    styles = asyncio.run(_request(app, "GET", "/assets/styles.css"))
    script = asyncio.run(_request(app, "GET", "/assets/app.js"))

    assert "Verified deployment snapshot" in page.text
    assert '<link rel="stylesheet" href="/assets/styles.css">' in page.text
    assert "POST /v1/query" not in page.text
    assert "raw JSON" not in page.text
    assert styles.headers["content-type"].startswith("text/css")
    assert "showCase" in script.text


def test_portfolio_cli_exposes_the_evidence_only_application() -> None:
    from raglab.portfolio_cli import app

    assert "/v1/query" not in app.openapi()["paths"]


@pytest.mark.parametrize("mutation", ["build", "metrics", "query", "gate"])
def test_readiness_fails_closed_for_missing_or_manipulated_evidence(
    tmp_path: Path, mutation: str
) -> None:
    artifact = tmp_path / "demo-release.json"
    value = deepcopy(_release())
    if mutation == "build":
        value["release"]["build_sha"] = "d" * 40
    elif mutation == "metrics":
        value["evaluation"]["report"]["generation_summary"]["fact_coverage"] = 0.25
    elif mutation == "query":
        value["evaluation"]["traces"][0]["response"]["retrieval"]["query"] = "tampered"
    else:
        value["gate"]["passed"] = False
    _write(artifact, value)
    app = create_portfolio_app(release_path=artifact, build_sha=BUILD_SHA)

    response = asyncio.run(_request(app, "GET", "/health/ready"))

    assert response.status_code == 503
    assert "Release evidence is not ready" in response.json()["detail"]


def test_optional_artifact_digest_detects_any_file_change(tmp_path: Path) -> None:
    artifact = tmp_path / "demo-release.json"
    _write(artifact, _release())
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert load_verified_release(
        artifact, expected_build=BUILD_SHA, expected_sha256=digest
    )["schema_version"] == 1

    artifact.write_text(artifact.read_text() + "\n", encoding="utf-8")
    with pytest.raises(InvalidDemoRelease, match="SHA-256 does not match"):
        load_verified_release(artifact, expected_build=BUILD_SHA, expected_sha256=digest)


def test_missing_artifact_and_unknown_case_fail_cleanly(tmp_path: Path) -> None:
    missing_app = create_portfolio_app(
        release_path=tmp_path / "missing.json", build_sha=BUILD_SHA
    )
    assert asyncio.run(_request(missing_app, "GET", "/health/ready")).status_code == 503

    artifact = tmp_path / "demo-release.json"
    _write(artifact, _release())
    app = create_portfolio_app(release_path=artifact, build_sha=BUILD_SHA)
    assert asyncio.run(_request(app, "GET", "/v1/demo/cases/nope")).status_code == 404
