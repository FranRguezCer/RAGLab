"""Read-only FastAPI adapter for the recruiter demo."""
# ruff: noqa: E501

from __future__ import annotations

import hmac
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Protocol, cast

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from raglab.evaluation_application import LiveEvaluationSettings, create_live_evaluation_application
from raglab.evaluation_gate import validate_promotion
from raglab.generation import GenerationConfig, GenerationRequest, GenerationResponse
from raglab.retrieval import RetrievalConfig, RetrievalRequest
from raglab.storage import PostgresRepository

DEMO_COLLECTIONS = ("rpi-computers", "rpi-microcontrollers", "rpi-camera-ai")


class GenerationStage(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


class QueryBody(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    collection: str
    history: list[str] = Field(default_factory=list, max_length=8)


def create_app(
    *,
    pipeline: GenerationStage | None = None,
    repository: PostgresRepository | None = None,
    token: str | None = None,
    artifacts_dir: Path | None = None,
) -> FastAPI:
    settings = LiveEvaluationSettings.from_env()
    app = FastAPI(title="RAGLab demo", version="1.0.0", docs_url="/docs")
    repo = repository or PostgresRepository(settings.dsn)
    generation = pipeline or create_live_evaluation_application(settings).generation_pipeline
    expected_token = token if token is not None else os.environ.get("RAGLAB_DEMO_TOKEN", "")
    artifacts = artifacts_dir or Path(os.environ.get("RAGLAB_ARTIFACTS_DIR", "artifacts"))
    baseline_path = Path(
        os.environ.get(
            "RAGLAB_EVALUATION_BASELINE",
            "data/evaluation/baselines/raspberry_pi_demo_v1.json",
        )
    )

    async def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not expected_token:
            raise HTTPException(503, "Live queries are not configured")
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not hmac.compare_digest(supplied, expected_token):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, Any]:
        return repo.healthcheck()

    @app.get("/v1/demo/status")
    async def status() -> dict[str, Any]:
        return {
            "collections": {name: repo.collection_stats(name) for name in DEMO_COLLECTIONS},
            "ingestion": _read_json(artifacts / "ingestion" / "raspberry_pi_v1.json"),
            "evaluation": _evaluation_status(
                artifacts / "evaluation" / "raspberry_pi_demo_v1.json",
                baseline_path,
            ),
            "models": {
                "embedding": settings.embedding_model,
                "generation": settings.generation_model,
            },
            "build": os.environ.get("RAGLAB_BUILD_SHA", "development"),
        }

    @app.post("/v1/query", dependencies=[Depends(authorize)])
    async def query(body: QueryBody) -> dict[str, Any]:
        if body.collection not in DEMO_COLLECTIONS:
            raise HTTPException(422, "Unknown demo collection")
        started = time.perf_counter()
        request = GenerationRequest(
            RetrievalRequest(
                body.query,
                collection=body.collection,
                history=tuple(body.history),
                config=RetrievalConfig(top_k=5, candidate_k=50, rewrite=bool(body.history)),
            ),
            GenerationConfig(model=settings.generation_model, minimum_sources=1),
        )
        response = asdict(generation.generate(request))
        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return response

    return app


def _read_json(path: Path) -> object | None:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


def _evaluation_status(run_path: Path, baseline_path: Path) -> dict[str, object]:
    run = _read_json(run_path)
    baseline = _read_json(baseline_path)
    approved = False
    if isinstance(run, dict) and isinstance(baseline, dict):
        try:
            validate_promotion(run, baseline)
            approved = True
        except (KeyError, TypeError, ValueError):
            pass
    return {"approved": approved, "run": run, "baseline": baseline}


_INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RAGLab — inspectable local RAG</title><style>
body{font:16px system-ui;max-width:960px;margin:auto;padding:2rem;background:#0b1020;color:#eef2ff}main{display:grid;gap:1rem}.card{background:#161d33;padding:1.2rem;border-radius:12px}input,select,button{font:inherit;padding:.7rem;margin:.25rem}input{width:55%}button{background:#79e2a8;border:0;border-radius:6px}pre{white-space:pre-wrap;overflow:auto}a{color:#79e2a8}
</style></head><body><main><h1>RAGLab</h1><p>A local-first RAG system built from first principles. Inspect the evidence, ingestion receipt, and evaluation gate—not hidden chain-of-thought.</p>
<section class="card"><h2>Evaluated example</h2><p><strong>How do I configure a headless Raspberry Pi?</strong></p><p>Use Raspberry Pi Imager to configure the OS, network, credentials, and SSH or Raspberry Pi Connect before first boot.</p><a href="https://www.raspberrypi.com/documentation/computers/getting-started.html">Official source</a></section>
<section class="card"><h2>Try live</h2><select id="collection"><option>rpi-computers</option><option>rpi-microcontrollers</option><option>rpi-camera-ai</option></select><select id="suggestion"></select><input id="query" value="How do I configure a headless Raspberry Pi?"><input id="token" type="password" placeholder="Bearer token (kept in memory only)"><button id="ask">Ask</button><pre id="answer"></pre><div id="citations"></div></section>
<section class="card"><h2>How it worked</h2><pre id="trace">Run a live query to inspect rewritten query, hybrid ranking, selected evidence, abstention, validation, calls, and tokens.</pre></section>
<section class="card"><h2>Ingestion and release evidence</h2><pre id="status">Loading…</pre><p><a href="https://github.com/FranRguezCer/RAGLab#90-second-tour">Engineering case study</a></p></section>
</main><script>
fetch('/v1/demo/status').then(r=>r.json()).then(x=>status.textContent=JSON.stringify(x,null,2));
const suggestions={'rpi-computers':['How do I configure a headless Raspberry Pi?','Which Raspberry Pi OS edition fits a server?'],'rpi-microcontrollers':['How do I install MicroPython on a Pico?','How can one Pico debug another?'],'rpi-camera-ai':['Where does inference run on the AI Camera?','How do I start with Picamera2?']};
function choose(){suggestion.innerHTML=suggestions[collection.value].map(x=>`<option>${x}</option>`).join('');query.value=suggestion.value} collection.onchange=choose;suggestion.onchange=()=>query.value=suggestion.value;choose();
ask.onclick=async()=>{answer.textContent='Running…';citations.replaceChildren();let r=await fetch('/v1/query',{method:'POST',headers:{'content-type':'application/json','authorization':'Bearer '+token.value},body:JSON.stringify({query:query.value,collection:collection.value,history:[]})});let x=await r.json();answer.textContent=x.answer||x.detail;for(let s of x.sources||[]){let u=s.citation?.source_uri||'';let e=document.createElement(/^https?:\/\//.test(u)?'a':'span');e.textContent=(s.citation?.title||s.citation?.source_name||s.id)+' ';if(e.tagName==='A'){e.href=u;e.target='_blank';e.rel='noreferrer'}citations.append(e)}trace.textContent=JSON.stringify(x,null,2)};
</script></body></html>"""
