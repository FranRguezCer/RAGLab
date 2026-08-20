"""Adapters that execute the evaluation against PostgreSQL and Ollama."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from raglab.chunking import ChunkingConfig
from raglab.contracts import CollectionConfig, SourceInput
from raglab.embeddings import OllamaEmbeddingProvider
from raglab.errors import EvaluationError, GenerationError
from raglab.evaluation.manifest import resolved_source
from raglab.evaluation.metrics import normalize
from raglab.evaluation.models import (
    PROTECTED_COLLECTION_PREFIX,
    EvaluationCase,
    EvaluationExecutor,
    EvaluationJudge,
    EvaluationManifest,
    GenerationObservation,
    IngestionObservation,
    RetrievalObservation,
)
from raglab.generation import GenerationConfig, GenerationPipeline, GenerationRequest
from raglab.generation.ollama import OllamaGenerationModel
from raglab.pipeline import ingest
from raglab.retrieval import (
    CollectionMetadata,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalRequest,
    RetrievalResponse,
)
from raglab.retrieval.repository import PostgresRetrievalRepository
from raglab.retrieval.rewriting import OllamaQueryRewriter
from raglab.storage import PostgresRepository


class _FixedRetrievalStage:
    def __init__(self, response: RetrievalResponse, metadata: CollectionMetadata) -> None:
        self.response = response
        self.metadata = metadata

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        del request
        return self.response

    def collection_metadata(self, collection: str) -> CollectionMetadata | None:
        return self.metadata if collection == self.metadata.name else None


class LiveEvaluationExecutor(EvaluationExecutor):
    """Run the native pipelines while retaining stable manifest identities."""

    def __init__(
        self,
        *,
        dsn: str,
        generation_model: str = "qwen3:4b",
        embedding_model: str = "qwen3-embedding:0.6b",
        ollama_base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "5m",
    ) -> None:
        self.dsn = dsn
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.storage = PostgresRepository(dsn)
        self._manifest: EvaluationManifest | None = None
        self._source_ids: dict[str, str] = {}
        self._responses: dict[str, RetrievalResponse] = {}
        self._embedding = OllamaEmbeddingProvider(
            model=embedding_model,
            dimension=1024,
            base_url=self.ollama_base_url,
            num_gpu=0,
            num_ctx=4096,
            keep_alive=keep_alive,
        )
        self._retrieval = RetrievalPipeline(
            PostgresRetrievalRepository(dsn),
            self._embedding,
            rewriter=OllamaQueryRewriter(model=generation_model, base_url=self.ollama_base_url),
        )
        self._model = OllamaGenerationModel(base_url=self.ollama_base_url)

    def prepare(
        self, manifest: EvaluationManifest, *, collection: str, reuse_index: bool
    ) -> IngestionObservation:
        started = time.perf_counter()
        self._manifest = manifest
        self._source_ids = {
            resolved_source(manifest, source): source.id for source in manifest.sources
        }
        self.storage.migrate()
        config = ChunkingConfig(semantic_percentile=85)
        collections = self._collections(manifest, collection)
        if not reuse_index:
            for target in collections:
                self.storage.reset_evaluation_collection(
                    target, protected_prefix=PROTECTED_COLLECTION_PREFIX
                )
            for source in manifest.sources:
                location = resolved_source(manifest, source)
                source_input = (
                    SourceInput.url(location, source_id=source.id)
                    if urlsplit(location).scheme in {"http", "https"}
                    else SourceInput.path(Path(location), source_id=source.id)
                )
                targets = (
                    (collection,)
                    if manifest.profile == "core"
                    else (f"{collection}-all", f"{collection}-{source.domain}")
                )
                for target in targets:
                    ingest(
                        source_input,
                        self._collection_config(target, config),
                        dsn=self.dsn,
                        chunk_config=config,
                    )
        primary_collection = collection if manifest.profile == "core" else f"{collection}-all"
        rows = self.storage.evaluation_chunks(primary_collection)
        if not rows:
            raise EvaluationError(
                f"Evaluation collection {collection!r} is empty; omit --reuse-index to build it"
            )
        chunks_by_source: dict[str, list[str]] = {}
        for uri, content, _tokens in rows:
            source_id = self._source_ids.get(uri)
            if source_id is not None:
                chunks_by_source.setdefault(source_id, []).append(normalize(content))
        separation = sum(
            not any(
                normalize(check.left_anchor) in chunk and normalize(check.right_anchor) in chunk
                for chunk in chunks_by_source.get(check.source_id, [])
            )
            for check in manifest.must_separate
        )
        cohesion = sum(
            any(
                normalize(check.left_anchor) in chunk and normalize(check.right_anchor) in chunk
                for chunk in chunks_by_source.get(check.source_id, [])
            )
            for check in manifest.must_keep
        )
        return IngestionObservation(
            len({uri for uri, _content, _tokens in rows}),
            len(rows),
            tuple(tokens for _uri, _content, tokens in rows),
            separation,
            len(manifest.must_separate),
            cohesion,
            len(manifest.must_keep),
            (time.perf_counter() - started) * 1000,
        )

    def retrieve(
        self, case: EvaluationCase, *, collection: str, exact: bool
    ) -> RetrievalObservation:
        started = time.perf_counter()
        config = RetrievalConfig(
            candidate_k=50,
            top_k=5,
            exact=exact,
            rewrite=bool(case.history),
            rerank=True,
        )
        response = self._retrieval.retrieve(
            RetrievalRequest(
                case.query,
                self._case_collection(case, collection),
                history=case.history,
                config=config,
            )
        )
        if not exact:
            self._responses[case.id] = response
        source_ids = tuple(
            source_id
            for item in response.results
            if (source_id := self._source_ids.get(item.citation.source_uri)) is not None
        )
        anchors = tuple(
            tuple(
                fact
                for fact in case.required_facts
                if normalize(fact) in normalize(item.content)
            )
            for item in response.results
        )
        aggregate_ids: tuple[str, ...] = ()
        if self._manifest is not None and self._manifest.profile == "live":
            aggregate = self._retrieval.retrieve(
                RetrievalRequest(
                    case.query,
                    f"{collection}-all",
                    history=case.history,
                    config=config,
                )
            )
            aggregate_ids = tuple(
                source_id
                for item in aggregate.results
                if (source_id := self._source_ids.get(item.citation.source_uri)) is not None
            )
        return RetrievalObservation(
            source_ids,
            anchors,
            (time.perf_counter() - started) * 1000,
            aggregate_ids,
        )

    def generate(
        self, case: EvaluationCase, *, collection: str
    ) -> GenerationObservation:
        response = self._responses.get(case.id)
        if response is None:
            raise EvaluationError(f"Retrieval must run before generation for case {case.id!r}")
        target_collection = self._case_collection(case, collection)
        metadata = self._retrieval.collection_metadata(target_collection)
        if metadata is None:
            raise EvaluationError(f"Evaluation collection {collection!r} disappeared")
        fixed = _FixedRetrievalStage(response, metadata)
        pipeline = GenerationPipeline(
            fixed,
            self._model,
            embedding_model=self.embedding_model,
        )
        started = time.perf_counter()
        result = pipeline.generate(
            GenerationRequest(
                RetrievalRequest(
                    case.query,
                    target_collection,
                    history=case.history,
                    config=replace(RetrievalConfig(), top_k=5),
                ),
                GenerationConfig(
                    model=self.generation_model,
                    keep_alive=self.keep_alive,
                    minimum_sources=max(1, len(case.expected_source_ids)),
                ),
            )
        )
        cited = tuple(
            source_id
            for source in result.sources
            if (source_id := self._source_ids.get(source.citation.source_uri)) is not None
        )
        return GenerationObservation(
            result.answer,
            result.abstained,
            cited,
            result.metrics.prompt_tokens,
            result.metrics.generated_tokens,
            result.metrics.model_calls,
            (time.perf_counter() - started) * 1000,
        )

    def release_generator(self) -> None:
        body = {"model": self.generation_model, "keep_alive": 0}
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30):  # noqa: S310
                pass
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GenerationError(f"Could not unload {self.generation_model}: {exc}") from exc

    def _collection_config(self, name: str, config: ChunkingConfig) -> CollectionConfig:
        return CollectionConfig(
            name=name,
            model=self.embedding_model,
            chunk_config={
                "strategy": "structure_plus_semantics",
                "target_tokens": config.target_tokens,
                "min_tokens": config.min_tokens,
                "max_tokens": config.max_tokens,
                "semantic_percentile": 85,
                "overlap_tokens": config.overlap_tokens,
            },
        )

    @staticmethod
    def _collections(manifest: EvaluationManifest, base: str) -> tuple[str, ...]:
        if manifest.profile == "core":
            return (base,)
        domains = sorted({source.domain for source in manifest.sources if source.domain})
        return (f"{base}-all", *(f"{base}-{domain}" for domain in domains))

    def _case_collection(self, case: EvaluationCase, base: str) -> str:
        if self._manifest is None or self._manifest.profile == "core":
            return base
        if not case.domain:
            raise EvaluationError(f"Live case {case.id!r} has no domain")
        return f"{base}-{case.domain}"


class OllamaEvaluationJudge(EvaluationJudge):
    """Optional schema-constrained advisory judge."""

    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:11434") -> None:
        self._model_name = model
        self.adapter = OllamaGenerationModel(base_url=base_url)

    @property
    def model(self) -> str:
        return self._model_name

    def evaluate(self, case: EvaluationCase, answer: str) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "grounded": {"type": "boolean"},
                "relevant": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["grounded", "relevant", "reason"],
        }
        prompt = (
            f"Question: {case.query}\nAnswer: {answer}\n"
            f"Required facts: {list(case.required_facts)}\n"
            "Assess only whether the answer is relevant and consistent with the required facts."
        )
        invocation = self.adapter.generate(
            prompt,
            system="You are an advisory RAG evaluator. Return only the requested JSON.",
            schema=schema,
            config=GenerationConfig(model=self.model, minimum_sources=1),
        )
        return invocation.payload
