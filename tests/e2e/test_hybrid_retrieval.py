from __future__ import annotations

import os
from uuid import uuid4

import pytest

from raglab import Chunk, CollectionConfig, ConvertedDocument, EmbeddedChunk
from raglab.embeddings import OllamaEmbeddingProvider
from raglab.retrieval import (
    PostgresRetrievalRepository,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalRequest,
)
from raglab.storage import PostgresRepository

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    os.getenv("RAGLAB_RETRIEVAL_E2E") != "1" or not os.getenv("RAGLAB_TEST_DSN"),
    reason="Set RAGLAB_RETRIEVAL_E2E=1 and RAGLAB_TEST_DSN to run ParadeDB/Ollama/BGE",
)
def test_real_hybrid_retrieval_with_ollama_and_bge() -> None:
    dsn = os.environ["RAGLAB_TEST_DSN"]
    storage = PostgresRepository(dsn)
    storage.migrate()
    collection = CollectionConfig(name=f"retrieval-e2e-{uuid4().hex}")
    contents = (
        "Fault E17 occurs when coolant pressure falls below 1.2 bar.",
        "Reset the controller only after restoring coolant pressure.",
        "The greenhouse lighting schedule starts at sunrise.",
    )
    embeddings = OllamaEmbeddingProvider().embed_documents(contents)
    for index, (content, embedding) in enumerate(zip(contents, embeddings, strict=True)):
        document = ConvertedDocument(
            source_uri=f"memory://retrieval-e2e/{index}",
            markdown=content,
            content_hash=uuid4().hex,
            converter="test",
            converter_version="1",
            source_name=f"document-{index}.md",
        )
        storage.store(
            collection,
            document,
            [
                EmbeddedChunk(
                    Chunk(0, content, content, len(content.split()), (), 1, 1), tuple(embedding)
                )
            ],
            uuid4().hex,
        )

    response = RetrievalPipeline(
        PostgresRetrievalRepository(dsn), OllamaEmbeddingProvider()
    ).retrieve(
        RetrievalRequest(
            "What causes fault E17?",
            collection=collection.name,
            config=RetrievalConfig(top_k=2),
        )
    )

    assert response.results
    assert "coolant pressure" in response.results[0].content.lower()
    assert any(result.trace.bm25_score is not None for result in response.results)
    assert all(result.trace.reranker_score is not None for result in response.results)
    assert all(result.trace.mmr_score is not None for result in response.results)
