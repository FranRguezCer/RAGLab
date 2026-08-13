from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from raglab import CollectionConfig, SourceInput, ingest
from raglab.embeddings import OllamaEmbeddingProvider
from raglab.storage import PostgresRepository

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not os.getenv("RAGLAB_E2E"), reason="Set RAGLAB_E2E=1 to run external services")
def test_baswe_pdf_to_pgvector_query() -> None:
    pdf = Path(os.getenv("RAGLAB_PDF", "BASWE-15-RAG-Concepts-Guide.pdf"))
    dsn = os.getenv("RAGLAB_DSN", "postgresql://raglab:raglab@127.0.0.1:5432/raglab")
    repository = PostgresRepository(dsn)
    repository.migrate()
    collection = f"baswe-e2e-{uuid4().hex}"
    report = ingest(SourceInput.path(pdf), CollectionConfig(collection), dsn=dsn)
    query = OllamaEmbeddingProvider().embed_documents(["RAG ingestion principles"])[0]
    assert report.chunk_count > 0
    assert repository.search(collection, query)
