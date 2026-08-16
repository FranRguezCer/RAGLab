from __future__ import annotations

import os
from collections.abc import Sequence
from uuid import uuid4

import pytest

from raglab import Chunk, CollectionConfig, ConvertedDocument, EmbeddedChunk
from raglab.retrieval import (
    MetadataFilter,
    PostgresRetrievalRepository,
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalRequest,
)
from raglab.storage import PostgresRepository

pytestmark = pytest.mark.integration


class _QueryEmbeddings:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]


def _document(uri: str, tenant: str) -> ConvertedDocument:
    return ConvertedDocument(
        source_uri=uri,
        markdown="# Fault E17\n\nCause and recovery.",
        content_hash=uuid4().hex,
        converter="test",
        converter_version="1",
        source_name=f"{tenant}.md",
        title="Fault guide",
        metadata={"tenant_id": tenant},
    )


@pytest.mark.skipif(not os.getenv("RAGLAB_TEST_DSN"), reason="RAGLAB_TEST_DSN is not set")
def test_hybrid_channels_and_parent_expansion_share_tenant_prefilters() -> None:
    dsn = os.environ["RAGLAB_TEST_DSN"]
    storage = PostgresRepository(dsn)
    storage.migrate()
    collection = CollectionConfig(name=f"retrieval-integration-{uuid4().hex}")
    primary = tuple([1.0] + [0.0] * 1023)
    secondary = tuple([0.9, 0.1] + [0.0] * 1022)

    acme_id, _ = storage.store(
        collection,
        _document("memory://acme/e17", "acme"),
        [
            EmbeddedChunk(
                Chunk(
                    0,
                    "Fault E17 is caused by low coolant pressure.",
                    "Fault E17",
                    9,
                    ("Fault E17",),
                    1,
                    1,
                ),
                primary,
            ),
            EmbeddedChunk(
                Chunk(
                    1,
                    "Restore pressure, then reset the controller.",
                    "Fault E17 reset",
                    8,
                    ("Fault E17",),
                    2,
                    2,
                ),
                secondary,
            ),
        ],
        uuid4().hex,
    )
    beta_id, _ = storage.store(
        collection,
        _document("memory://beta/e17", "beta"),
        [
            EmbeddedChunk(
                Chunk(
                    0,
                    "Fault E17 is a tenant-specific calibration warning.",
                    "Fault E17",
                    8,
                    ("Fault E17",),
                    1,
                    1,
                ),
                primary,
            )
        ],
        uuid4().hex,
    )

    repository = PostgresRetrievalRepository(dsn)
    filters = (MetadataFilter("document.metadata.tenant_id", "eq", "acme"),)
    semantic = repository.semantic_search(
        collection.name, primary, filters, limit=50, exact=True, ef_search=100
    )
    lexical = repository.lexical_search(collection.name, "Fault E17", filters, limit=50)
    assert semantic and {item.document_id for item in semantic} == {acme_id}
    assert lexical and {item.document_id for item in lexical} == {acme_id}
    assert all(item.matches_filters for item in repository.document_chunks(acme_id, filters))
    assert not any(item.matches_filters for item in repository.document_chunks(beta_id, filters))

    response = RetrievalPipeline(repository, _QueryEmbeddings(primary)).retrieve(
        RetrievalRequest(
            "What causes fault E17?",
            collection=collection.name,
            filters=filters,
            config=RetrievalConfig(rerank=False, mmr=False),
        )
    )
    assert response.results
    assert {result.document_id for result in response.results} == {acme_id}
    assert "Restore pressure" in response.results[0].content
