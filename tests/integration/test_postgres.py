from __future__ import annotations

import os
from uuid import uuid4

import pytest

from raglab import (
    Chunk,
    CollectionConfig,
    ConvertedDocument,
    EmbeddedChunk,
    LineProvenance,
    ProvenanceStatus,
)
from raglab.errors import StorageError
from raglab.storage import PostgresRepository

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("RAGLAB_TEST_DSN"), reason="RAGLAB_TEST_DSN is not set")
def test_migration_idempotency_view_and_search() -> None:
    repository = PostgresRepository(os.environ["RAGLAB_TEST_DSN"])
    repository.migrate()
    repository.migrate()
    run_id = uuid4().hex
    config = CollectionConfig(name=f"integration-test-{run_id}")
    source_uri = f"memory://integration/{run_id}"
    document = ConvertedDocument(
        source_uri,
        "hello",
        "hash",
        "test",
        "1",
        "guide.pdf",
        "application/pdf",
        "Guide title",
        line_provenance=(LineProvenance(1, page_number=2),),
        provenance_status=ProvenanceStatus.PARTIAL,
        provenance_warnings=("Some source locations were unavailable.",),
    )
    vector = tuple([1.0] + [0.0] * 1023)
    chunk = Chunk(0, "hello", "Guide title\nhello", 1, ("Guide title",), 1, 1)
    first = repository.store(config, document, [EmbeddedChunk(chunk, vector)], "fingerprint")
    second = repository.store(config, document, [EmbeddedChunk(chunk, vector)], "fingerprint")
    assert first[1] is False
    assert second[1] is True
    result = repository.search(config.name, vector, exact=True)[0]
    assert result.content == "hello"
    assert result.citation.source_name == "guide.pdf"
    assert result.citation.title == "Guide title"
    assert result.citation.heading_path == ("Guide title",)
    assert (result.citation.start_page, result.citation.end_page) == (2, 2)
    assert (result.citation.start_line, result.citation.end_line) == (1, 1)
    assert result.citation.provenance_status is ProvenanceStatus.PARTIAL
    assert result.citation.provenance_warnings == ("Some source locations were unavailable.",)
    assert repository.search(config.name, vector)[0].content == "hello"
    assert repository.collection_stats(config.name) == {
        "name": config.name,
        "model": config.model,
        "dimension": 1024,
        "document_count": 1,
        "chunk_count": 1,
        "token_count": 1,
    }

    changed = ConvertedDocument(
        source_uri,
        "changed",
        "new-hash",
        "test",
        "1",
        "guide.pdf",
    )
    invalid = Chunk(0, "invalid", "invalid", 0, (), 1, 1)
    with pytest.raises(StorageError):
        repository.store(config, changed, [EmbeddedChunk(invalid, vector)], "new-fingerprint")
    assert repository.search(config.name, vector, exact=True)[0].content == "hello"
