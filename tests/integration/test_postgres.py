from __future__ import annotations

import os
from uuid import uuid4

import pytest

from raglab import Chunk, CollectionConfig, ConvertedDocument, EmbeddedChunk
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
    document = ConvertedDocument(source_uri, "hello", "hash", "test", "1")
    vector = tuple([1.0] + [0.0] * 1023)
    chunk = Chunk(0, "hello", "hello", 1, (), 1, 1)
    first = repository.store(config, document, [EmbeddedChunk(chunk, vector)], "fingerprint")
    second = repository.store(config, document, [EmbeddedChunk(chunk, vector)], "fingerprint")
    assert first[1] is False
    assert second[1] is True
    assert repository.search(config.name, vector, exact=True)[0].content == "hello"
    assert repository.search(config.name, vector)[0].content == "hello"
    assert repository.collection_stats(config.name) == {
        "name": config.name,
        "model": config.model,
        "dimension": 1024,
        "document_count": 1,
        "chunk_count": 1,
        "token_count": 1,
    }

    changed = ConvertedDocument(source_uri, "changed", "new-hash", "test", "1")
    invalid = Chunk(0, "invalid", "invalid", 0, (), 1, 1)
    with pytest.raises(StorageError):
        repository.store(config, changed, [EmbeddedChunk(invalid, vector)], "new-fingerprint")
    assert repository.search(config.name, vector, exact=True)[0].content == "hello"
