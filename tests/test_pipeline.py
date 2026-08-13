from __future__ import annotations

from collections.abc import Sequence

from raglab import CollectionConfig, SourceInput
from raglab.chunking import ChunkingConfig, SemanticChunker
from raglab.conversion import Converter
from raglab.parsing import MarkdownParser
from raglab.pipeline import IngestionPipeline


class FakeEmbeddings:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeRepository:
    def __init__(self) -> None:
        self.calls = []

    def current_document_id(self, config, source_uri, content_hash, fingerprint):
        return "document-id" if self.calls else None

    def store(self, config, document, chunks, fingerprint):
        self.calls.append((config, document, chunks, fingerprint))
        return "document-id", False


def test_pipeline_embeds_before_storage_and_reports_idempotency() -> None:
    repository = FakeRepository()
    embeddings = FakeEmbeddings()
    pipeline = IngestionPipeline(
        converter=Converter(),
        parser=MarkdownParser(),
        chunker=SemanticChunker(
            embedding_provider=embeddings,
            config=ChunkingConfig(target_tokens=8, min_tokens=2, max_tokens=20),
        ),
        embeddings=embeddings,
        repository=repository,
    )
    source = SourceInput.text("# Title\n\nUseful content.")
    collection = CollectionConfig("test", dimension=2)

    first = pipeline.ingest(source, collection)
    second = pipeline.ingest(source, collection)

    assert first.status == "indexed"
    assert first.chunk_count == 1
    assert second.status == "skipped"
    assert second.chunk_count == 0
    assert first.fingerprint == second.fingerprint
    assert len(repository.calls[0][2][0].embedding) == 2
