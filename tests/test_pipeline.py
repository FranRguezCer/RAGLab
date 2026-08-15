from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from raglab import CollectionConfig, LineProvenance, ProvenanceStatus, SourceInput
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
        if self.calls and self.calls[-1][3] == fingerprint:
            return "document-id"
        return None

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
    assert repository.calls[0][1].title == "Title"
    assert first.provenance_status is ProvenanceStatus.COMPLETE
    assert first.provenance_warnings == ()


def test_pipeline_reports_and_persists_degraded_provenance() -> None:
    repository = FakeRepository()
    embeddings = FakeEmbeddings()
    converted = Converter().convert(SourceInput.text("# Title\n\nUseful content."))
    converted = replace(
        converted,
        provenance_status=ProvenanceStatus.PARTIAL,
        provenance_warnings=("Page mapping failed.",),
    )

    class StaticConverter:
        def convert(self, source: SourceInput, *, use_jina: bool = False):
            return converted

    pipeline = IngestionPipeline(
        converter=StaticConverter(),
        parser=MarkdownParser(),
        chunker=SemanticChunker(
            embedding_provider=embeddings,
            config=ChunkingConfig(target_tokens=8, min_tokens=2, max_tokens=20),
        ),
        embeddings=embeddings,
        repository=repository,
    )

    report = pipeline.ingest(SourceInput.text("ignored"), CollectionConfig("test", dimension=2))

    assert report.provenance_status is ProvenanceStatus.PARTIAL
    assert report.provenance_warnings == ("Page mapping failed.",)
    stored_document = repository.calls[0][1]
    assert stored_document.provenance_status is ProvenanceStatus.PARTIAL
    assert stored_document.provenance_warnings == report.provenance_warnings


def test_citable_provenance_changes_prevent_idempotent_skip() -> None:
    repository = FakeRepository()
    embeddings = FakeEmbeddings()
    source = SourceInput.text("# Title\n\nUseful content.")
    converted = Converter().convert(source)
    partial = replace(
        converted,
        source_name="guide.pdf",
        media_type="application/pdf",
        title="Converter title",
        line_provenance=(
            LineProvenance(1),
            LineProvenance(2),
            LineProvenance(3),
        ),
        provenance_status=ProvenanceStatus.PARTIAL,
        provenance_warnings=("Pages unavailable.", "Source lines unavailable."),
    )
    complete = replace(
        partial,
        line_provenance=(
            LineProvenance(1, page_number=1),
            LineProvenance(2, page_number=1),
            LineProvenance(3, page_number=2),
        ),
        provenance_status=ProvenanceStatus.COMPLETE,
        provenance_warnings=(),
    )

    class SequenceConverter:
        def __init__(self) -> None:
            self.documents = [partial, complete]

        def convert(self, incoming_source: SourceInput, *, use_jina: bool = False):
            return self.documents.pop(0)

    pipeline = IngestionPipeline(
        converter=SequenceConverter(),
        parser=MarkdownParser(),
        chunker=SemanticChunker(
            embedding_provider=embeddings,
            config=ChunkingConfig(target_tokens=8, min_tokens=2, max_tokens=20),
        ),
        embeddings=embeddings,
        repository=repository,
    )
    collection = CollectionConfig("test", dimension=2)

    partial_fingerprint = pipeline._fingerprint(partial, collection)
    assert pipeline._fingerprint(complete, collection) != partial_fingerprint
    assert pipeline._fingerprint(
        replace(partial, source_name="renamed.pdf"), collection
    ) != partial_fingerprint
    assert pipeline._fingerprint(
        replace(partial, media_type="application/octet-stream"), collection
    ) != partial_fingerprint
    assert pipeline._fingerprint(
        replace(partial, title="Revised converter title"), collection
    ) != partial_fingerprint
    normalized = replace(
        partial,
        line_provenance=tuple(reversed(partial.line_provenance)),
        provenance_warnings=tuple(reversed(partial.provenance_warnings)),
    )
    assert pipeline._fingerprint(normalized, collection) == partial_fingerprint

    first = pipeline.ingest(source, collection)
    second = pipeline.ingest(source, collection)

    assert first.status == "indexed"
    assert second.status == "indexed"
    assert first.fingerprint != second.fingerprint
    assert len(repository.calls) == 2
