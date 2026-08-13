from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from raglab.chunking import ChunkingConfig, SemanticChunker, TransformersTokenCounter
from raglab.contracts import (
    CollectionConfig,
    ConvertedDocument,
    EmbeddedChunk,
    IngestionReport,
    SourceInput,
)
from raglab.conversion import Converter
from raglab.embeddings import OllamaEmbeddingProvider
from raglab.parsing import MarkdownParser
from raglab.storage import PostgresRepository


class Embeddings(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class Repository(Protocol):
    def current_document_id(
        self,
        config: CollectionConfig,
        source_uri: str,
        content_hash: str,
        fingerprint: str,
    ) -> str | None: ...

    def store(
        self,
        config: CollectionConfig,
        document: ConvertedDocument,
        chunks: Sequence[EmbeddedChunk],
        fingerprint: str,
    ) -> tuple[str, bool]: ...


class IngestionPipeline:
    """Coordinates explicit stages; embeddings complete before storage opens a transaction."""

    def __init__(
        self,
        *,
        converter: Converter,
        parser: MarkdownParser,
        chunker: SemanticChunker,
        embeddings: Embeddings,
        repository: Repository,
    ) -> None:
        self.converter = converter
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.repository = repository

    def ingest(
        self,
        source: SourceInput,
        collection: CollectionConfig,
        *,
        use_jina: bool = False,
    ) -> IngestionReport:
        converted = self.converter.convert(source, use_jina=use_jina)
        fingerprint = self._fingerprint(converted, collection)
        current_id = self.repository.current_document_id(
            collection,
            converted.source_uri,
            converted.content_hash,
            fingerprint,
        )
        if current_id is not None:
            return IngestionReport(
                source_uri=source.uri,
                collection=collection.name,
                status="skipped",
                document_id=current_id,
                chunk_count=0,
                content_hash=converted.content_hash,
                fingerprint=fingerprint,
            )
        parsed = self.parser.parse(converted)
        chunks = self.chunker.chunk(parsed)
        vectors = self.embeddings.embed_documents([item.embedding_text for item in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding provider returned a different number of vectors")
        embedded = [
            EmbeddedChunk(chunk=item, embedding=tuple(vector))
            for item, vector in zip(chunks, vectors, strict=True)
        ]
        document_id, skipped = self.repository.store(collection, converted, embedded, fingerprint)
        return IngestionReport(
            source_uri=source.uri,
            collection=collection.name,
            status="skipped" if skipped else "indexed",
            document_id=document_id,
            chunk_count=0 if skipped else len(chunks),
            content_hash=converted.content_hash,
            fingerprint=fingerprint,
        )

    def _fingerprint(self, document: ConvertedDocument, collection: CollectionConfig) -> str:
        payload = {
            "pipeline": 1,
            "converter": [document.converter, document.converter_version],
            "model": collection.model,
            "dimension": collection.dimension,
            "metric": collection.metric,
            "collection_chunk_config": dict(collection.chunk_config),
            "chunker": {
                "target_tokens": self.chunker.config.target_tokens,
                "min_tokens": self.chunker.config.min_tokens,
                "max_tokens": self.chunker.config.max_tokens,
                "semantic_percentile": self.chunker.config.semantic_percentile,
                "overlap_tokens": self.chunker.config.overlap_tokens,
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()


def ingest(
    source: SourceInput,
    collection: CollectionConfig,
    *,
    dsn: str,
    use_jina: bool = False,
    chunk_config: ChunkingConfig | None = None,
) -> IngestionReport:
    embeddings = OllamaEmbeddingProvider(model=collection.model, dimension=collection.dimension)
    pipeline = IngestionPipeline(
        converter=Converter(),
        parser=MarkdownParser(),
        chunker=SemanticChunker(
            token_counter=TransformersTokenCounter(),
            embedding_provider=embeddings,
            config=chunk_config,
        ),
        embeddings=embeddings,
        repository=PostgresRepository(dsn),
    )
    return pipeline.ingest(source, collection, use_jina=use_jina)
