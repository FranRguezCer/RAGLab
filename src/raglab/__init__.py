"""Public contracts and entry points for RAGLab."""

from raglab.contracts import (
    Chunk,
    CollectionConfig,
    ConvertedDocument,
    EmbeddedChunk,
    IngestionReport,
    MarkdownBlock,
    ParsedMarkdown,
    SourceInput,
)
from raglab.pipeline import IngestionPipeline, ingest

__all__ = [
    "Chunk",
    "CollectionConfig",
    "ConvertedDocument",
    "EmbeddedChunk",
    "IngestionPipeline",
    "IngestionReport",
    "MarkdownBlock",
    "ParsedMarkdown",
    "SourceInput",
    "ingest",
]
