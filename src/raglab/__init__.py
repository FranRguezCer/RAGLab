"""Public contracts and entry points for RAGLab."""

from raglab.contracts import (
    Chunk,
    Citation,
    CollectionConfig,
    ConvertedDocument,
    EmbeddedChunk,
    IngestionReport,
    LineProvenance,
    MarkdownBlock,
    ParsedMarkdown,
    ProvenanceStatus,
    SourceInput,
)
from raglab.pipeline import IngestionPipeline, ingest

__all__ = [
    "Citation",
    "Chunk",
    "CollectionConfig",
    "ConvertedDocument",
    "EmbeddedChunk",
    "IngestionPipeline",
    "IngestionReport",
    "LineProvenance",
    "MarkdownBlock",
    "ParsedMarkdown",
    "ProvenanceStatus",
    "SourceInput",
    "ingest",
]
