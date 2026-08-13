"""Small immutable contracts shared by pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SourceKind(StrEnum):
    PATH = "path"
    URL = "url"
    TEXT = "text"


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class SourceInput:
    uri: str
    kind: SourceKind = SourceKind.PATH
    content: str | bytes | None = None
    allow_remote_service: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def path(cls, path: str | Path, **metadata: Any) -> SourceInput:
        return cls(uri=str(Path(path)), kind=SourceKind.PATH, metadata=metadata)

    @classmethod
    def url(cls, url: str, *, allow_remote_service: bool = False, **metadata: Any) -> SourceInput:
        return cls(
            uri=url,
            kind=SourceKind.URL,
            allow_remote_service=allow_remote_service,
            metadata=metadata,
        )

    @classmethod
    def text(cls, content: str, *, uri: str = "memory://document", **metadata: Any) -> SourceInput:
        return cls(uri=uri, kind=SourceKind.TEXT, content=content, metadata=metadata)


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    source_uri: str
    markdown: str
    content_hash: str
    converter: str
    converter_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    line_provenance: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    kind: BlockKind
    content: str
    start_line: int | None
    end_line: int | None
    heading_path: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedMarkdown:
    source_uri: str
    title: str | None
    blocks: tuple[MarkdownBlock, ...]
    markdown: str


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    content: str
    embedding_text: str
    token_count: int
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    name: str
    model: str = "qwen3-embedding:0.6b"
    dimension: int = 1024
    metric: str = "cosine"
    chunk_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    source_uri: str
    collection: str
    status: str
    document_id: str | None
    chunk_count: int
    content_hash: str
    fingerprint: str
