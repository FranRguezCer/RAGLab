from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from raglab.contracts import BlockKind, Chunk, MarkdownBlock, ParsedMarkdown


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class SimpleTokenCounter:
    """Deterministic fallback for tests; production can use the Qwen tokenizer adapter."""

    _tokens = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count(self, text: str) -> int:
        return len(self._tokens.findall(text))


class TransformersTokenCounter:
    """Lazy local tokenizer for the embedding model; it never calls an inference API."""

    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B") -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install `raglab[tokenizers]` to use the Qwen tokenizer") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model, local_files_only=True
        )

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 512
    min_tokens: int = 120
    max_tokens: int = 768
    semantic_percentile: float = 90.0
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Expected 0 < min_tokens <= target_tokens <= max_tokens")
        if not 0 <= self.semantic_percentile <= 100:
            raise ValueError("semantic_percentile must be between 0 and 100")
        if self.overlap_tokens != 0:
            raise ValueError("Only zero overlap is supported; structure supplies the context")


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    heading_path: tuple[str, ...]
    start_line: int | None
    end_line: int | None


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    lnorm = math.sqrt(sum(value * value for value in left))
    rnorm = math.sqrt(sum(value * value for value in right))
    if lnorm == 0 or rnorm == 0:
        return 1.0
    return 1.0 - dot / (lnorm * rnorm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


class SemanticChunker:
    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.tokens = token_counter or SimpleTokenCounter()
        self.embeddings = embedding_provider
        self.config = config or ChunkingConfig()
        self.last_distances: tuple[float, ...] = ()
        self.last_threshold: float = math.inf

    def chunk(self, document: ParsedMarkdown) -> list[Chunk]:
        units = [unit for block in document.blocks for unit in self._split_block(block)]
        if not units:
            return []
        distances = self._distances(units)
        threshold = _percentile(distances, self.config.semantic_percentile)
        self.last_distances = tuple(distances)
        self.last_threshold = threshold
        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        for index, unit in enumerate(units):
            if current and self._must_break(current, unit, distances, index, threshold):
                groups.append(current)
                current = []
            current.append(unit)
        if current:
            groups.append(current)
        groups = self._merge_small(groups)
        return [self._to_chunk(index, group, document.title) for index, group in enumerate(groups)]

    def _split_block(self, block: MarkdownBlock) -> list[_Unit]:
        if block.kind is BlockKind.HEADING:
            return []
        if self.tokens.count(block.content) <= self.config.max_tokens:
            return [self._unit(block.content, block)]
        if block.kind is BlockKind.TABLE:
            parts = self._split_table(block.content)
        elif block.kind is BlockKind.LIST:
            parts = self._split_list(block.content)
        elif block.kind is BlockKind.CODE:
            parts = self._split_code(block.content)
        else:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])", block.content)
        return [self._unit(text, block) for text in self._pack(parts) if text.strip()]

    @staticmethod
    def _unit(text: str, block: MarkdownBlock) -> _Unit:
        return _Unit(text.strip(), block.heading_path, block.start_line, block.end_line)

    @staticmethod
    def _split_list(text: str) -> list[str]:
        starts = list(re.finditer(r"(?m)^(?:\s*[-+*]|\s*\d+[.)])\s+", text))
        if len(starts) < 2:
            return text.splitlines()
        items = [
            text[item.start() : starts[index + 1].start()] for index, item in enumerate(starts[:-1])
        ]
        return items + [text[starts[-1].start() :]]

    @staticmethod
    def _split_table(text: str) -> list[str]:
        rows = text.splitlines()
        if len(rows) <= 2:
            return rows
        header = "\n".join(rows[:2])
        return [f"{header}\n{row}" for row in rows[2:]]

    def _split_code(self, text: str) -> list[str]:
        lines = text.splitlines()
        if len(lines) < 3 or not lines[0].lstrip().startswith("```"):
            return lines
        opening = lines[0]
        body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        overhead = self.tokens.count(f"{opening}\n```")
        budget = max(1, self.config.max_tokens - overhead)
        segments = [segment for line in body for segment in self._hard_split(line, budget)]
        return [f"{opening}\n{segment}\n```" for segment in segments]

    def _hard_split(self, text: str, limit: int) -> list[str]:
        result: list[str] = []
        remaining = text.strip()
        while remaining and self.tokens.count(remaining) > limit:
            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if self.tokens.count(remaining[:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            cut = low
            whitespace = remaining.rfind(" ", 0, cut + 1)
            if whitespace > 0:
                cut = whitespace
            result.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            result.append(remaining)
        return result

    def _pack(self, parts: Sequence[str]) -> list[str]:
        packed: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}\n{part}".strip()
            if current and self.tokens.count(candidate) > self.config.max_tokens:
                packed.append(current)
                current = part.strip()
            else:
                current = candidate
            # Last-resort split for a sentence/line longer than max.
            while self.tokens.count(current) > self.config.max_tokens:
                pieces = self._hard_split(current, self.config.max_tokens)
                packed.extend(pieces[:-1])
                current = pieces[-1]
        if current:
            packed.append(current)
        return packed

    def _distances(self, units: Sequence[_Unit]) -> list[float]:
        if self.embeddings is None or len(units) < 2:
            return [0.0] * max(0, len(units) - 1)
        vectors = self.embeddings.embed_documents([unit.text for unit in units])
        if len(vectors) != len(units):
            raise ValueError("Embedding provider returned a different number of vectors")
        return [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]

    def _must_break(
        self,
        current: Sequence[_Unit],
        unit: _Unit,
        distances: Sequence[float],
        index: int,
        threshold: float,
    ) -> bool:
        candidate = "\n\n".join([item.text for item in current] + [unit.text])
        if self.tokens.count(candidate) > self.config.max_tokens:
            return True
        current_tokens = self.tokens.count("\n\n".join(item.text for item in current))
        heading_changed = current[-1].heading_path != unit.heading_path
        semantic_break = index > 0 and distances[index - 1] >= threshold
        return current_tokens >= self.config.min_tokens and (
            heading_changed or current_tokens >= self.config.target_tokens or semantic_break
        )

    def _merge_small(self, groups: list[list[_Unit]]) -> list[list[_Unit]]:
        if len(groups) < 2:
            return groups
        merged: list[list[_Unit]] = []
        for group in groups:
            size = self.tokens.count("\n\n".join(unit.text for unit in group))
            if merged and size < self.config.min_tokens:
                candidate = merged[-1] + group
                text = "\n\n".join(unit.text for unit in candidate)
                if self.tokens.count(text) <= self.config.max_tokens:
                    merged[-1] = candidate
                    continue
            merged.append(group)
        return merged

    def _to_chunk(self, index: int, units: Sequence[_Unit], title: str | None) -> Chunk:
        content = "\n\n".join(unit.text for unit in units)
        path = units[0].heading_path
        context_parts = ((title,) if title else ()) + path
        deduplicated = tuple(
            item
            for index, item in enumerate(context_parts)
            if item and item not in context_parts[:index]
        )
        context = " > ".join(deduplicated)
        embedding_text = f"{context}\n\n{content}" if context else content
        return Chunk(
            index=index,
            content=content,
            embedding_text=embedding_text,
            token_count=self.tokens.count(content),
            heading_path=path,
            start_line=min((unit.start_line for unit in units if unit.start_line), default=None),
            end_line=max((unit.end_line for unit in units if unit.end_line), default=None),
        )


def chunk(
    document: ParsedMarkdown,
    *,
    token_counter: TokenCounter | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    return SemanticChunker(
        token_counter=token_counter,
        embedding_provider=embedding_provider,
        config=config,
    ).chunk(document)
