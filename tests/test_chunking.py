from __future__ import annotations

from collections.abc import Sequence

from raglab.chunking import ChunkingConfig, SemanticChunker, SimpleTokenCounter
from raglab.contracts import BlockKind, MarkdownBlock, ParsedMarkdown


class TopicEmbeddings:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


def parsed(*blocks: MarkdownBlock) -> ParsedMarkdown:
    return ParsedMarkdown("memory://test", "Guide", blocks, "")


def block(text: str, path: tuple[str, ...] = ("Section",)) -> MarkdownBlock:
    return MarkdownBlock(BlockKind.PARAGRAPH, text, 1, 1, path)


def test_semantic_boundary_refines_structure_and_keeps_content_faithful() -> None:
    chunker = SemanticChunker(
        token_counter=SimpleTokenCounter(),
        embedding_provider=TopicEmbeddings(),
        config=ChunkingConfig(target_tokens=8, min_tokens=2, max_tokens=20, semantic_percentile=90),
    )
    chunks = chunker.chunk(parsed(block("alpha one"), block("alpha two"), block("beta topic")))

    assert len(chunks) == 2
    assert chunks[0].content == "alpha one\n\nalpha two"
    assert chunks[1].content == "beta topic"
    assert not chunks[0].content.startswith("Guide")
    assert chunks[0].embedding_text.startswith("Guide > Section")
    assert chunker.last_distances == (0.0, 1.0)


def test_large_table_repeats_header_and_respects_maximum() -> None:
    table = "| A | B |\n|---|---|\n" + "\n".join(f"| row{i} | value{i} |" for i in range(8))
    table_block = MarkdownBlock(BlockKind.TABLE, table, 1, 10, ("Data",))
    chunker = SemanticChunker(config=ChunkingConfig(target_tokens=10, min_tokens=2, max_tokens=20))
    chunks = chunker.chunk(parsed(table_block))
    assert len(chunks) > 1
    assert all(chunk.content.startswith("| A | B |\n|---|---|") for chunk in chunks)
    assert all(chunk.token_count <= 20 for chunk in chunks)


def test_large_code_block_keeps_valid_fences() -> None:
    code = "```python\n" + " ".join(f"value_{i}" for i in range(30)) + "\n```"
    code_block = MarkdownBlock(BlockKind.CODE, code, 1, 3, ("Code",))
    chunker = SemanticChunker(config=ChunkingConfig(target_tokens=8, min_tokens=2, max_tokens=12))
    chunks = chunker.chunk(parsed(code_block))
    assert len(chunks) > 1
    assert all(chunk.content.startswith("```python\n") for chunk in chunks)
    assert all(chunk.content.endswith("\n```") for chunk in chunks)
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_invalid_configuration_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        ChunkingConfig(target_tokens=10, min_tokens=20, max_tokens=30)
