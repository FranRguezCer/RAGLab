from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from raglab.chunking import (
    ChunkingConfig,
    SemanticChunker,
    SimpleTokenCounter,
    TransformersTokenCounter,
)
from raglab.contracts import BlockKind, MarkdownBlock, ParsedMarkdown


class TopicEmbeddings:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


def fake_tokenizer_modules(
    *, snapshot_download: Mock, from_pretrained: Mock
) -> dict[str, ModuleType]:
    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=from_pretrained
    )
    return {"huggingface_hub": huggingface_hub, "transformers": transformers}


def parsed(*blocks: MarkdownBlock) -> ParsedMarkdown:
    return ParsedMarkdown("memory://test", "Guide", blocks, "")


def block(text: str, path: tuple[str, ...] = ("Section",)) -> MarkdownBlock:
    return MarkdownBlock(BlockKind.PARAGRAPH, text, 1, 1, path)


def test_transformers_counter_loads_repository_id_from_local_snapshot() -> None:
    snapshot_download = Mock(return_value="/cache/snapshots/revision")
    tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2]))
    from_pretrained = Mock(return_value=tokenizer)

    with patch.dict(
        sys.modules,
        fake_tokenizer_modules(
            snapshot_download=snapshot_download,
            from_pretrained=from_pretrained,
        ),
    ):
        counter = TransformersTokenCounter("owner/model")

    assert counter.count("hello") == 2
    snapshot_download.assert_called_once_with(repo_id="owner/model", local_files_only=True)
    from_pretrained.assert_called_once_with(
        "/cache/snapshots/revision", local_files_only=True
    )
    tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)


def test_transformers_counter_preserves_existing_local_path(tmp_path: Path) -> None:
    snapshot_download = Mock()
    from_pretrained = Mock(return_value=SimpleNamespace(encode=Mock(return_value=[])))
    model_path = tmp_path / "tokenizer"
    model_path.mkdir()

    with patch.dict(
        sys.modules,
        fake_tokenizer_modules(
            snapshot_download=snapshot_download,
            from_pretrained=from_pretrained,
        ),
    ):
        TransformersTokenCounter(str(model_path))

    snapshot_download.assert_not_called()
    from_pretrained.assert_called_once_with(str(model_path), local_files_only=True)


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


def test_structural_mode_does_not_create_synthetic_semantic_boundaries() -> None:
    chunker = SemanticChunker(
        token_counter=SimpleTokenCounter(),
        config=ChunkingConfig(target_tokens=20, min_tokens=2, max_tokens=30),
    )

    chunks = chunker.chunk(parsed(block("first paragraph"), block("second paragraph")))

    assert len(chunks) == 1
    assert chunker.last_threshold == float("inf")


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
