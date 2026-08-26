import pytest

from raglab.contracts import BlockKind, ConvertedDocument
from raglab.errors import ParsingError
from raglab.parsing import MarkdownParser


def document(markdown: str) -> ConvertedDocument:
    return ConvertedDocument("memory://test", markdown, "hash", "test", "1", "test.md")


def test_parser_preserves_structural_blocks_and_lines() -> None:
    markdown = """# Title

Intro paragraph.

- first
- second

| A | B |
|---|---|
| 1 | 2 |

```python
print('hello')
```
"""
    parsed = MarkdownParser().parse(document(markdown))

    assert parsed.title == "Title"
    assert [block.kind for block in parsed.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST,
        BlockKind.TABLE,
        BlockKind.CODE,
    ]
    assert parsed.blocks[2].content == "- first\n- second"
    assert parsed.blocks[3].content.startswith("| A | B |")
    assert parsed.blocks[4].start_line == 12


def test_nested_paragraphs_are_not_duplicated() -> None:
    parsed = MarkdownParser().parse(document("- first\n  continued\n- second"))
    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].kind is BlockKind.LIST


def test_parser_rejects_markdown_without_nonempty_blocks() -> None:
    with pytest.raises(ParsingError, match="no non-empty blocks"):
        MarkdownParser().parse(document(" \n\t"))


def test_heading_only_markdown_remains_valid_parsed_markdown() -> None:
    parsed = MarkdownParser().parse(document("# Title\n\n## Section"))

    assert parsed.title == "Title"
    assert [block.kind for block in parsed.blocks] == [BlockKind.HEADING, BlockKind.HEADING]


def test_heading_with_body_produces_structural_and_searchable_blocks() -> None:
    parsed = MarkdownParser().parse(document("# Title\n\nUseful body."))

    assert [block.kind for block in parsed.blocks] == [BlockKind.HEADING, BlockKind.PARAGRAPH]
    assert parsed.blocks[1].content == "Useful body."
