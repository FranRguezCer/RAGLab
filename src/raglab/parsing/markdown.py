from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from raglab.contracts import BlockKind, ConvertedDocument, MarkdownBlock, ParsedMarkdown
from raglab.errors import ParsingError

_OPEN_BLOCKS = {
    "bullet_list_open": BlockKind.LIST,
    "ordered_list_open": BlockKind.LIST,
    "table_open": BlockKind.TABLE,
}


class MarkdownParser:
    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark", {"html": True}).enable("table")

    def parse(self, document: ConvertedDocument) -> ParsedMarkdown:
        try:
            tokens = self._parser.parse(document.markdown)
        except Exception as exc:
            raise ParsingError(
                f"Could not parse Markdown from {document.source_uri}: {exc}"
            ) from exc
        lines = document.markdown.splitlines(keepends=True)
        blocks: list[MarkdownBlock] = []
        headings: list[str] = []
        title: str | None = None
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type == "heading_open":
                level = int(token.tag[1:])
                text = self._inline_content(tokens, index + 1)
                headings[level - 1 :] = [text]
                if level == 1 and title is None:
                    title = text
                blocks.append(self._block(BlockKind.HEADING, token, lines, tuple(headings)))
                index = self._matching_close(tokens, index)
            elif token.type in _OPEN_BLOCKS:
                blocks.append(self._block(_OPEN_BLOCKS[token.type], token, lines, tuple(headings)))
                index = self._matching_close(tokens, index)
            elif token.type == "paragraph_open" and token.level == 0:
                blocks.append(self._block(BlockKind.PARAGRAPH, token, lines, tuple(headings)))
                index = self._matching_close(tokens, index)
            elif token.type in {"fence", "code_block"}:
                blocks.append(self._block(BlockKind.CODE, token, lines, tuple(headings)))
            elif token.map is not None and token.level == 0 and token.nesting == 0:
                blocks.append(self._block(BlockKind.OTHER, token, lines, tuple(headings)))
            index += 1
        return ParsedMarkdown(
            source_uri=document.source_uri,
            title=title,
            blocks=tuple(self._deduplicate(blocks)),
            markdown=document.markdown,
        )

    @staticmethod
    def _inline_content(tokens: list[Token], index: int) -> str:
        return tokens[index].content.strip() if index < len(tokens) else ""

    @staticmethod
    def _matching_close(tokens: list[Token], start: int) -> int:
        opening = tokens[start]
        if opening.nesting != 1:
            return start
        depth = 1
        for index in range(start + 1, len(tokens)):
            if tokens[index].type == opening.type:
                depth += tokens[index].nesting
            elif tokens[index].type == opening.type.replace("_open", "_close"):
                depth -= 1
                if depth == 0:
                    return index
        return start

    @staticmethod
    def _block(
        kind: BlockKind,
        token: Token,
        lines: list[str],
        headings: tuple[str, ...],
    ) -> MarkdownBlock:
        if token.map is None:
            content, start, end = token.content, None, None
        else:
            start, end_exclusive = token.map
            content = "".join(lines[start:end_exclusive]).rstrip("\n")
            end = max(start, end_exclusive - 1) + 1
            start += 1
        return MarkdownBlock(
            kind=kind,
            content=content,
            start_line=start,
            end_line=end,
            heading_path=headings,
            metadata={"markup": token.markup, "info": token.info},
        )

    @staticmethod
    def _deduplicate(blocks: list[MarkdownBlock]) -> list[MarkdownBlock]:
        result: list[MarkdownBlock] = []
        seen: set[tuple[int | None, int | None, str]] = set()
        for block in blocks:
            key = (block.start_line, block.end_line, block.content)
            if block.content.strip() and key not in seen:
                result.append(block)
                seen.add(key)
        return result


def parse(document: ConvertedDocument) -> ParsedMarkdown:
    return MarkdownParser().parse(document)
