from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from raglab.contracts import LineProvenance, ProvenanceStatus, SourceInput
from raglab.conversion import Converter
from raglab.conversion.service import _DownloadedResource
from raglab.errors import EmptySourceError, UnsafeRemoteURLError


def test_direct_markdown_preserves_content_and_structured_lines(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text("# Exact\n\nKeep  spaces.", encoding="utf-8")

    result = Converter().convert(SourceInput.path(path, owner="test"))

    assert result.markdown == "# Exact\n\nKeep  spaces."
    assert result.converter == "direct"
    assert result.source_name == "source.md"
    assert result.media_type == "text/markdown"
    assert result.title is None
    assert result.metadata["owner"] == "test"
    assert result.line_provenance == (
        LineProvenance(markdown_line=1, source_line=1),
        LineProvenance(markdown_line=2, source_line=2),
        LineProvenance(markdown_line=3, source_line=3),
    )
    assert result.provenance_status is ProvenanceStatus.COMPLETE
    assert result.provenance_warnings == ()


def test_empty_source_is_rejected() -> None:
    with pytest.raises(EmptySourceError):
        Converter().convert(SourceInput.text("  \n"))


@pytest.mark.parametrize(
    "source",
    [
        SourceInput.path("private.pdf"),
        SourceInput.url("http://127.0.0.1/admin", allow_remote_service=True),
        SourceInput.url("https://example.org", allow_remote_service=False),
    ],
)
def test_jina_requires_explicit_public_url(source: SourceInput) -> None:
    address = [(2, 1, 6, "", ("127.0.0.1", 0))]
    with (
        patch("raglab.conversion.service.socket.getaddrinfo", return_value=address),
        pytest.raises(UnsafeRemoteURLError),
    ):
        Converter().convert(source, use_jina=True)


@pytest.mark.parametrize(
    "url",
    ["https://user@example.org/document", "https://:secret@example.org/document"],
)
def test_jina_rejects_urls_with_credentials(url: str) -> None:
    with pytest.raises(UnsafeRemoteURLError):
        Converter().convert(
            SourceInput.url(url, allow_remote_service=True),
            use_jina=True,
        )


@pytest.mark.parametrize("suffix", [".pdf", ".html", ".docx", ".odt", ".ods", ".odp"])
def test_supported_document_extensions_route_to_docling(tmp_path: Path, suffix: str) -> None:
    converted_suffixes: list[str] = []

    class Document:
        pages: dict[int, object] = {}
        texts: tuple[object, ...] = ()

        def export_to_markdown(self) -> str:
            return "# Converted"

    class Result:
        document = Document()

    class FakeDocling:
        def convert(self, path: Path) -> Result:
            converted_suffixes.append(path.suffix)
            return Result()

    path = tmp_path / f"source{suffix}"
    path.write_bytes(b"fake")

    result = Converter(docling_factory=FakeDocling).convert(SourceInput.path(path))

    assert converted_suffixes == [suffix]
    assert result.converter == "docling"
    assert result.source_name == path.name
    assert result.markdown == "# Converted"


def test_public_url_source_name_uses_content_disposition_then_final_path() -> None:
    source = SourceInput.url("https://example.org/download")
    disposition = _DownloadedResource(
        payload=b"# Remote",
        media_type="text/markdown",
        content_disposition_filename="published-guide.md",
        final_url="https://cdn.example.org/files/redirected.md",
    )
    with patch("raglab.conversion.service._read_url", return_value=disposition):
        result = Converter().convert(source)
    assert result.source_name == "published-guide.md"

    final_path = _DownloadedResource(
        payload=b"# Remote",
        media_type="text/markdown",
        content_disposition_filename=None,
        final_url="https://cdn.example.org/files/My%20Guide.md",
    )
    with patch("raglab.conversion.service._read_url", return_value=final_path):
        result = Converter().convert(source)
    assert result.source_name == "My Guide.md"

    hostname = _DownloadedResource(
        payload=b"# Remote",
        media_type="text/markdown",
        content_disposition_filename=None,
        final_url="https://docs.example.org/",
    )
    with patch("raglab.conversion.service._read_url", return_value=hostname):
        result = Converter().convert(source)
    assert result.source_name == "docs.example.org"


def test_docling_maps_canonical_markdown_lines_to_pages(tmp_path: Path) -> None:
    class Title:
        label = "title"
        text = "Converter title"

    class Document:
        pages = {1: object(), 2: object()}
        texts = (Title(),)

        def export_to_markdown(self, *, page_break_placeholder: str | None = None) -> str:
            if page_break_placeholder is None:
                return "# Converted\n\nPage one.\n\nPage two."
            return (
                "# Converted\n\nPage one.\n\n"
                f"{page_break_placeholder}\n\nPage two."
            )

    class Result:
        document = Document()

    class FakeDocling:
        def convert(self, path: Path) -> Result:
            return Result()

    path = tmp_path / "source.pdf"
    path.write_bytes(b"fake")

    result = Converter(docling_factory=FakeDocling).convert(SourceInput.path(path))

    assert result.markdown == "# Converted\n\nPage one.\n\nPage two."
    assert result.title == "Converter title"
    assert result.provenance_status is ProvenanceStatus.COMPLETE
    assert {line.page_number for line in result.line_provenance} == {1, 2}
    page_two_line = next(
        index
        for index, line in enumerate(result.markdown.splitlines(), start=1)
        if "Page two" in line
    )
    assert result.line_provenance[page_two_line - 1].page_number == 2


def test_docling_page_mapping_failure_is_visible_and_non_fatal(tmp_path: Path) -> None:
    class Document:
        pages = {1: object(), 2: object()}
        texts: tuple[object, ...] = ()

        def export_to_markdown(self, *, page_break_placeholder: str | None = None) -> str:
            if page_break_placeholder is not None:
                raise RuntimeError("page markers unavailable")
            return "# Converted\n\nStill usable."

    class Result:
        document = Document()

    class FakeDocling:
        def convert(self, path: Path) -> Result:
            return Result()

    path = tmp_path / "source.pdf"
    path.write_bytes(b"fake")

    result = Converter(docling_factory=FakeDocling).convert(SourceInput.path(path))

    assert result.markdown == "# Converted\n\nStill usable."
    assert result.provenance_status is ProvenanceStatus.PARTIAL
    assert "page provenance mapping failed" in result.provenance_warnings[0].lower()
    assert all(line.page_number is None for line in result.line_provenance)
