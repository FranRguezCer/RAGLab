from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from raglab.contracts import SourceInput
from raglab.conversion import Converter
from raglab.errors import EmptySourceError, UnsafeRemoteURLError


def test_direct_markdown_is_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "source.md"
    path.write_text("# Exact\n\nKeep  spaces.", encoding="utf-8")

    result = Converter().convert(SourceInput.path(path, owner="test"))

    assert result.markdown == "# Exact\n\nKeep  spaces."
    assert result.converter == "direct"
    assert result.metadata["owner"] == "test"
    assert result.line_provenance[-1].endswith("#L3")


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


def test_docling_is_lazy_and_injectable(tmp_path: Path) -> None:
    class Document:
        def export_to_markdown(self) -> str:
            return "# Converted"

    class Result:
        document = Document()

    class FakeDocling:
        def convert(self, path: Path) -> Result:
            assert path.suffix == ".pdf"
            return Result()

    path = tmp_path / "source.pdf"
    path.write_bytes(b"fake")
    result = Converter(docling_factory=FakeDocling).convert(SourceInput.path(path))
    assert result.converter == "docling"
    assert result.markdown == "# Converted"
