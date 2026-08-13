from __future__ import annotations

import hashlib
import importlib.metadata
import ipaddress
import mimetypes
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from raglab.contracts import ConvertedDocument, SourceInput, SourceKind
from raglab.errors import ConversionError, EmptySourceError, UnsafeRemoteURLError

_DIRECT_SUFFIXES = {".md", ".markdown", ".txt"}
_USER_AGENT = "raglab/0.1 (+local ingestion)"


def _nonempty(text: str, uri: str) -> str:
    text = text.strip("\ufeff")
    if not text.strip():
        raise EmptySourceError(f"Source produced no content: {uri}")
    return text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _provenance(text: str, uri: str) -> tuple[str, ...]:
    return tuple(f"{uri}#L{line}" for line in range(1, text.count("\n") + 2))


def _read_url(url: str, timeout: float) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read(), response.headers.get_content_type()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ConversionError(f"Could not download {url}: {exc}") from exc


def _is_public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port)}
    except socket.gaierror:
        return False
    if not addresses:
        return False
    return all(ipaddress.ip_address(address).is_global for address in addresses)


class Converter:
    """Dispatch sources without importing heavy optional dependencies at startup."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        docling_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.timeout = timeout
        self._docling_factory = docling_factory

    def convert(self, source: SourceInput, *, use_jina: bool = False) -> ConvertedDocument:
        if use_jina:
            return self._convert_jina(source)
        if source.kind is SourceKind.TEXT:
            return self._direct(source, source.content)
        if source.kind is SourceKind.URL:
            return self._convert_url_locally(source)
        return self._convert_path(source)

    def _direct(self, source: SourceInput, content: str | bytes | None) -> ConvertedDocument:
        if content is None:
            raise EmptySourceError(f"Source has no content: {source.uri}")
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        text = _nonempty(text, source.uri)
        return ConvertedDocument(
            source_uri=source.uri,
            markdown=text,
            content_hash=_hash(text),
            converter="direct",
            converter_version="1",
            metadata=dict(source.metadata),
            line_provenance=_provenance(text, source.uri),
        )

    def _convert_path(self, source: SourceInput) -> ConvertedDocument:
        path = Path(source.uri).expanduser()
        if not path.is_file():
            raise ConversionError(f"Source file does not exist: {path}")
        if path.suffix.lower() in _DIRECT_SUFFIXES:
            try:
                return self._direct(source, path.read_text(encoding="utf-8"))
            except UnicodeDecodeError as exc:
                raise ConversionError(f"Source is not valid UTF-8: {path}") from exc
        return self._docling(source, path)

    def _convert_url_locally(self, source: SourceInput) -> ConvertedDocument:
        parsed = urlparse(source.uri)
        if parsed.scheme not in {"http", "https"}:
            raise ConversionError("Local URL conversion accepts only http and https")
        payload, content_type = _read_url(source.uri, self.timeout)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in _DIRECT_SUFFIXES or content_type in {"text/plain", "text/markdown"}:
            return self._direct(source, payload)

        # Docling needs a path for binary inputs. The suffix preserves format hints.
        import tempfile

        guessed = mimetypes.guess_extension(content_type or "") or suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="raglab-") as directory:
            path = Path(directory, f"download{guessed}")
            path.write_bytes(payload)
            document = self._docling(source, path)
        return document

    def _new_docling_converter(self) -> Any:
        if self._docling_factory is not None:
            return self._docling_factory()
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ConversionError(
                "Docling conversion is unavailable. Install `raglab[conversion]`."
            ) from exc
        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.ocr_options = EasyOcrOptions(lang=["es", "en"])
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def _docling(self, source: SourceInput, path: Path) -> ConvertedDocument:
        try:
            result = self._new_docling_converter().convert(path)
            markdown = _nonempty(result.document.export_to_markdown(), source.uri)
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Docling failed to convert {source.uri}: {exc}") from exc
        try:
            version = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            version = "injected"
        metadata = {**source.metadata, "local_path": str(path), "ocr_languages": ["es", "en"]}
        return ConvertedDocument(
            source_uri=source.uri,
            markdown=markdown,
            content_hash=_hash(markdown),
            converter="docling",
            converter_version=version,
            metadata=metadata,
            line_provenance=_provenance(markdown, source.uri),
        )

    def _convert_jina(self, source: SourceInput) -> ConvertedDocument:
        if source.kind is not SourceKind.URL:
            raise UnsafeRemoteURLError("Jina Reader accepts public URLs only, never local files")
        if not source.allow_remote_service:
            raise UnsafeRemoteURLError("Set allow_remote_service=True to opt in to Jina Reader")
        if not _is_public_http_url(source.uri):
            raise UnsafeRemoteURLError(
                "Jina Reader target must resolve only to public IP addresses"
            )
        endpoint = f"https://r.jina.ai/{quote(source.uri, safe=':/?&=%#')}"
        payload, _ = _read_url(endpoint, self.timeout)
        text = _nonempty(payload.decode("utf-8"), source.uri)
        metadata = {**source.metadata, "remote_service": "jina-reader"}
        return ConvertedDocument(
            source_uri=source.uri,
            markdown=text,
            content_hash=_hash(text),
            converter="jina-reader",
            converter_version="http-api",
            metadata=metadata,
            line_provenance=_provenance(text, source.uri),
        )


def convert(source: SourceInput, *, use_jina: bool = False) -> ConvertedDocument:
    return Converter().convert(source, use_jina=use_jina)
