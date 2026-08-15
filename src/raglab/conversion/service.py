from __future__ import annotations

import hashlib
import importlib.metadata
import ipaddress
import mimetypes
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from raglab.contracts import (
    ConvertedDocument,
    LineProvenance,
    ProvenanceStatus,
    SourceInput,
    SourceKind,
)
from raglab.errors import ConversionError, EmptySourceError, UnsafeRemoteURLError

_DIRECT_SUFFIXES = {".md", ".markdown", ".txt"}
_NONPAGINATED_SUFFIXES = {".html", ".htm", ".md", ".markdown", ".txt"}
_PAGE_BREAK = "<!-- RAGLAB_PAGE_BREAK -->"
_USER_AGENT = "raglab/0.1 (+local ingestion)"


@dataclass(frozen=True, slots=True)
class _DownloadedResource:
    payload: bytes
    media_type: str | None
    content_disposition_filename: str | None
    final_url: str


def _nonempty(text: str, uri: str) -> str:
    text = text.strip("\ufeff")
    if not text.strip():
        raise EmptySourceError(f"Source produced no content: {uri}")
    return text


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _line_count(text: str) -> int:
    return text.count("\n") + 1


def _canonical_provenance(
    text: str,
    *,
    source_lines: bool = False,
    page_numbers: Sequence[int | None] | None = None,
) -> tuple[LineProvenance, ...]:
    count = _line_count(text)
    if page_numbers is not None and len(page_numbers) != count:
        raise ValueError("Page provenance must contain one entry per canonical Markdown line")
    return tuple(
        LineProvenance(
            markdown_line=line,
            source_line=line if source_lines else None,
            page_number=None if page_numbers is None else page_numbers[line - 1],
        )
        for line in range(1, count + 1)
    )


def _clean_filename(value: str | None) -> str | None:
    if value is None:
        return None
    name = unquote(value).replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name or None


def _url_source_name(
    source_uri: str,
    *,
    content_disposition_filename: str | None = None,
    final_url: str | None = None,
) -> str:
    disposition_name = _clean_filename(content_disposition_filename)
    if disposition_name is not None:
        return disposition_name
    resolved_uri = final_url or source_uri
    parsed = urlparse(resolved_uri)
    path_name = _clean_filename(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    if path_name is not None:
        return path_name
    return parsed.hostname or resolved_uri


def _source_name(source: SourceInput) -> str:
    if source.kind is SourceKind.PATH:
        return Path(source.uri).expanduser().name
    if source.kind is SourceKind.URL:
        return _url_source_name(source.uri)
    parsed = urlparse(source.uri)
    path_name = _clean_filename(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    return path_name or parsed.hostname or source.uri


def _media_type_for_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    if suffix == ".txt":
        return "text/plain"
    return mimetypes.guess_type(path.name)[0]


def _read_url(url: str, timeout: float) -> _DownloadedResource:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            headers = response.headers
            media_type = headers.get_content_type() if headers.get("Content-Type") else None
            return _DownloadedResource(
                payload=response.read(),
                media_type=media_type,
                content_disposition_filename=headers.get_filename(),
                final_url=response.geturl(),
            )
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


def _docling_title(document: Any) -> str | None:
    """Read a converter-recognised title without treating its filename as a title."""
    for item in getattr(document, "texts", ()):
        label = getattr(item, "label", None)
        label_value = getattr(label, "value", label)
        if label_value == "title":
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _docling_page_numbers(document: Any) -> tuple[int, ...]:
    pages = getattr(document, "pages", None)
    if not pages:
        return ()
    page_numbers = tuple(sorted(int(number) for number in pages))
    if any(number < 1 for number in page_numbers) or len(set(page_numbers)) != len(page_numbers):
        raise ValueError("Docling returned invalid page identifiers")
    return page_numbers


def _mapped_docling_markdown(
    document: Any,
    fallback_markdown: str,
    *,
    source_uri: str,
    allow_pages: bool,
) -> tuple[
    str,
    tuple[LineProvenance, ...],
    ProvenanceStatus,
    tuple[str, ...],
]:
    unavailable = "Original line/page provenance is unavailable for this converted source."
    if not allow_pages:
        return (
            fallback_markdown,
            _canonical_provenance(fallback_markdown),
            ProvenanceStatus.UNAVAILABLE,
            (unavailable,),
        )
    try:
        page_numbers = _docling_page_numbers(document)
    except (TypeError, ValueError, AttributeError) as exc:
        warning = f"Docling page provenance could not be read: {exc}"
        return (
            fallback_markdown,
            _canonical_provenance(fallback_markdown),
            ProvenanceStatus.PARTIAL,
            (warning,),
        )
    if not page_numbers:
        return (
            fallback_markdown,
            _canonical_provenance(fallback_markdown),
            ProvenanceStatus.UNAVAILABLE,
            (unavailable,),
        )
    if len(page_numbers) == 1:
        pages = (page_numbers[0],) * _line_count(fallback_markdown)
        return (
            fallback_markdown,
            _canonical_provenance(fallback_markdown, page_numbers=pages),
            ProvenanceStatus.COMPLETE,
            (),
        )

    try:
        marked = _nonempty(
            document.export_to_markdown(page_break_placeholder=_PAGE_BREAK),
            source_uri,
        )
        output_lines: list[str] = []
        mapped_pages: list[int | None] = []
        page_index = 0
        markers = 0
        marked_lines = marked.splitlines(keepends=True)
        for index, line in enumerate(marked_lines):
            if line.strip() == _PAGE_BREAK:
                if (
                    not output_lines
                    or output_lines[-1].strip()
                    or index + 1 >= len(marked_lines)
                    or marked_lines[index + 1].strip()
                ):
                    raise ValueError("page break was not surrounded by separator blank lines")
                output_lines.pop()
                mapped_pages.pop()
                markers += 1
                page_index += 1
                continue
            if page_index >= len(page_numbers):
                raise ValueError("page break count exceeds Docling document pages")
            output_lines.append(line)
            mapped_pages.append(page_numbers[page_index])
        if markers != len(page_numbers) - 1:
            raise ValueError(
                f"expected {len(page_numbers) - 1} page breaks, found {markers}"
            )
        markdown = _nonempty("".join(output_lines), source_uri)
        if markdown != fallback_markdown:
            raise ValueError("page-marked export did not reconstruct canonical Markdown")
        while len(mapped_pages) < _line_count(markdown):
            mapped_pages.append(page_numbers[-1])
        provenance = _canonical_provenance(markdown, page_numbers=mapped_pages)
    except Exception as exc:
        warning = f"Docling page provenance mapping failed: {exc}"
        return (
            fallback_markdown,
            _canonical_provenance(fallback_markdown),
            ProvenanceStatus.PARTIAL,
            (warning,),
        )
    return markdown, provenance, ProvenanceStatus.COMPLETE, ()


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
            return self._direct(
                source,
                source.content,
                source_name=_source_name(source),
                media_type="text/markdown",
            )
        if source.kind is SourceKind.URL:
            return self._convert_url_locally(source)
        return self._convert_path(source)

    def _direct(
        self,
        source: SourceInput,
        content: str | bytes | None,
        *,
        source_name: str,
        media_type: str | None,
    ) -> ConvertedDocument:
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
            source_name=source_name,
            media_type=media_type,
            metadata=dict(source.metadata),
            line_provenance=_canonical_provenance(text, source_lines=True),
            provenance_status=ProvenanceStatus.COMPLETE,
        )

    def _convert_path(self, source: SourceInput) -> ConvertedDocument:
        path = Path(source.uri).expanduser()
        if not path.is_file():
            raise ConversionError(f"Source file does not exist: {path}")
        source_name = path.name
        media_type = _media_type_for_path(path)
        if path.suffix.lower() in _DIRECT_SUFFIXES:
            try:
                return self._direct(
                    source,
                    path.read_text(encoding="utf-8"),
                    source_name=source_name,
                    media_type=media_type,
                )
            except UnicodeDecodeError as exc:
                raise ConversionError(f"Source is not valid UTF-8: {path}") from exc
        return self._docling(
            source,
            path,
            source_name=source_name,
            media_type=media_type,
        )

    def _convert_url_locally(self, source: SourceInput) -> ConvertedDocument:
        parsed = urlparse(source.uri)
        if parsed.scheme not in {"http", "https"}:
            raise ConversionError("Local URL conversion accepts only http and https")
        downloaded = _read_url(source.uri, self.timeout)
        source_name = _url_source_name(
            source.uri,
            content_disposition_filename=downloaded.content_disposition_filename,
            final_url=downloaded.final_url,
        )
        suffix = Path(source_name).suffix.lower() or Path(
            urlparse(downloaded.final_url).path
        ).suffix
        if suffix in _DIRECT_SUFFIXES or downloaded.media_type in {
            "text/plain",
            "text/markdown",
        }:
            return self._direct(
                source,
                downloaded.payload,
                source_name=source_name,
                media_type=downloaded.media_type,
            )

        # Docling needs a path for binary inputs. The suffix preserves format hints.
        import tempfile

        guessed = suffix or mimetypes.guess_extension(downloaded.media_type or "") or ".bin"
        with tempfile.TemporaryDirectory(prefix="raglab-") as directory:
            path = Path(directory, f"download{guessed}")
            path.write_bytes(downloaded.payload)
            document = self._docling(
                source,
                path,
                source_name=source_name,
                media_type=downloaded.media_type,
            )
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

    def _docling(
        self,
        source: SourceInput,
        path: Path,
        *,
        source_name: str,
        media_type: str | None,
    ) -> ConvertedDocument:
        try:
            result = self._new_docling_converter().convert(path)
            fallback_markdown = _nonempty(result.document.export_to_markdown(), source.uri)
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Docling failed to convert {source.uri}: {exc}") from exc

        allow_pages = path.suffix.lower() not in _NONPAGINATED_SUFFIXES and media_type not in {
            "text/html",
            "text/markdown",
            "text/plain",
        }
        markdown, provenance, status, warnings = _mapped_docling_markdown(
            result.document,
            fallback_markdown,
            source_uri=source.uri,
            allow_pages=allow_pages,
        )
        try:
            version = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            version = "injected"
        metadata = {**source.metadata, "ocr_languages": ["es", "en"]}
        if source.kind is SourceKind.PATH:
            metadata["local_path"] = str(path)
        try:
            title = _docling_title(result.document)
        except Exception as exc:
            warnings = (*warnings, f"Docling title extraction failed: {exc}")
            if status is ProvenanceStatus.COMPLETE:
                status = ProvenanceStatus.PARTIAL
            title = None
        return ConvertedDocument(
            source_uri=source.uri,
            markdown=markdown,
            content_hash=_hash(markdown),
            converter="docling",
            converter_version=version,
            source_name=source_name,
            media_type=media_type,
            title=title,
            metadata=metadata,
            line_provenance=provenance,
            provenance_status=status,
            provenance_warnings=warnings,
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
        downloaded = _read_url(endpoint, self.timeout)
        text = _nonempty(downloaded.payload.decode("utf-8"), source.uri)
        metadata = {**source.metadata, "remote_service": "jina-reader"}
        warning = "Original line/page provenance is unavailable for Jina Reader output."
        return ConvertedDocument(
            source_uri=source.uri,
            markdown=text,
            content_hash=_hash(text),
            converter="jina-reader",
            converter_version="http-api",
            source_name=_url_source_name(source.uri),
            media_type=downloaded.media_type,
            metadata=metadata,
            line_provenance=_canonical_provenance(text),
            provenance_status=ProvenanceStatus.UNAVAILABLE,
            provenance_warnings=(warning,),
        )


def convert(source: SourceInput, *, use_jina: bool = False) -> ConvertedDocument:
    return Converter().convert(source, use_jina=use_jina)
