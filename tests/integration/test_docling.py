from __future__ import annotations

import importlib.util

import pytest

from raglab import ProvenanceStatus, SourceInput
from raglab.conversion import Converter

pytestmark = pytest.mark.integration


@pytest.mark.skipif(importlib.util.find_spec("docling") is None, reason="Docling is not installed")
def test_docling_converts_lightweight_html(tmp_path) -> None:
    path = tmp_path / "sample.html"
    path.write_text("<h1>Local</h1><p>Converted content.</p>", encoding="utf-8")
    result = Converter().convert(SourceInput.path(path))
    assert "Local" in result.markdown
    assert "Converted content" in result.markdown
    assert result.source_name == "sample.html"
    assert result.provenance_status is ProvenanceStatus.UNAVAILABLE
    assert result.provenance_warnings
    assert all(line.page_number is None for line in result.line_provenance)
