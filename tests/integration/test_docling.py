from __future__ import annotations

import importlib.util

import pytest

from raglab import SourceInput
from raglab.conversion import Converter

pytestmark = pytest.mark.integration


@pytest.mark.skipif(importlib.util.find_spec("docling") is None, reason="Docling is not installed")
def test_docling_converts_lightweight_html(tmp_path) -> None:
    path = tmp_path / "sample.html"
    path.write_text("<h1>Local</h1><p>Converted content.</p>", encoding="utf-8")
    result = Converter().convert(SourceInput.path(path))
    assert "Local" in result.markdown
    assert "Converted content" in result.markdown
