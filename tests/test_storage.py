from __future__ import annotations

import math
from typing import Any

from raglab import ConvertedDocument, LineProvenance, ProvenanceStatus
from raglab.storage import PostgresRepository
from raglab.storage.postgres import _chunk_page_range, _vector


def test_vector_serialization_is_explicit_and_precise() -> None:
    assert _vector([1.0, -0.25, math.pi]) == "[1,-0.25,3.1415926535897931]"


def test_chunk_page_range_uses_only_mapped_canonical_lines() -> None:
    document = ConvertedDocument(
        "memory://guide",
        "one\ntwo\nthree",
        "hash",
        "test",
        "1",
        "guide.pdf",
        line_provenance=(
            LineProvenance(1, page_number=3),
            LineProvenance(2),
            LineProvenance(3, page_number=4),
        ),
        provenance_status=ProvenanceStatus.PARTIAL,
    )

    assert _chunk_page_range(document, 1, 3) == (3, 4)
    assert _chunk_page_range(document, 2, 2) == (None, None)
    assert _chunk_page_range(document, None, None) == (None, None)


class _Context:
    def __init__(self, value: object) -> None:
        self.value = value

    def __enter__(self) -> object:
        return self.value

    def __exit__(self, *args: object) -> None:
        return None


class _Cursor:
    def __init__(self, row: tuple[object, ...] | None) -> None:
        self.row = row
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, parameters: tuple[object, ...]) -> None:
        self.executed.append((sql, parameters))

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def cursor(self) -> _Cursor:
        return self._cursor


def test_collection_reconciliation_api_reads_and_deletes_exact_name(
    monkeypatch: Any,
) -> None:
    repository = PostgresRepository("postgresql://unused")
    config_cursor = _Cursor(("rpi-computers", "embed", 1024, "cosine", {"percentile": 85}))
    monkeypatch.setattr(repository, "_connect", lambda: _Context(_Connection(config_cursor)))
    config = repository.collection_config("rpi-computers")
    assert config is not None
    assert config.chunk_config == {"percentile": 85}
    assert config_cursor.executed[0][1] == ("rpi-computers",)

    delete_cursor = _Cursor(("collection-id",))
    monkeypatch.setattr(repository, "_connect", lambda: _Context(_Connection(delete_cursor)))
    assert repository.delete_collection("rpi-computers") is True
    assert "DELETE FROM collections WHERE name = %s" in delete_cursor.executed[0][0]
    assert delete_cursor.executed[0][1] == ("rpi-computers",)
