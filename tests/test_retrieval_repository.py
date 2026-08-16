from __future__ import annotations

import json

import pytest

from raglab.retrieval import MetadataFilter, PostgresRetrievalRepository, compile_filters


def test_metadata_filters_are_fully_parameterized() -> None:
    hostile = "acme' OR TRUE --"
    sql, params = compile_filters(
        (
            MetadataFilter("document.metadata.tenant_id", "eq", hostile),
            MetadataFilter("chunk.metadata.language", "in", ("en", "es")),
            MetadataFilter("document.source_uri", "contains", "100%_safe"),
        )
    )

    assert hostile not in sql
    assert "tenant_id" not in sql
    assert params == (
        ["tenant_id"],
        json.dumps(hostile),
        ["language"],
        json.dumps("en"),
        json.dumps("es"),
        "%100\\%\\_safe%",
    )
    assert "d.metadata" in sql
    assert "ch.metadata" in sql


def test_empty_in_filter_is_always_false() -> None:
    sql, params = compile_filters((MetadataFilter("document.metadata.tenant", "in", ()),))

    assert sql == "(FALSE)"
    assert params == ()


def test_nullable_document_fields_use_sql_null_semantics() -> None:
    sql, params = compile_filters(
        (
            MetadataFilter("document.title", "eq", None),
            MetadataFilter("document.media_type", "ne", None),
            MetadataFilter("document.source_name", "in", ("guide.md", None)),
        )
    )

    assert "d.title IS NULL" in sql
    assert "d.media_type IS NOT NULL" in sql
    assert "d.source_name IN (%s) OR d.source_name IS NULL" in sql
    assert params == ("guide.md",)


def test_lexical_search_uses_paradedb_025_operator_and_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    class Cursor:
        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, sql: str, _params: object) -> None:
            executed.append(sql)

        def fetchall(self) -> list[tuple[object, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    class Context:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    repository = PostgresRetrievalRepository("unused")
    monkeypatch.setattr(repository, "_connect", lambda: Context())

    assert repository.lexical_search("documents", "E17", (), limit=5) == []
    assert "pdb.score(ch.id)" in executed[0]
    assert "ch.content ||| %s" in executed[0]


@pytest.mark.parametrize(
    ("field", "operator", "value"),
    [
        ("chunks.metadata.tenant", "eq", "a"),
        ("document.id", "eq", "a"),
        ("document.metadata.tenant", "gt", "a"),
        ("document.metadata.tenant", "in", "a"),
    ],
)
def test_filter_rejects_unsupported_or_malformed_input(
    field: str, operator: str, value: str
) -> None:
    with pytest.raises(ValueError):
        MetadataFilter(field, operator, value)
