"""Read-only PostgreSQL access for hybrid retrieval."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

from raglab.contracts import Citation, ProvenanceStatus
from raglab.errors import StorageError
from raglab.retrieval.models import (
    CollectionMetadata,
    MetadataFilter,
    NeighborChunk,
    RetrievedChunk,
)

if TYPE_CHECKING:
    from psycopg import Connection


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def compile_filters(filters: Sequence[MetadataFilter]) -> tuple[str, tuple[object, ...]]:
    """Compile allow-listed filters to SQL with values and JSON paths as parameters."""
    clauses: list[str] = []
    params: list[object] = []
    document_columns = {
        "document.source_uri": "d.source_uri",
        "document.source_name": "d.source_name",
        "document.title": "d.title",
        "document.media_type": "d.media_type",
    }
    for item in filters:
        column = document_columns.get(item.field)
        if column is not None:
            if item.operator == "contains":
                clauses.append(f"{column} LIKE %s ESCAPE '\\\\'")
                params.append(f"%{_escape_like(str(item.value))}%")
            elif item.operator == "in":
                values = item.value
                assert isinstance(values, tuple)
                if not values:
                    clauses.append("FALSE")
                else:
                    non_null = tuple(value for value in values if value is not None)
                    alternatives: list[str] = []
                    if non_null:
                        placeholders = ", ".join("%s" for _ in non_null)
                        alternatives.append(f"{column} IN ({placeholders})")
                        params.extend(non_null)
                    if None in values:
                        alternatives.append(f"{column} IS NULL")
                    clauses.append("(" + " OR ".join(alternatives) + ")")
            elif item.value is None:
                null_check = "IS NULL" if item.operator == "eq" else "IS NOT NULL"
                clauses.append(f"{column} {null_check}")
            else:
                clauses.append(f"{column} {'=' if item.operator == 'eq' else '<>'} %s")
                params.append(item.value)
            continue

        parts = item.field.split(".")
        json_column = "d.metadata" if parts[0] == "document" else "ch.metadata"
        path = parts[2:]
        if item.operator == "contains":
            clauses.append(f"({json_column} #>> %s::text[]) LIKE %s ESCAPE '\\\\'")
            params.extend((path, f"%{_escape_like(str(item.value))}%"))
        elif item.operator == "in":
            values = item.value
            assert isinstance(values, tuple)
            if not values:
                clauses.append("FALSE")
            else:
                placeholders = ", ".join("%s::jsonb" for _ in values)
                clauses.append(f"({json_column} #> %s::text[]) IN ({placeholders})")
                params.append(path)
                params.extend(json.dumps(value) for value in values)
        else:
            comparison = "=" if item.operator == "eq" else "<>"
            clauses.append(f"({json_column} #> %s::text[]) {comparison} %s::jsonb")
            params.extend((path, json.dumps(item.value)))
    if not clauses:
        return "TRUE", ()
    return " AND ".join(f"({clause})" for clause in clauses), tuple(params)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class RetrievalRepository(Protocol):
    def collection_metadata(self, collection: str) -> CollectionMetadata | None: ...

    def semantic_search(
        self,
        collection: str,
        embedding: Sequence[float],
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
        exact: bool,
        ef_search: int,
    ) -> list[RetrievedChunk]: ...

    def lexical_search(
        self,
        collection: str,
        query: str,
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
    ) -> list[RetrievedChunk]: ...

    def document_chunks(
        self, document_id: str, filters: Sequence[MetadataFilter]
    ) -> list[NeighborChunk]: ...


class PostgresRetrievalRepository:
    """Dedicated read-only repository; it never migrates or changes ingestion data."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def collection_metadata(self, collection: str) -> CollectionMetadata | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT name, model, dimension FROM collections WHERE name = %s",
                (collection,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return CollectionMetadata(str(row[0]), str(row[1]), int(row[2]))

    @contextmanager
    def _connect(self) -> Iterator[Connection[Any]]:
        try:
            import psycopg
        except ImportError as exc:
            raise StorageError("Install the project dependencies to use PostgreSQL") from exc
        try:
            with psycopg.connect(self.dsn) as connection:
                connection.execute("SET TRANSACTION READ ONLY")
                yield connection
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"PostgreSQL retrieval failed: {exc}") from exc

    def semantic_search(
        self,
        collection: str,
        embedding: Sequence[float],
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
        exact: bool,
        ef_search: int,
    ) -> list[RetrievedChunk]:
        where, filter_params = compile_filters(filters)
        vector = _vector(embedding)
        with self._connect() as connection, connection.cursor() as cursor:
            if exact:
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute("SET LOCAL enable_bitmapscan = off")
            else:
                cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            cursor.execute(
                f"""SELECT {self._columns()}, ch.embedding <=> %s::vector AS channel_score
                    FROM chunks ch
                    JOIN documents d ON d.id = ch.document_id
                    JOIN collections c ON c.id = d.collection_id
                    WHERE c.name = %s AND {where}
                    ORDER BY ch.embedding <=> %s::vector, ch.id
                    LIMIT %s""",
                (vector, collection, *filter_params, vector, limit),
            )
            rows = cursor.fetchall()
        return [self._row(row, ann=True) for row in rows]

    def lexical_search(
        self,
        collection: str,
        query: str,
        filters: Sequence[MetadataFilter],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        where, filter_params = compile_filters(filters)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT {self._columns()}, pdb.score(ch.id) AS channel_score
                    FROM chunks ch
                    JOIN documents d ON d.id = ch.document_id
                    JOIN collections c ON c.id = d.collection_id
                    WHERE c.name = %s AND ch.content ||| %s AND {where}
                    ORDER BY channel_score DESC, ch.id
                    LIMIT %s""",
                (collection, query, *filter_params, limit),
            )
            rows = cursor.fetchall()
        return [self._row(row, ann=False) for row in rows]

    def document_chunks(
        self, document_id: str, filters: Sequence[MetadataFilter]
    ) -> list[NeighborChunk]:
        where, filter_params = compile_filters(filters)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT {self._columns()}, ({where}) AS matches_filters
                    FROM chunks ch
                    JOIN documents d ON d.id = ch.document_id
                    WHERE d.id = %s
                    ORDER BY ch.chunk_index""",
                (*filter_params, document_id),
            )
            rows = cursor.fetchall()
        return [NeighborChunk(self._row(row[:-1]), bool(row[-1])) for row in rows]

    @staticmethod
    def _columns() -> str:
        return """ch.id, d.id, ch.chunk_index, ch.content, ch.token_count, ch.heading_path,
                  ch.embedding::text, d.source_uri, d.source_name, d.title,
                  ch.start_page, ch.end_page, ch.start_line, ch.end_line,
                  d.provenance_status, d.provenance_warnings, d.metadata, ch.metadata"""

    @staticmethod
    def _row(row: Sequence[Any], *, ann: bool | None = None) -> RetrievedChunk:
        score = float(row[18]) if ann is not None else None
        return RetrievedChunk(
            chunk_id=str(row[0]),
            document_id=str(row[1]),
            chunk_index=int(row[2]),
            content=str(row[3]),
            token_count=int(row[4]),
            heading_path=tuple(row[5]),
            embedding=tuple(float(value) for value in str(row[6]).strip("[]").split(",")),
            citation=Citation(
                source_uri=str(row[7]),
                source_name=str(row[8]),
                title=row[9],
                heading_path=tuple(row[5]),
                start_page=row[10],
                end_page=row[11],
                start_line=row[12],
                end_line=row[13],
                provenance_status=ProvenanceStatus(row[14]),
                provenance_warnings=tuple(row[15]),
            ),
            document_metadata=row[16],
            chunk_metadata=row[17],
            ann_distance=score if ann else None,
            bm25_score=score if ann is False else None,
        )
