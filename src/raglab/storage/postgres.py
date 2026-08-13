from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Any

from raglab.contracts import CollectionConfig, ConvertedDocument, EmbeddedChunk
from raglab.errors import StorageError

if TYPE_CHECKING:
    from psycopg import Connection


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    source_uri: str
    content: str
    distance: float


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


class PostgresRepository:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextmanager
    def _connect(self) -> Iterator[Connection[Any]]:
        try:
            import psycopg
        except ImportError as exc:
            raise StorageError("Install the project dependencies to use PostgreSQL") from exc
        try:
            with psycopg.connect(self.dsn) as connection:
                yield connection
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"PostgreSQL operation failed: {exc}") from exc

    def migrate(self) -> None:
        sql = files("raglab.storage.migrations").joinpath("001_initial.sql").read_text()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(sql)

    def healthcheck(self) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT current_setting('server_version'), extversion
                   FROM pg_extension WHERE extname = 'vector'"""
            )
            row = cursor.fetchone()
        if row is None:
            raise StorageError(
                "pgvector is not installed; run migrations against the pgvector image"
            )
        return {"status": "ok", "postgres": row[0], "pgvector": row[1]}

    def collection_stats(self, collection: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT name, model, dimension, document_count, chunk_count, token_count
                   FROM collection_stats WHERE name = %s""",
                (collection,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "name": row[0],
            "model": row[1],
            "dimension": row[2],
            "document_count": row[3],
            "chunk_count": row[4],
            "token_count": row[5],
        }

    def current_document_id(
        self,
        config: CollectionConfig,
        source_uri: str,
        content_hash: str,
        fingerprint: str,
    ) -> str | None:
        """Cheap idempotency check; `store` repeats it to close the race window."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT d.id
                   FROM documents d
                   JOIN collections c ON c.id = d.collection_id
                   WHERE c.name = %s AND c.model = %s AND c.dimension = %s
                     AND c.metric = %s AND c.chunk_config = %s::jsonb
                     AND d.source_uri = %s AND d.content_hash = %s
                     AND d.fingerprint = %s""",
                (
                    config.name,
                    config.model,
                    config.dimension,
                    config.metric,
                    json.dumps(dict(config.chunk_config)),
                    source_uri,
                    content_hash,
                    fingerprint,
                ),
            )
            row = cursor.fetchone()
        return str(row[0]) if row else None

    def store(
        self,
        config: CollectionConfig,
        document: ConvertedDocument,
        chunks: Sequence[EmbeddedChunk],
        fingerprint: str,
    ) -> tuple[str, bool]:
        if config.dimension != 1024:
            raise StorageError("The current schema is intentionally fixed to vector(1024)")
        if any(len(item.embedding) != config.dimension for item in chunks):
            raise StorageError("A chunk embedding does not match the collection dimension")
        with self._connect() as connection, connection.cursor() as cursor:
            collection_id = self._ensure_collection(cursor, config)
            cursor.execute(
                """SELECT id FROM documents
                   WHERE collection_id = %s AND source_uri = %s
                     AND content_hash = %s AND fingerprint = %s""",
                (collection_id, document.source_uri, document.content_hash, fingerprint),
            )
            existing = cursor.fetchone()
            if existing:
                return str(existing[0]), True
            cursor.execute(
                """INSERT INTO documents
                       (collection_id, source_uri, markdown, content_hash, converter,
                        converter_version, metadata, fingerprint)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (collection_id, source_uri) DO UPDATE SET
                       markdown = EXCLUDED.markdown,
                       content_hash = EXCLUDED.content_hash,
                       converter = EXCLUDED.converter,
                       converter_version = EXCLUDED.converter_version,
                       metadata = EXCLUDED.metadata,
                       fingerprint = EXCLUDED.fingerprint,
                       updated_at = now()
                   RETURNING id""",
                (
                    collection_id,
                    document.source_uri,
                    document.markdown,
                    document.content_hash,
                    document.converter,
                    document.converter_version,
                    json.dumps(dict(document.metadata)),
                    fingerprint,
                ),
            )
            document_row = cursor.fetchone()
            if document_row is None:
                raise StorageError("Document upsert returned no identifier")
            document_id = document_row[0]
            cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            for item in chunks:
                chunk = item.chunk
                cursor.execute(
                    """INSERT INTO chunks
                           (document_id, chunk_index, content, embedding_text, token_count,
                            heading_path, start_line, end_line, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::vector)""",
                    (
                        document_id,
                        chunk.index,
                        chunk.content,
                        chunk.embedding_text,
                        chunk.token_count,
                        json.dumps(chunk.heading_path),
                        chunk.start_line,
                        chunk.end_line,
                        json.dumps(dict(chunk.metadata)),
                        _vector(item.embedding),
                    ),
                )
        return str(document_id), False

    @staticmethod
    def _ensure_collection(cursor: Any, config: CollectionConfig) -> Any:
        cursor.execute(
            """INSERT INTO collections (name, model, dimension, metric, chunk_config)
               VALUES (%s, %s, %s, %s, %s::jsonb)
               ON CONFLICT (name) DO NOTHING
               RETURNING id""",
            (
                config.name,
                config.model,
                config.dimension,
                config.metric,
                json.dumps(dict(config.chunk_config)),
            ),
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        cursor.execute(
            "SELECT id, model, dimension, metric, chunk_config FROM collections WHERE name = %s",
            (config.name,),
        )
        row = cursor.fetchone()
        if row is None:
            raise StorageError("Collection disappeared during creation")
        expected = (config.model, config.dimension, config.metric, dict(config.chunk_config))
        actual = (row[1], row[2], row[3], row[4])
        if actual != expected:
            raise StorageError(f"Collection {config.name!r} exists with a different configuration")
        return row[0]

    def search(
        self,
        collection: str,
        query: Sequence[float],
        *,
        limit: int = 5,
        exact: bool = False,
        ef_search: int = 100,
    ) -> list[SearchResult]:
        if len(query) != 1024:
            raise StorageError("Query embedding must have dimension 1024")
        with self._connect() as connection, connection.cursor() as cursor:
            if exact:
                cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute("SET LOCAL enable_bitmapscan = off")
            else:
                cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            cursor.execute(
                """SELECT ch.id, d.id, d.source_uri, ch.content,
                          ch.embedding <=> %s::vector AS distance
                   FROM chunks ch
                   JOIN documents d ON d.id = ch.document_id
                   JOIN collections c ON c.id = d.collection_id
                   WHERE c.name = %s
                   ORDER BY ch.embedding <=> %s::vector
                   LIMIT %s""",
                (_vector(query), collection, _vector(query), limit),
            )
            rows = cursor.fetchall()
        return [
            SearchResult(str(row[0]), str(row[1]), row[2], row[3], float(row[4])) for row in rows
        ]
