from __future__ import annotations

import os
from uuid import uuid4

import psycopg
import pytest

from raglab.storage import PostgresRepository

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("RAGLAB_TEST_DSN"), reason="RAGLAB_TEST_DSN is not set")
def test_evaluation_reset_cascades_without_touching_personal_collections() -> None:
    dsn = os.environ["RAGLAB_TEST_DSN"]
    repository = PostgresRepository(dsn)
    repository.migrate()
    suffix = uuid4().hex
    evaluation = f"raglab-eval-{suffix}"
    personal = f"personal-{suffix}"
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        for name in (evaluation, personal):
            cursor.execute(
                """INSERT INTO collections (name, model, dimension, metric, chunk_config)
                   VALUES (%s, 'qwen3-embedding:0.6b', 1024, 'cosine', '{}'::jsonb)""",
                (name,),
            )
    assert repository.reset_evaluation_collection(evaluation) is True
    with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT name FROM collections WHERE name IN (%s, %s)",
            (evaluation, personal),
        )
        assert cursor.fetchall() == [(personal,)]
        cursor.execute("DELETE FROM collections WHERE name = %s", (personal,))
