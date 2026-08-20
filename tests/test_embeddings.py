import math
import urllib.error
from unittest.mock import patch

import pytest

from raglab.embeddings import OllamaEmbeddingProvider
from raglab.errors import EmbeddingError


class StubOllama(OllamaEmbeddingProvider):
    response: dict[str, object]

    def _request(self, path: str, body: dict[str, object] | None) -> dict[str, object]:
        return self.response


def test_embedding_dimension_is_validated() -> None:
    client = StubOllama(dimension=3)
    client.response = {"embeddings": [[1.0, 2.0]]}
    with pytest.raises(EmbeddingError, match="dimension"):
        client.embed_documents(["text"])


def test_non_finite_embedding_is_rejected() -> None:
    client = StubOllama(dimension=2)
    client.response = {"embeddings": [[1.0, math.nan]]}
    with pytest.raises(EmbeddingError, match="non-finite"):
        client.embed_documents(["text"])


def test_missing_model_has_actionable_health_error() -> None:
    client = StubOllama(model="missing")
    client.response = {"models": [{"name": "other"}]}
    with pytest.raises(EmbeddingError, match="ollama pull missing"):
        client.healthcheck()


def test_unreachable_ollama_has_actionable_error() -> None:
    client = OllamaEmbeddingProvider()
    with (
        patch(
            "raglab.embeddings.ollama.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ),
        pytest.raises(EmbeddingError, match="start Ollama"),
    ):
        client.embed_documents(["text"])


def test_embedding_request_can_pin_model_to_cpu_with_positive_ttl() -> None:
    client = StubOllama(dimension=2, num_gpu=0, num_ctx=4096, keep_alive="5m")
    captured: dict[str, object] = {}

    def request(path: str, body: dict[str, object] | None) -> dict[str, object]:
        captured.update({"path": path, "body": body})
        return {"embeddings": [[1.0, 2.0]]}

    client._request = request  # type: ignore[method-assign]

    assert client.embed_documents(["text"]) == [[1.0, 2.0]]
    assert captured == {
        "path": "/api/embed",
        "body": {
            "model": "qwen3-embedding:0.6b",
            "input": ["text"],
            "options": {"num_gpu": 0, "num_ctx": 4096},
            "keep_alive": "5m",
        },
    }


@pytest.mark.parametrize("ttl", ["-1m", "0h", "forever"])
def test_embedding_rejects_non_positive_or_invalid_ttl(ttl: str) -> None:
    with pytest.raises(ValueError, match="positive Ollama duration"):
        OllamaEmbeddingProvider(keep_alive=ttl)
