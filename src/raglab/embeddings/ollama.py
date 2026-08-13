from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from raglab.errors import EmbeddingError


class OllamaEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "qwen3-embedding:0.6b",
        dimension: int = 1024,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._request("/api/embed", {"model": self.model, "input": list(texts)})
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of embeddings")
        result: list[list[float]] = []
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self.dimension:
                actual = len(vector) if isinstance(vector, list) else "non-vector"
                raise EmbeddingError(
                    f"Embedding {index} has dimension {actual}; expected {self.dimension}"
                )
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise EmbeddingError(f"Embedding {index} contains non-finite values")
            result.append(values)
        return result

    def healthcheck(self) -> dict[str, Any]:
        payload = self._request("/api/tags", None)
        models = {
            item.get("name")
            for item in payload.get("models", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if self.model not in models:
            raise EmbeddingError(
                f"Ollama is running but model {self.model!r} is unavailable; "
                f"run `ollama pull {self.model}`"
            )
        return {"status": "ok", "model": self.model, "dimension": self.dimension}

    def _request(self, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if exc.code == 404:
                raise EmbeddingError(
                    f"Ollama model or endpoint was not found ({self.model}): {detail}"
                ) from exc
            raise EmbeddingError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.base_url}; start Ollama and retry: {exc}"
            ) from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise EmbeddingError("Ollama returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise EmbeddingError("Ollama returned a non-object JSON response")
        return result


def embed_documents(
    texts: Sequence[str],
    *,
    model: str = "qwen3-embedding:0.6b",
    dimension: int = 1024,
) -> list[list[float]]:
    return OllamaEmbeddingProvider(model=model, dimension=dimension).embed_documents(texts)
