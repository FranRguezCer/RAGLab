"""Ollama-backed query rewriting with a strict JSON boundary."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from raglab.retrieval.models import QueryRewrite


class OllamaQueryRewriter:
    """Generate a standalone query and optional lexical/semantic expansions."""

    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rewrite(self, query: str, history: Sequence[str], *, max_expansions: int) -> QueryRewrite:
        if not 0 <= max_expansions <= 2:
            raise ValueError("max_expansions must be between zero and two")
        prompt = self._prompt(query, history, max_expansions)
        payload = self._request(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        )
        raw = payload.get("response")
        if not isinstance(raw, str):
            raise RuntimeError("Ollama returned no textual rewrite response")
        try:
            parsed = json.loads(_strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid rewrite JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Ollama rewrite JSON must be an object")
        standalone = parsed.get("standalone_query")
        expansions = parsed.get("expansions", [])
        if not isinstance(standalone, str) or not standalone.strip():
            raise RuntimeError("Ollama rewrite omitted a non-empty standalone_query")
        if not isinstance(expansions, list) or not all(
            isinstance(item, str) for item in expansions
        ):
            raise RuntimeError("Ollama rewrite expansions must be a list of strings")
        cleaned = tuple(item.strip() for item in expansions if item.strip())[:max_expansions]
        return QueryRewrite(standalone.strip(), cleaned)

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Ollama rewrite returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned a non-object JSON response")
        return payload

    @staticmethod
    def _prompt(query: str, history: Sequence[str], max_expansions: int) -> str:
        history_text = "\n".join(f"- {item}" for item in history) or "(none)"
        return (
            "Rewrite the current retrieval query so it is self-contained. "
            f"Return JSON only with standalone_query and at most {max_expansions} expansions. "
            "Expansions should improve lexical or semantic recall without changing intent.\n\n"
            f"Conversation history:\n{history_text}\n\nCurrent query:\n{query}"
        )


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            return stripped[first_newline + 1 : -3].strip()
    return stripped
