"""Ollama adapter with a schema-validated JSON boundary."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from raglab.errors import GenerationError, GenerationLengthError
from raglab.generation.models import GenerationConfig, ModelInvocation
from raglab.ollama import model_uses_no_think


class OllamaGenerationModel:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system: str,
        schema: dict[str, Any],
        config: GenerationConfig,
    ) -> ModelInvocation:
        rendered_prompt = f"/no_think\n{prompt}" if model_uses_no_think(config.model) else prompt
        body = {
            "model": config.model,
            "system": system,
            "prompt": rendered_prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": config.keep_alive,
            "options": {
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
                "temperature": 0,
            },
        }
        payload = self._request(body)
        if payload.get("done_reason") == "length":
            raise GenerationLengthError(
                "Ollama generation exhausted num_predict",
                prompt_tokens=_optional_int(payload.get("prompt_eval_count")),
                generated_tokens=_optional_int(payload.get("eval_count")),
            )
        if payload.get("done") is not True or payload.get("done_reason") != "stop":
            raise GenerationError(
                f"Ollama generation did not finish cleanly: {payload.get('done_reason')!r}"
            )
        raw = payload.get("response")
        if not isinstance(raw, str):
            raise GenerationError("Ollama returned no textual generation response")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GenerationError("Ollama returned invalid generation JSON") from exc
        if not isinstance(parsed, dict):
            raise GenerationError("Ollama generation JSON must be an object")
        return ModelInvocation(
            parsed,
            _optional_int(payload.get("prompt_eval_count")),
            _optional_int(payload.get("eval_count")),
        )

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
            raise GenerationError(f"Ollama generation returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GenerationError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise GenerationError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise GenerationError("Ollama returned a non-object JSON response")
        return payload


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
