from __future__ import annotations

import pytest

from raglab.errors import GenerationError
from raglab.generation import GenerationConfig
from raglab.generation.ollama import OllamaGenerationModel


class StubOllama(OllamaGenerationModel):
    response: dict[str, object]
    body: dict[str, object]

    def _request(self, body: dict[str, object]) -> dict[str, object]:
        self.body = body
        return self.response


def test_ollama_uses_schema_no_think_context_and_positive_ttl() -> None:
    client = StubOllama()
    client.response = {
        "done": True,
        "done_reason": "stop",
        "response": '{"answer":"ok"}',
        "prompt_eval_count": 12,
        "eval_count": 3,
    }
    schema = {"type": "object"}

    result = client.generate(
        "prompt", system="strict system", schema=schema, config=GenerationConfig()
    )

    assert client.body["prompt"] == "/no_think\nprompt"
    assert client.body["system"] == "strict system"
    assert client.body["think"] is False
    assert client.body["format"] is schema
    assert client.body["keep_alive"] == "5m"
    assert client.body["options"] == {
        "num_ctx": 12288,
        "num_predict": 512,
        "temperature": 0,
    }
    assert result.prompt_tokens == 12


def test_ollama_rejects_length_termination() -> None:
    client = StubOllama()
    client.response = {"done": True, "done_reason": "length", "response": "{}"}

    with pytest.raises(GenerationError, match="exhausted num_predict"):
        client.generate("prompt", system="system", schema={}, config=GenerationConfig())


def test_ollama_does_not_send_qwen_command_to_other_models() -> None:
    client = StubOllama()
    client.response = {"done": True, "done_reason": "stop", "response": "{}"}

    client.generate(
        "prompt",
        system="system",
        schema={},
        config=GenerationConfig(model="gemma3:4b"),
    )

    assert client.body["prompt"] == "prompt"
    assert client.body["think"] is False


@pytest.mark.parametrize(
    "ttl", ["", "0", "0s", "0m", "0h", "-1m", "banana", "5minutes", "1e3s"]
)
def test_generation_config_rejects_non_positive_ttl(ttl: str) -> None:
    with pytest.raises(ValueError, match="positive Ollama duration"):
        GenerationConfig(keep_alive=ttl)


@pytest.mark.parametrize("ttl", ["5m", "1h30m", "0.5m", "250ms"])
def test_generation_config_accepts_positive_ollama_durations(ttl: str) -> None:
    assert GenerationConfig(keep_alive=f" {ttl} ").keep_alive == ttl
