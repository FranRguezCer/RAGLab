"""Shared validation for Ollama-specific configuration."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|μs|ms|s|m|h)")


def validate_positive_duration(value: str) -> str:
    """Return a normalized positive Go-style duration or raise ``ValueError``."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("keep_alive must be a positive Ollama duration")
    position = 0
    positive = False
    while position < len(normalized):
        match = _DURATION_PART.match(normalized, position)
        if match is None:
            raise ValueError("keep_alive must be a positive Ollama duration such as '5m'")
        try:
            positive = positive or Decimal(match.group(1)) > 0
        except InvalidOperation as exc:  # pragma: no cover - regex keeps the number decimal
            raise ValueError("keep_alive must be a positive Ollama duration") from exc
        position = match.end()
    if not positive:
        raise ValueError("keep_alive must be a positive Ollama duration")
    return normalized


def model_uses_no_think(model: str) -> bool:
    """Whether the selected Ollama model understands Qwen's ``/no_think`` switch."""
    name = model.rsplit("/", 1)[-1].split(":", 1)[0].casefold()
    return name.startswith("qwen")
