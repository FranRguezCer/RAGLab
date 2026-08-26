"""Pinned, local-only NLI primitives shared by evaluation and generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

NLI_MODEL = "tasksource/deberta-small-long-nli"
NLI_REVISION = "9a77395d4d3751be9e2a69c4ae318491d9b3fffb"
NLI_SNAPSHOT_FILES = (
    "added_tokens.json",
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "spm.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
MAX_BATCH_SIZE = 8
MAX_TOKENS = 512


class PinnedNLIError(RuntimeError):
    """The mandatory pinned local NLI scorer cannot be used safely."""


class NLIModelConfig(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...


@dataclass(frozen=True, slots=True)
class PinnedNLIConfig:
    name: str = NLI_MODEL
    revision: str = NLI_REVISION
    device: str = "cpu"
    batch_size: int = MAX_BATCH_SIZE
    max_tokens: int = MAX_TOKENS


@dataclass(frozen=True, slots=True)
class NLIScores:
    entailment: float
    contradiction: float


class NLIScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores | None]: ...


class ClaimVerifier(Protocol):
    """Verifies each ``premise -> claim`` pair without changing its order."""

    def verify(self, pairs: Sequence[tuple[str, str]]) -> tuple[bool, ...]: ...


class PinnedTransformersNLIScorer:
    """Lazy CPU scorer resolved only from the pinned local safetensors snapshot."""

    def __init__(self, config: NLIModelConfig | None = None) -> None:
        resolved_config: NLIModelConfig = config if config is not None else PinnedNLIConfig()
        if (
            resolved_config.name != NLI_MODEL
            or resolved_config.revision != NLI_REVISION
            or resolved_config.device != "cpu"
            or not 1 <= resolved_config.batch_size <= MAX_BATCH_SIZE
            or resolved_config.max_tokens != MAX_TOKENS
        ):
            raise PinnedNLIError("NLI scoring requires the pinned CPU configuration")
        self.model_config = resolved_config
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._entailment_index: int | None = None
        self._contradiction_index: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            snapshot = snapshot_download(
                repo_id=self.model_config.name,
                revision=self.model_config.revision,
                local_files_only=True,
                allow_patterns=list(NLI_SNAPSHOT_FILES),
            )
            tokenizer = cast(Any, AutoTokenizer).from_pretrained(snapshot, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True
            ).to("cpu")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise PinnedNLIError(
                "Pinned NLI model is unavailable locally; install raglab[generation] and "
                f"download {NLI_MODEL}@{NLI_REVISION} before generating"
            ) from exc
        labels = {
            str(value).casefold(): int(key)
            for key, value in cast(dict[Any, Any], model.config.id2label).items()
        }
        entailment = next((value for key, value in labels.items() if "entail" in key), None)
        contradiction = next((value for key, value in labels.items() if "contrad" in key), None)
        if entailment is None or contradiction is None or entailment == contradiction:
            raise PinnedNLIError(
                "Pinned NLI model needs distinct entailment and contradiction labels"
            )
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._entailment_index = entailment
        self._contradiction_index = contradiction

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores | None]:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        assert self._entailment_index is not None and self._contradiction_index is not None
        results: list[NLIScores | None] = [None] * len(pairs)
        eligible: list[tuple[int, str, str]] = []
        for index, (premise, hypothesis) in enumerate(pairs):
            token_count = len(
                self._tokenizer(premise, hypothesis, truncation=False)["input_ids"]
            )
            if token_count <= self.model_config.max_tokens:
                eligible.append((index, premise, hypothesis))
        for offset in range(0, len(eligible), self.model_config.batch_size):
            batch = eligible[offset : offset + self.model_config.batch_size]
            encoded = self._tokenizer(
                [row[1] for row in batch],
                [row[2] for row in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > self.model_config.max_tokens:
                continue
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(
                    self._model(**{key: value.to("cpu") for key, value in encoded.items()}).logits,
                    dim=-1,
                )
            for row, entailment, contradiction in zip(
                batch,
                probabilities[:, self._entailment_index].tolist(),
                probabilities[:, self._contradiction_index].tolist(),
                strict=True,
            ):
                scores = NLIScores(float(entailment), float(contradiction))
                if all(
                    math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in (scores.entailment, scores.contradiction)
                ):
                    results[row[0]] = scores
        return results
