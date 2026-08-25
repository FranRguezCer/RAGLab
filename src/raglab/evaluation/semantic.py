"""Bounded, independent semantic rescue for lexically missed benchmark facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol, cast

from raglab.errors import EvaluationError
from raglab.evaluation.models import SemanticConfig

SEMANTIC_MODEL = "cross-encoder/nli-deberta-v3-small"
SEMANTIC_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
MAX_BATCH_SIZE = 8
MAX_TOKENS = 512


class SemanticScorer(Protocol):
    """Score premise/hypothesis pairs with entailment probabilities."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float | None]: ...


class TransformersNLIScorer:
    """Lazy, local-only CPU adapter for the pinned DeBERTa NLI checkpoint."""

    def __init__(self, config: SemanticConfig) -> None:
        model = config.model
        if (
            model.name != SEMANTIC_MODEL
            or model.revision != SEMANTIC_REVISION
            or model.device != "cpu"
            or not 1 <= model.batch_size <= MAX_BATCH_SIZE
            or model.max_tokens != MAX_TOKENS
        ):
            raise EvaluationError("Semantic scoring requires the pinned CPU NLI configuration")
        self.config = config
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._entailment_index: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name,
                revision=self.config.model.revision,
                local_files_only=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name,
                revision=self.config.model.revision,
                local_files_only=True,
            ).to("cpu")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise EvaluationError(
                "Pinned semantic NLI model is unavailable locally; semantic rescue is disabled"
            ) from exc
        labels = {
            str(label).casefold(): int(index)
            for index, label in cast(dict[Any, Any], model.config.id2label).items()
        }
        entailment = next((index for label, index in labels.items() if "entail" in label), None)
        if entailment is None:
            raise EvaluationError("Pinned semantic NLI model has no entailment label")
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._entailment_index = entailment

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float | None]:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._entailment_index is not None
        results: list[float | None] = [None] * len(pairs)
        eligible: list[tuple[int, str, str]] = []
        for index, (premise, hypothesis) in enumerate(pairs):
            encoded = self._tokenizer(premise, hypothesis, truncation=False)
            if len(encoded["input_ids"]) <= self.config.model.max_tokens:
                eligible.append((index, premise, hypothesis))
        batch_size = self.config.model.batch_size
        for offset in range(0, len(eligible), batch_size):
            batch = eligible[offset : offset + batch_size]
            encoded = self._tokenizer(
                [row[1] for row in batch],
                [row[2] for row in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > self.config.model.max_tokens:
                continue
            with self._torch.inference_mode():
                logits = self._model(
                    **{key: value.to("cpu") for key, value in encoded.items()}
                ).logits
                probabilities = self._torch.softmax(logits, dim=-1)[:, self._entailment_index]
            for row, probability in zip(batch, probabilities.tolist(), strict=True):
                results[row[0]] = float(probability)
        return results


def semantic_fingerprint(config: SemanticConfig) -> str:
    payload_dict = asdict(config)
    payload_dict["calibration"]["fixture"] = Path(config.calibration.fixture).name
    payload = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def render_pair(config: SemanticConfig, *, answer: str, claim: str) -> tuple[str, str]:
    return (
        config.template.premise.format(answer=answer, semantic_claim=claim),
        config.template.hypothesis.format(answer=answer, semantic_claim=claim),
    )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    enabled: bool
    threshold: float | None
    calibration_pairs: int
    holdout_pairs: int
    false_promotions: int
    valid_promotions: int
    model: str
    revision: str
    fixture_fingerprint: str
    reason: str | None = None


def load_calibration_fixture(path: str | Path | None = None) -> dict[str, Any]:
    fixture_path = (
        Path(path)
        if path is not None
        else Path(str(files("raglab.evaluation.fixtures").joinpath("semantic_calibration_v1.json")))
    )
    try:
        payload = json.loads(fixture_path.read_text())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Could not load semantic calibration fixture: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Semantic calibration fixture must be an object")
    calibration = payload.get("calibration")
    holdout = payload.get("holdout")
    if not isinstance(calibration, list) or len(calibration) != 96:
        raise EvaluationError("Semantic calibration fixture must contain 96 calibration pairs")
    if not isinstance(holdout, list) or len(holdout) != 32:
        raise EvaluationError("Semantic calibration fixture must contain 32 holdout pairs")
    return cast(dict[str, Any], payload)


def calibration_fixture_fingerprint(path: str | Path | None = None) -> str:
    payload = load_calibration_fixture(path)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def calibrate(config: SemanticConfig, scorer: SemanticScorer) -> CalibrationResult:
    payload = load_calibration_fixture(config.calibration.fixture)
    fixture_fingerprint = calibration_fixture_fingerprint(config.calibration.fixture)
    calibration = cast(list[dict[str, str]], payload["calibration"])
    holdout = cast(list[dict[str, str]], payload["holdout"])
    pairs = [render_pair(config, answer=row["answer"], claim=row["claim"]) for row in calibration]
    try:
        scores = scorer.score(pairs)
    except Exception as exc:
        return _disabled(config, fixture_fingerprint, str(exc))
    if len(scores) != len(calibration) or any(score is None for score in scores):
        return _disabled(
            config, fixture_fingerprint, "one or more calibration pairs were unscorable"
        )
    invalid_max = max(
        cast(float, score)
        for row, score in zip(calibration, scores, strict=True)
        if row["label"] != "valid"
    )
    valid_scores = sorted(
        cast(float, score)
        for row, score in zip(calibration, scores, strict=True)
        if row["label"] == "valid" and cast(float, score) > invalid_max
    )
    if not valid_scores:
        return _disabled(config, fixture_fingerprint, "no zero-false-promotion threshold exists")
    threshold = valid_scores[0]
    try:
        holdout_scores = scorer.score(
            [render_pair(config, answer=row["answer"], claim=row["claim"]) for row in holdout]
        )
    except Exception as exc:
        return _disabled(config, fixture_fingerprint, str(exc))
    if len(holdout_scores) != len(holdout) or any(score is None for score in holdout_scores):
        return _disabled(config, fixture_fingerprint, "one or more holdout pairs were unscorable")
    false_promotions = sum(
        row["label"] != "valid" and cast(float, score) >= threshold
        for row, score in zip(holdout, holdout_scores, strict=True)
    )
    if false_promotions:
        return CalibrationResult(
            False,
            None,
            len(calibration),
            len(holdout),
            false_promotions,
            0,
            config.model.name,
            config.model.revision,
            fixture_fingerprint,
            "holdout produced a false promotion",
        )
    return CalibrationResult(
        True,
        threshold,
        len(calibration),
        len(holdout),
        0,
        sum(
            row["label"] == "valid" and cast(float, score) >= threshold
            for row, score in zip(calibration, scores, strict=True)
        ),
        config.model.name,
        config.model.revision,
        fixture_fingerprint,
    )


def _disabled(config: SemanticConfig, fingerprint: str, reason: str) -> CalibrationResult:
    return CalibrationResult(
        False,
        None,
        96,
        32,
        0,
        0,
        config.model.name,
        config.model.revision,
        fingerprint,
        reason,
    )
