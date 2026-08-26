"""Calibrated, local-only NLI grading for benchmark facts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol, cast

from raglab.errors import EvaluationError
from raglab.evaluation.models import EvaluationManifest, SemanticConfig

SEMANTIC_MODEL = "tasksource/deberta-small-long-nli"
SEMANTIC_REVISION = "9a77395d4d3751be9e2a69c4ae318491d9b3fffb"
SEMANTIC_SNAPSHOT_FILES = (
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
CALIBRATION_PAIRS = 128
HOLDOUT_PAIRS = 112
CALIBRATION_RESCUE_PAIRS = 48
HOLDOUT_RESCUE_PAIRS = 32
LEXICAL_CONTRADICTIONS_PER_SPLIT = 16


@dataclass(frozen=True, slots=True)
class NLIScores:
    entailment: float
    contradiction: float


class SemanticScorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores | None]: ...


class TransformersNLIScorer:
    def __init__(self, config: SemanticConfig) -> None:
        m = config.model
        if (
            m.name != SEMANTIC_MODEL
            or m.revision != SEMANTIC_REVISION
            or m.device != "cpu"
            or not 1 <= m.batch_size <= MAX_BATCH_SIZE
            or m.max_tokens != MAX_TOKENS
        ):
            raise EvaluationError("Semantic scoring requires the pinned CPU NLI configuration")
        self.config = config
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
                repo_id=self.config.model.name,
                revision=self.config.model.revision,
                local_files_only=True,
                allow_patterns=list(SEMANTIC_SNAPSHOT_FILES),
            )
            tokenizer = cast(Any, AutoTokenizer).from_pretrained(snapshot, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True
            ).to("cpu")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise EvaluationError(
                "Pinned semantic NLI model is unavailable locally; semantic grading is disabled"
            ) from exc
        labels = {
            str(v).casefold(): int(k)
            for k, v in cast(dict[Any, Any], model.config.id2label).items()
        }
        entailment = next((v for k, v in labels.items() if "entail" in k), None)
        contradiction = next((v for k, v in labels.items() if "contrad" in k), None)
        if entailment is None or contradiction is None or entailment == contradiction:
            raise EvaluationError(
                "Pinned semantic NLI model needs distinct entailment and contradiction labels"
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
        eligible = []
        for i, (premise, hypothesis) in enumerate(pairs):
            if (
                len(self._tokenizer(premise, hypothesis, truncation=False)["input_ids"])
                <= self.config.model.max_tokens
            ):
                eligible.append((i, premise, hypothesis))
        for offset in range(0, len(eligible), self.config.model.batch_size):
            batch = eligible[offset : offset + self.config.model.batch_size]
            encoded = self._tokenizer(
                [x[1] for x in batch],
                [x[2] for x in batch],
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if int(encoded["input_ids"].shape[1]) > self.config.model.max_tokens:
                continue
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(
                    self._model(**{k: v.to("cpu") for k, v in encoded.items()}).logits, dim=-1
                )
            for row, e, c in zip(
                batch,
                probabilities[:, self._entailment_index].tolist(),
                probabilities[:, self._contradiction_index].tolist(),
                strict=True,
            ):
                results[row[0]] = NLIScores(float(e), float(c))
        return results


def semantic_fingerprint(config: SemanticConfig) -> str:
    payload = asdict(config)
    payload["calibration"]["fixture"] = Path(config.calibration.fixture).name
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def render_pair(config: SemanticConfig, *, answer: str, claim: str) -> tuple[str, str]:
    return config.template.premise.format(
        answer=answer, semantic_claim=claim
    ), config.template.hypothesis.format(answer=answer, semantic_claim=claim)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    enabled: bool
    threshold: float | None
    contradiction_threshold: float | None
    calibration_pairs: int
    holdout_pairs: int
    false_promotions: int
    valid_promotions: int
    calibration_valid_pairs: int
    calibration_valid_promotions: int
    calibration_false_promotions: int
    calibration_false_vetoes: int
    calibration_contradiction_vetoes: int
    holdout_valid_pairs: int
    holdout_valid_promotions: int
    holdout_false_promotions: int
    holdout_false_vetoes: int
    holdout_contradiction_vetoes: int
    frozen_rescues: int
    frozen_rejections: int
    model: str
    revision: str
    fixture_fingerprint: str
    reason: str | None = None


def load_calibration_fixture(path: str | Path | None = None) -> dict[str, Any]:
    p = (
        Path(path)
        if path is not None
        else Path(str(files("raglab.evaluation.fixtures").joinpath("semantic_calibration_v2.json")))
    )
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Could not load semantic calibration fixture: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Semantic calibration fixture must be an object")
    calibration, holdout = payload.get("calibration"), payload.get("holdout")
    if not isinstance(calibration, list) or len(calibration) != CALIBRATION_PAIRS:
        raise EvaluationError(
            f"Semantic calibration fixture must contain {CALIBRATION_PAIRS} calibration rows"
        )
    if not isinstance(holdout, list) or len(holdout) != HOLDOUT_PAIRS:
        raise EvaluationError(
            f"Semantic calibration fixture must contain {HOLDOUT_PAIRS} holdout rows"
        )
    allowed = {
        "semantic_positive",
        "semantic_negative",
        "lexical_valid",
        "lexical_neutral",
        "lexical_contradiction",
    }
    for row in [*calibration, *holdout]:
        if (
            not isinstance(row, dict)
            or set(row) != {"fact_id", "answer", "label"}
            or row["label"] not in allowed
        ):
            raise EvaluationError(
                "Semantic calibration rows must contain only fact_id, answer, and a valid label"
            )
        if str(row["fact_id"]) in str(row["answer"]):
            raise EvaluationError("Semantic calibration answers must not contain fact identifiers")
    return cast(dict[str, Any], payload)


def calibration_fixture_fingerprint(path: str | Path | None = None) -> str:
    return hashlib.sha256(
        json.dumps(load_calibration_fixture(path), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def calibrate(
    config: SemanticConfig, scorer: SemanticScorer, manifest: EvaluationManifest | None = None
) -> CalibrationResult:
    payload = load_calibration_fixture(config.calibration.fixture)
    fingerprint = calibration_fixture_fingerprint(config.calibration.fixture)
    facts = _fact_lookup(manifest)
    calibration = cast(list[dict[str, str]], payload["calibration"])
    holdout = cast(list[dict[str, str]], payload["holdout"])
    reason = _validate_routing(calibration, holdout, facts)
    if reason:
        return _disabled(config, fingerprint, reason)
    try:
        raw = scorer.score(_pairs(config, calibration, facts))
    except Exception as exc:
        return _disabled(config, fingerprint, str(exc))
    if not _scoreable(raw, len(calibration)):
        return _disabled(config, fingerprint, "one or more calibration pairs were unscorable")
    scores = cast(list[NLIScores], raw)
    threshold = math.nextafter(
        max(
            s.entailment
            for r, s in zip(calibration, scores, strict=True)
            if r["label"] == "semantic_negative"
        ),
        math.inf,
    )
    contradiction_threshold = math.nextafter(
        max(
            s.contradiction
            for r, s in zip(calibration, scores, strict=True)
            if r["label"] in {"lexical_valid", "lexical_neutral"}
        ),
        math.inf,
    )
    cm = _split_metrics(calibration, scores, threshold, contradiction_threshold)
    try:
        hraw = scorer.score(_pairs(config, holdout, facts))
    except Exception as exc:
        return _result(
            config, fingerprint, threshold, contradiction_threshold, cm, {}, 0, 0, str(exc)
        )
    if not _scoreable(hraw, len(holdout)):
        return _result(
            config,
            fingerprint,
            threshold,
            contradiction_threshold,
            cm,
            {},
            0,
            0,
            "one or more holdout pairs were unscorable",
        )
    hm = _split_metrics(holdout, cast(list[NLIScores], hraw), threshold, contradiction_threshold)
    try:
        fraw = scorer.score(_frozen_pairs(config, facts))
    except Exception as exc:
        return _result(
            config, fingerprint, threshold, contradiction_threshold, cm, hm, 0, 0, str(exc)
        )
    if not _scoreable(fraw, 18):
        return _result(
            config,
            fingerprint,
            threshold,
            contradiction_threshold,
            cm,
            hm,
            0,
            0,
            "one or more frozen regression pairs were unscorable",
        )
    rescues = sum(s.entailment >= threshold for s in cast(list[NLIScores], fraw))
    rejects = 18 - rescues
    enabled = (
        cm["false_promotions"]
        == hm["false_promotions"]
        == cm["false_vetoes"]
        == hm["false_vetoes"]
        == 0
        and cm["contradiction_vetoes"] == hm["contradiction_vetoes"] == 16
        and cm["valid_promotions"] >= 39
        and hm["valid_promotions"] >= 26
        and rescues == 12
        and rejects == 6
    )
    reason = (
        None
        if enabled
        else "calibration did not satisfy precision, veto, recall, and frozen-regression gates"
    )
    return _result(
        config, fingerprint, threshold, contradiction_threshold, cm, hm, rescues, rejects, reason
    )


def _fact_lookup(manifest: EvaluationManifest | None) -> dict[str, tuple[str, tuple[str, ...]]]:
    if manifest is None:
        from raglab.evaluation.manifest import load_manifest

        manifest = load_manifest()
    return {
        f.id: (f.semantic_claim or "", f.answer_variants)
        for c in manifest.cases
        for f in c.required_facts
    }


def _pairs(
    config: SemanticConfig,
    rows: list[dict[str, str]],
    facts: Mapping[str, tuple[str, tuple[str, ...]]],
) -> list[tuple[str, str]]:
    return [render_pair(config, answer=r["answer"], claim=facts[r["fact_id"]][0]) for r in rows]


def _validate_routing(
    calibration: list[dict[str, str]],
    holdout: list[dict[str, str]],
    facts: Mapping[str, tuple[str, tuple[str, ...]]],
) -> str | None:
    from raglab.evaluation.metrics import normalize

    expected = (
        {
            "semantic_positive": 48,
            "semantic_negative": 32,
            "lexical_valid": 16,
            "lexical_neutral": 16,
            "lexical_contradiction": 16,
        },
        {
            "semantic_positive": 32,
            "semantic_negative": 32,
            "lexical_valid": 16,
            "lexical_neutral": 16,
            "lexical_contradiction": 16,
        },
    )
    for name, rows, counts in zip(
        ("calibration", "holdout"), (calibration, holdout), expected, strict=True
    ):
        if {k: sum(r["label"] == k for r in rows) for k in counts} != counts:
            return f"{name} does not contain the locked v2 class counts"
        for r in rows:
            if r["fact_id"] not in facts:
                return f"{name} references an unknown fact"
            lexical = any(normalize(v) in normalize(r["answer"]) for v in facts[r["fact_id"]][1])
            if lexical != r["label"].startswith("lexical_"):
                return f"{name} row does not follow production lexical routing"
    return None


def _scoreable(scores: Sequence[NLIScores | None], expected: int) -> bool:
    return len(scores) == expected and all(
        s is not None
        and math.isfinite(s.entailment)
        and math.isfinite(s.contradiction)
        and 0 <= s.entailment <= 1
        and 0 <= s.contradiction <= 1
        for s in scores
    )


def _split_metrics(
    rows: list[dict[str, str]], scores: list[NLIScores], threshold: float, ct: float
) -> dict[str, int]:
    return {
        "valid_promotions": sum(
            r["label"] == "semantic_positive" and s.entailment >= threshold
            for r, s in zip(rows, scores, strict=True)
        ),
        "false_promotions": sum(
            r["label"] == "semantic_negative" and s.entailment >= threshold
            for r, s in zip(rows, scores, strict=True)
        ),
        "false_vetoes": sum(
            r["label"] in {"lexical_valid", "lexical_neutral"} and s.contradiction >= ct
            for r, s in zip(rows, scores, strict=True)
        ),
        "contradiction_vetoes": sum(
            r["label"] == "lexical_contradiction" and s.contradiction >= ct
            for r, s in zip(rows, scores, strict=True)
        ),
    }


def _frozen_pairs(
    config: SemanticConfig, facts: Mapping[str, tuple[str, tuple[str, ...]]]
) -> list[tuple[str, str]]:
    path = Path(
        str(files("raglab.evaluation.fixtures").joinpath("semantic_frozen_regression_v1.json"))
    )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("Frozen semantic regression artifact is unavailable") from exc
    if payload.get("source_artifact") != "20260825T153524.720063Z":
        raise EvaluationError("Frozen semantic regression has an invalid source lineage")
    pairs = [
        render_pair(config, answer=row["answer"], claim=facts[row["fact_id"]][0])
        for row in payload.get("rows", [])
    ]
    if len(pairs) != 18:
        raise EvaluationError("Frozen semantic regression must contain exactly 18 lexical misses")
    return pairs


def _disabled(config: SemanticConfig, fingerprint: str, reason: str) -> CalibrationResult:
    return _result(config, fingerprint, None, None, {}, {}, 0, 0, reason)


def _result(
    config: SemanticConfig,
    fingerprint: str,
    threshold: float | None,
    ct: float | None,
    c: Mapping[str, int],
    h: Mapping[str, int],
    fr: int,
    fj: int,
    reason: str | None,
) -> CalibrationResult:
    return CalibrationResult(
        reason is None,
        threshold,
        ct,
        128,
        112,
        c.get("false_promotions", 0),
        c.get("valid_promotions", 0),
        48,
        c.get("valid_promotions", 0),
        c.get("false_promotions", 0),
        c.get("false_vetoes", 0),
        c.get("contradiction_vetoes", 0),
        32,
        h.get("valid_promotions", 0),
        h.get("false_promotions", 0),
        h.get("false_vetoes", 0),
        h.get("contradiction_vetoes", 0),
        fr,
        fj,
        config.model.name,
        config.model.revision,
        fingerprint,
        reason,
    )
