"""Calibrated evidence-quote to claim verification for generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Any, cast

from raglab.errors import GenerationError
from raglab.nli import (
    ClaimVerifier,
    NLIScorer,
    PinnedNLIError,
    PinnedTransformersNLIScorer,
)

EVIDENCE_CLAIM_THRESHOLD = 0.7
EVIDENCE_CLAIM_FIXTURE_FINGERPRINT = (
    "d7b0b27cc040b81a564b6584ae4cbbaaaf92719d68feddb6daafbac995504bd2"
)


def load_evidence_claim_fixture() -> dict[str, Any]:
    path = files("raglab.generation.fixtures").joinpath("evidence_claim_calibration_v1.json")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise GenerationError(f"Could not load generation grounding calibration: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise GenerationError("Generation grounding calibration has an invalid schema")
    for split in ("calibration", "holdout"):
        rows = payload.get(split)
        if not isinstance(rows, list) or not rows:
            raise GenerationError(f"Generation grounding calibration needs a non-empty {split}")
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"evidence_quote", "claim", "supported"}
                or not isinstance(row["evidence_quote"], str)
                or not row["evidence_quote"].strip()
                or not isinstance(row["claim"], str)
                or not row["claim"].strip()
                or not isinstance(row["supported"], bool)
            ):
                raise GenerationError("Generation grounding calibration contains an invalid row")
    return cast(dict[str, Any], payload)


def evidence_claim_fixture_fingerprint() -> str:
    return hashlib.sha256(
        json.dumps(
            load_evidence_claim_fixture(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class EvidenceClaimVerifier(ClaimVerifier):
    """Mandatory local verifier with locked calibration and holdout gates."""

    def __init__(self, scorer: NLIScorer | None = None) -> None:
        self.scorer = scorer or PinnedTransformersNLIScorer()
        self._calibrated = False

    def _calibrate(self) -> None:
        fingerprint = evidence_claim_fixture_fingerprint()
        if fingerprint != EVIDENCE_CLAIM_FIXTURE_FINGERPRINT:
            raise GenerationError("Generation grounding calibration fingerprint does not match")
        fixture = load_evidence_claim_fixture()
        for split in ("calibration", "holdout"):
            rows = cast(list[dict[str, Any]], fixture[split])
            pairs = [(str(row["evidence_quote"]), str(row["claim"])) for row in rows]
            try:
                scores = self.scorer.score(pairs)
            except PinnedNLIError as exc:
                raise GenerationError(str(exc)) from exc
            if len(scores) != len(rows) or any(score is None for score in scores):
                raise GenerationError(f"Generation grounding {split} contains unscorable pairs")
            accepted = [
                score is not None and score.entailment >= EVIDENCE_CLAIM_THRESHOLD
                for score in scores
            ]
            false_accepts = sum(
                accepted_result and not row["supported"]
                for row, accepted_result in zip(rows, accepted, strict=True)
            )
            positives = sum(bool(row["supported"]) for row in rows)
            recalled = sum(
                accepted_result and bool(row["supported"])
                for row, accepted_result in zip(rows, accepted, strict=True)
            )
            if false_accepts or recalled / positives < 0.9:
                raise GenerationError(
                    f"Generation grounding {split} failed precision or positive-recall gates"
                )
        self._calibrated = True

    def verify(self, pairs: Sequence[tuple[str, str]]) -> tuple[bool, ...]:
        if not self._calibrated:
            self._calibrate()
        try:
            scores = self.scorer.score(pairs)
        except PinnedNLIError as exc:
            raise GenerationError(str(exc)) from exc
        if len(scores) != len(pairs):
            raise GenerationError("Pinned NLI scorer returned the wrong number of claim scores")
        return tuple(
            score is not None and score.entailment >= EVIDENCE_CLAIM_THRESHOLD
            for score in scores
        )
