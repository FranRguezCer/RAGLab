from __future__ import annotations

from collections.abc import Sequence

import pytest

from raglab.errors import GenerationError
from raglab.generation.grounding import (
    EVIDENCE_CLAIM_FIXTURE_FINGERPRINT,
    EvidenceClaimVerifier,
    evidence_claim_fixture_fingerprint,
    load_evidence_claim_fixture,
)
from raglab.nli import NLIScores


class CalibratedScorer:
    def __init__(self, *, accept_false: bool = False, reject_true: bool = False) -> None:
        fixture = load_evidence_claim_fixture()
        self.labels = {
            (row["evidence_quote"], row["claim"]): bool(row["supported"])
            for split in ("calibration", "holdout")
            for row in fixture[split]
        }
        self.accept_false = accept_false
        self.reject_true = reject_true

    def score(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[NLIScores | None]:
        scores = []
        for pair in pairs:
            supported = self.labels.get(pair, True)
            accepted = (supported and not self.reject_true) or (
                not supported and self.accept_false
            )
            scores.append(NLIScores(0.99 if accepted else 0.01, 0.0))
        return scores


def test_generation_calibration_fingerprint_is_locked() -> None:
    assert evidence_claim_fixture_fingerprint() == EVIDENCE_CLAIM_FIXTURE_FINGERPRINT


def test_calibrated_verifier_preserves_pair_order() -> None:
    verifier = EvidenceClaimVerifier(CalibratedScorer())

    assert verifier.verify((("evidence", "claim"), ("other", "other claim"))) == (
        True,
        True,
    )


def test_calibration_rejects_any_false_claim_acceptance() -> None:
    verifier = EvidenceClaimVerifier(CalibratedScorer(accept_false=True))

    with pytest.raises(GenerationError, match="precision"):
        verifier.verify(())


def test_calibration_requires_ninety_percent_positive_recall() -> None:
    verifier = EvidenceClaimVerifier(CalibratedScorer(reject_true=True))

    with pytest.raises(GenerationError, match="positive-recall"):
        verifier.verify(())
