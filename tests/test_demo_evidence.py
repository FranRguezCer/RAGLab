from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from demo_fixture import COMMIT, create_evidence

import raglab.demo_evidence as evidence_module
from raglab.demo_evidence import (
    InvalidDemoEvidence,
    build_evidence,
    current_commit,
    load_verified_evidence,
)

NOW = datetime(2026, 9, 10, 12, 30, tzinfo=UTC)


def test_verified_evidence_binds_commit_inputs_models_and_gate(tmp_path: Path) -> None:
    path = create_evidence(tmp_path)
    evidence = load_verified_evidence(path, project_root=tmp_path, now=NOW, expected_commit=COMMIT)
    assert evidence["gate"]["passed"] is True
    assert evidence["models"]["generation"]["name"] == "qwen3:4b"
    assert len(evidence["evaluation"]["traces"]) == 6


@pytest.mark.parametrize("mutation", ["bytes", "input", "commit"])
def test_tampered_or_outdated_evidence_fails_closed(tmp_path: Path, mutation: str) -> None:
    path = create_evidence(tmp_path)
    commit = COMMIT
    if mutation == "bytes":
        payload = json.loads(path.read_text())
        payload["evaluation"]["traces"][0]["response"]["answer"] = "tampered"
        path.write_text(json.dumps(payload))
    elif mutation == "input":
        (tmp_path / "data/dataset.json").write_text("changed")
    else:
        commit = "b" * 40
    with pytest.raises(InvalidDemoEvidence):
        load_verified_evidence(path, project_root=tmp_path, now=NOW, expected_commit=commit)


def test_stale_evidence_requires_prepare_again(tmp_path: Path) -> None:
    path = create_evidence(tmp_path, created_at="2026-08-01T00:00:00Z")
    with pytest.raises(InvalidDemoEvidence, match="stale"):
        load_verified_evidence(
            path,
            project_root=tmp_path,
            now=NOW,
            max_age=timedelta(days=7),
            expected_commit=COMMIT,
        )


def test_current_commit_and_builder_reject_invalid_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evidence_module.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": COMMIT + "\n"})(),
    )
    assert current_commit(tmp_path) == COMMIT
    path = create_evidence(tmp_path / "valid")
    valid = json.loads(path.read_text())
    with pytest.raises(InvalidDemoEvidence, match="commit"):
        build_evidence(
            valid["evaluation"],
            valid["gate"]["baseline"],
            {"receipt": True},
            valid["models"],
            commit="invalid",
            inputs={},
        )
    with pytest.raises(InvalidDemoEvidence, match="model identities"):
        build_evidence(
            valid["evaluation"],
            valid["gate"]["baseline"],
            {"receipt": True},
            {},
            commit=COMMIT,
            inputs={},
        )


def test_future_and_malformed_evidence_are_rejected(tmp_path: Path) -> None:
    future_root = tmp_path / "future"
    future = create_evidence(future_root, created_at="2026-09-11T00:00:00Z")
    with pytest.raises(InvalidDemoEvidence, match="stale"):
        load_verified_evidence(future, project_root=future_root, now=NOW, expected_commit=COMMIT)
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]")
    with pytest.raises(InvalidDemoEvidence, match="object"):
        load_verified_evidence(malformed, project_root=tmp_path, now=NOW, expected_commit=COMMIT)
