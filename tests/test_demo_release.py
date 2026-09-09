from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_evaluation_gate import _baseline, _run

from raglab.demo_release import build_demo_release, main

BUILD_SHA = "a" * 40
IMAGE_DIGEST = "sha256:" + "b" * 64
WORKFLOW_URL = "https://github.com/example/raglab/actions/runs/123"
MODELS = {
    "embedding": {"name": "embed", "digest": "sha256:" + "c" * 64},
    "generation": {"name": "generate", "digest": "sha256:" + "d" * 64},
    "reranker": {"name": "reranker", "revision": "commit-sha"},
}


def _build(**changes: object) -> dict[str, object]:
    arguments = {
        "build_sha": BUILD_SHA, "image_digest": IMAGE_DIGEST,
        "workflow_url": WORKFLOW_URL, "created_at": "2026-09-08T12:00:00Z", **changes,
    }
    run = _run()
    run["metadata"] = {"dataset_sha256": "a" * 64, "build": BUILD_SHA,
                       "image_digest": IMAGE_DIGEST}
    return build_demo_release(
        run, _baseline(), {"schema_version": 1, "source_count": 9}, MODELS,
        **arguments,  # type: ignore[arg-type]
    )


def test_builder_embeds_complete_verified_release_evidence() -> None:
    artifact = _build()
    assert artifact["schema_version"] == 1
    assert artifact["release"]["image_digest"] == IMAGE_DIGEST  # type: ignore[index]
    assert artifact["evaluation"]["traces"] == _run()["traces"]  # type: ignore[index]
    assert artifact["ingestion_receipt"]["source_count"] == 9  # type: ignore[index]
    assert artifact["gate"] == {"passed": True, "baseline": _baseline()}


def test_builder_rejects_unresolved_models_and_failed_evaluation() -> None:
    run = _build()["evaluation"]
    with pytest.raises(ValueError, match="resolved digest or revision"):
        build_demo_release(
            run, _baseline(), {"receipt": True}, {"generation": {"name": "model"}},  # type: ignore[arg-type]
            build_sha=BUILD_SHA, image_digest=IMAGE_DIGEST, workflow_url=WORKFLOW_URL,
            created_at="2026-09-08T12:00:00Z",
        )


def test_builder_binds_release_identity_to_evaluated_container() -> None:
    with pytest.raises(ValueError, match="evaluation build does not match"):
        _build(build_sha="c" * 40)
    with pytest.raises(ValueError, match="evaluation image does not match"):
        _build(image_digest="sha256:" + "e" * 64)
    run = _build()["evaluation"]
    run["traces"] = []
    with pytest.raises(ValueError, match="cannot be empty"):
        build_demo_release(
            run, _baseline(), {"receipt": True}, MODELS, build_sha=BUILD_SHA,  # type: ignore[arg-type]
            image_digest=IMAGE_DIGEST, workflow_url=WORKFLOW_URL,
            created_at="2026-09-08T12:00:00Z",
        )


def test_cli_atomically_writes_demo_release(tmp_path: Path) -> None:
    inputs = {"evaluation": _build()["evaluation"], "baseline": _baseline(),
              "receipt": {"schema_version": 1}, "models": MODELS}
    paths: dict[str, Path] = {}
    for name, value in inputs.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    output = tmp_path / "nested" / "demo-release.json"
    assert main([
        "--evaluation", str(paths["evaluation"]), "--baseline", str(paths["baseline"]),
        "--ingestion-receipt", str(paths["receipt"]), "--models", str(paths["models"]),
        "--build-sha", BUILD_SHA, "--image-digest", IMAGE_DIGEST,
        "--workflow-url", WORKFLOW_URL, "--created-at", "2026-09-08T12:00:00Z",
        "--output", str(output),
    ]) == 0
    assert json.loads(output.read_text())["release"]["build_sha"] == BUILD_SHA


def test_cli_fails_closed_for_invalid_input_json(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json")

    with pytest.raises(SystemExit) as error:
        main([
            "--evaluation", str(invalid), "--baseline", str(invalid),
            "--ingestion-receipt", str(invalid), "--models", str(invalid),
            "--build-sha", BUILD_SHA, "--image-digest", IMAGE_DIGEST,
            "--workflow-url", WORKFLOW_URL, "--output", str(tmp_path / "release.json"),
        ])

    assert error.value.code == 1
