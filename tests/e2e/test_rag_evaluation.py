from __future__ import annotations

import os
from pathlib import Path

import pytest

from raglab.evaluation import EvaluationApplication, load_manifest
from raglab.evaluation.runtime import LiveEvaluationExecutor

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    os.getenv("RAGLAB_RUN_EVALUATION_E2E") != "1",
    reason="RAGLAB_RUN_EVALUATION_E2E=1 is not set",
)
def test_real_core_evaluation(tmp_path: Path) -> None:
    executor = LiveEvaluationExecutor(
        dsn=os.environ.get(
            "RAGLAB_TEST_DSN", "postgresql://raglab:raglab@127.0.0.1:5432/raglab"
        )
    )
    run = EvaluationApplication(executor, artifact_dir=tmp_path).run(load_manifest())
    assert run["status"] == "complete"
    assert len(run["cases"]) == 12
