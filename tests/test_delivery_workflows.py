from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ci_gates_main_publication_on_quality() -> None:
    workflow = _workflow("ci.yml")

    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert "needs: quality" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "--cov-fail-under=85" in workflow
    assert "docker compose -f compose.production.yaml config --quiet" in workflow
    assert "cache-to: type=gha" in workflow


def test_release_keeps_pull_requests_off_gpu_and_requires_approval() -> None:
    workflow = _workflow("release.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "runs-on: [self-hosted, raglab-gpu]" in workflow
    assert "Verify successful CI run" in workflow
    assert "RAGLAB_BUILD_SHA:" in workflow
    assert "RAGLAB_IMAGE_DIGEST:" in workflow
    assert "if: always()" in workflow
    assert "down --volumes --remove-orphans" in workflow
    assert "platforms: linux/amd64,linux/arm64" in workflow
    assert "environment: production" in workflow


def test_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    workflows = _workflow("ci.yml") + _workflow("release.yml")
    action_references = re.findall(
        r"^\s*uses:\s*[^\s]+@([^\s#]+)", workflows, re.MULTILINE
    )

    assert action_references
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_references)


def test_disconnected_publish_and_deploy_workflows_are_removed() -> None:
    assert not (WORKFLOWS / "publish.yml").exists()
    assert not (WORKFLOWS / "deploy.yml").exists()
