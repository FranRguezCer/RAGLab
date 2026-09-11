from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_ci_keeps_quality_compose_and_local_build_without_publication() -> None:
    workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check" in workflow
    assert "mypy src/raglab" in workflow
    assert "--cov-fail-under=85" in workflow
    assert "docker compose -f compose.yaml config --quiet" in workflow
    assert "docker build --tag raglab:test ." in workflow
    assert "docker/login-action" not in workflow
    assert "push: true" not in workflow
    assert not (WORKFLOWS / "release.yml").exists()


def test_third_party_actions_are_pinned_to_full_commit_shas() -> None:
    workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    references = re.findall(r"^\s*uses:\s*[^\s]+@([^\s#]+)", workflow, re.MULTILINE)
    assert references
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in references)


def test_vps_delivery_assets_are_removed() -> None:
    for relative in (
        "compose.production.yaml",
        "Dockerfile.portfolio",
        "docs/production-runbook.md",
        "scripts/deploy_portfolio.sh",
    ):
        assert not (ROOT / relative).exists()
