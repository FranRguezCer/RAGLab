from __future__ import annotations

import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).parents[1]


def test_production_compose_is_stateless_and_digest_driven() -> None:
    compose = (ROOT / "compose.production.yaml").read_text()

    assert "RAGLAB_PORTFOLIO_IMAGE" in compose
    assert "TRAEFIK_IMAGE" in compose
    assert "postgres:" not in compose
    assert "ollama:" not in compose
    assert "gpus:" not in compose
    assert "traefik-certificates:" in compose


def test_deploy_restores_compose_and_release_state() -> None:
    deploy = (ROOT / "scripts/deploy_portfolio.sh").read_text()

    assert "sha256:[0-9a-f]{64}" in deploy
    assert "config --images" in deploy
    assert 'services" == $\'portfolio\\nproxy\'' in deploy
    assert 'install -m 0644 "$ROLLBACK_DIR/compose.production.yaml" "$LIVE_COMPOSE"' in deploy
    assert 'install -m 0600 "$ROLLBACK_DIR/release.env" "$RELEASE_ENV"' in deploy
    assert "--wait --remove-orphans" in deploy


def test_public_smoke_uses_only_demo_endpoints() -> None:
    smoke = (ROOT / "scripts/smoke_portfolio.py").read_text()

    assert "/health/ready" in smoke
    assert "/v1/demo/status" in smoke
    assert "/v1/demo/cases" in smoke
    assert "/v1/query" not in smoke
    assert "RAGLAB_DEMO_TOKEN" not in smoke


def test_portfolio_image_is_bound_to_exact_evidence_bytes() -> None:
    dockerfile = (ROOT / "Dockerfile.portfolio").read_text()
    workflow = (ROOT / ".github/workflows/release.yml").read_text()

    assert "ARG RAGLAB_DEMO_RELEASE_SHA256" in dockerfile
    assert "RAGLAB_DEMO_RELEASE_SHA256=${RAGLAB_DEMO_RELEASE_SHA256}" in dockerfile
    assert "sha256sum demo-release.json" in workflow
    assert "RAGLAB_DEMO_RELEASE_SHA256=${{ steps.evidence.outputs.sha256 }}" in workflow


def test_failed_deploy_restores_previous_compose_and_image(tmp_path: pathlib.Path) -> None:
    deploy_root = tmp_path / "server"
    deploy_root.mkdir()
    (deploy_root / ".env").write_text("RAGLAB_DOMAIN=example.test\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"config --images"* ]]; then
  printf '%s\\n' \\
    'traefik@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \\
    'portfolio@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
elif [[ "$*" == *"config --services"* ]]; then
  printf '%s\\n' proxy portfolio
elif [[ "$*" == *"exec -T portfolio"* && -f "$FAIL_ONCE" ]]; then
  rm "$FAIL_ONCE"
  exit 1
fi
"""
    )
    fake_docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RAGLAB_DEPLOY_ROOT": str(deploy_root),
        "FAIL_ONCE": str(tmp_path / "fail-once"),
    }
    deploy = ROOT / "scripts/deploy_portfolio.sh"
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    failed = tmp_path / "failed.yaml"
    first.write_text("release: first\n")
    second.write_text("release: second\n")
    failed.write_text("release: failed\n")

    def run(
        candidate: pathlib.Path, digit: str, build_digit: str
    ) -> subprocess.CompletedProcess[str]:
        image = f"portfolio@sha256:{digit * 64}"
        build = build_digit * 40
        return subprocess.run(
            ["bash", str(deploy), str(candidate), image, build],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    assert run(first, "1", "a").returncode == 0
    assert run(second, "2", "b").returncode == 0
    pathlib.Path(env["FAIL_ONCE"]).touch()

    result = run(failed, "3", "c")

    assert result.returncode == 1
    assert (deploy_root / "compose.production.yaml").read_text() == second.read_text()
    assert (deploy_root / ".release.env").read_text() == (
        f"RAGLAB_PORTFOLIO_IMAGE=portfolio@sha256:{'2' * 64}\n"
        f"RAGLAB_BUILD_SHA={'b' * 40}\n"
    )
