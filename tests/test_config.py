from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest

from raglab.config import load_project_env


def test_load_project_env_reads_nearest_parent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "nested" / "directory"
    nested.mkdir(parents=True)
    (tmp_path / ".env").write_text("RAGLAB_DSN=postgresql://dotenv\n", encoding="utf-8")
    monkeypatch.delenv("RAGLAB_DSN", raising=False)

    assert load_project_env(nested) is True
    assert os.environ["RAGLAB_DSN"] == "postgresql://dotenv"


def test_load_project_env_preserves_exported_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("RAGLAB_DSN=postgresql://dotenv\n", encoding="utf-8")
    monkeypatch.setenv("RAGLAB_DSN", "postgresql://exported")

    assert load_project_env(tmp_path) is True
    assert os.environ["RAGLAB_DSN"] == "postgresql://exported"


def test_load_project_env_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGLAB_DSN", raising=False)

    assert load_project_env(tmp_path) is False
    assert "RAGLAB_DSN" not in os.environ


@pytest.mark.parametrize(
    "module_name",
    ["raglab.cli", "raglab.retrieval.cli", "raglab.generation.cli"],
)
def test_cli_entrypoint_loads_environment_before_main(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module: ModuleType = importlib.import_module(module_name)
    events: list[str] = []

    def load_environment() -> bool:
        events.append("load")
        return True

    def main() -> int:
        events.append("main")
        return 0

    monkeypatch.setattr(module, "load_project_env", load_environment)
    monkeypatch.setattr(module, "main", main)

    assert module.entrypoint() == 0
    assert events == ["load", "main"]
