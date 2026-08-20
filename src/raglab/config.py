"""Project environment loading for command-line entry points."""

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(start: Path | None = None) -> bool:
    """Load the nearest project ``.env`` without replacing exported variables."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            return load_dotenv(dotenv_path=dotenv_path, override=False)
    return False
