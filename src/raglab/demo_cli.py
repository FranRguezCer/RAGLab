"""ASGI application entrypoint used by Uvicorn."""

from raglab.config import load_project_env
from raglab.demo import create_app

load_project_env()
app = create_app()
