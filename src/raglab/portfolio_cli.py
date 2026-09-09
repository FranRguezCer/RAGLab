"""ASGI entrypoint for the stateless public portfolio image."""

from raglab.config import load_project_env
from raglab.portfolio import create_portfolio_app

load_project_env()
app = create_portfolio_app()
