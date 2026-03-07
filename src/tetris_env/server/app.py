"""
FastAPI application for Tetris OpenEnv.
Uses openenv-core create_app for standard routes.
"""

from pathlib import Path
from fastapi.responses import HTMLResponse
from openenv.core import create_app
from ..models import TetrisAction, TetrisObservation
from .environment import TetrisEnvironment

app = create_app(
    env=TetrisEnvironment,
    action_cls=TetrisAction,
    observation_cls=TetrisObservation,
    env_name="tetris-env",
)

# Serve custom Tetris UI at /play
STATIC_DIR = Path(__file__).parent / "static"

@app.get("/play", response_class=HTMLResponse)
async def play():
    return (STATIC_DIR / "play.html").read_text(encoding="utf-8")
