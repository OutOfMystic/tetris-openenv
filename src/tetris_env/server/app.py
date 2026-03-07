"""
FastAPI application for Tetris OpenEnv.
Uses openenv-core create_app for standard routes.
"""

from openenv.core import create_app
from ..models import TetrisAction, TetrisObservation
from .environment import TetrisEnvironment

app = create_app(
    env=TetrisEnvironment,
    action_cls=TetrisAction,
    observation_cls=TetrisObservation,
    env_name="tetris-env",
)
