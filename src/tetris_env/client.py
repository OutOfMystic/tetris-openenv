"""
OpenEnv client for Tetris environment.
Used from Colab/training scripts to interact with the deployed environment.
"""

from typing import Dict, Any
from openenv.core import EnvClient
from openenv.core.env_client import StepResult
from .models import TetrisAction, TetrisObservation, TetrisState


class TetrisEnvClient(EnvClient[TetrisAction, TetrisObservation, TetrisState]):
    """
    Client for connecting to a remote Tetris OpenEnv server.

    Usage:
        with TetrisEnvClient(base_url="https://your-space.hf.space") as env:
            result = env.reset(seed=42)
            while not result.done:
                action = TetrisAction(action="drop")
                result = env.step(action)
                print(f"Reward: {result.reward}, Score: {result.observation.score}")
    """

    def _step_payload(self, action: TetrisAction) -> Dict[str, Any]:
        """Convert TetrisAction to JSON payload for the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TetrisObservation]:
        """Parse server response into StepResult with TetrisObservation."""
        obs_data = payload.get("observation", payload)
        reward = payload.get("reward")
        done = payload.get("done", False)
        obs = TetrisObservation(**obs_data, reward=reward, done=done)
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TetrisState:
        """Parse server state response into TetrisState."""
        return TetrisState(**payload)
