"""
Tetris Environment implementing OpenEnv Environment interface.
"""

import uuid
from typing import Optional, Any
from openenv.core import Environment
from ..models import TetrisAction, TetrisObservation, TetrisState
from .game_engine import TetrisEnv


class TetrisEnvironment(Environment[TetrisAction, TetrisObservation, TetrisState]):
    """
    Tetris RL environment for LLM agent training.

    LLM agents receive a text-based board representation and choose
    spatial actions (left, right, rotate, drop) to play Tetris.
    Features combo scoring where clearing multiple lines simultaneously
    gives disproportionately higher rewards.
    """

    def __init__(self):
        self._engine: Optional[TetrisEnv] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TetrisObservation:
        """Start a new Tetris episode."""
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._engine = TetrisEnv(seed=seed)
        result = self._engine.reset(seed=seed)
        return self._to_observation(result)

    def step(
        self,
        action: TetrisAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TetrisObservation:
        """Execute one action in Tetris."""
        if self._engine is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        result = self._engine.step(action.action.value)
        return self._to_observation(result)

    @property
    def state(self) -> TetrisState:
        """Return current episode state."""
        if self._engine is None:
            return TetrisState()
        return TetrisState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            score=self._engine.score,
            total_lines=self._engine.total_lines,
            done=self._engine.done,
        )

    def _to_observation(self, result: dict) -> TetrisObservation:
        """Convert engine result dict to TetrisObservation."""
        return TetrisObservation(
            board=result["board"],
            current_piece=result["current_piece"],
            current_piece_shape=result["current_piece_shape"],
            next_piece=result["next_piece"],
            next_piece_shape=result["next_piece_shape"],
            piece_x=result["piece_x"],
            piece_y=result["piece_y"],
            score=result["score"],
            total_lines=result["total_lines"],
            max_height=result["max_height"],
            holes=result["holes"],
            reward=result["reward"],
            done=result["done"],
        )
