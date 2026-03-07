"""
OpenEnv-compatible models for Tetris environment.
"""

from enum import Enum
from typing import Optional
from pydantic import Field
from openenv.core import Action, Observation, State


class ActionType(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    DROP = "drop"
    DOWN = "down"
    NOOP = "noop"


class TetrisAction(Action):
    action: ActionType = Field(
        ...,
        description="Action to perform: left, right, rotate_cw, rotate_ccw, drop, down, noop",
    )


class TetrisObservation(Observation):
    # Observation base already has: done, reward, metadata
    board: str = Field(..., description="Text representation of the 10x20 board")
    current_piece: str = Field(..., description="Current piece name (I, O, T, S, Z, L, J)")
    current_piece_shape: str = Field("", description="Text shape of current piece")
    next_piece: str = Field(..., description="Next piece name")
    next_piece_shape: str = Field("", description="Text shape of next piece")
    piece_x: int = Field(0, description="Current piece X position")
    piece_y: int = Field(0, description="Current piece Y position")
    score: int = Field(0, description="Current score")
    total_lines: int = Field(0, description="Total lines cleared")
    max_height: int = Field(0, description="Height of tallest column")
    holes: int = Field(0, description="Number of holes in the board")


class TetrisState(State):
    # State base already has: episode_id, step_count
    score: int = Field(default=0, description="Current score")
    total_lines: int = Field(default=0, description="Total lines cleared")
    done: bool = Field(default=False, description="Whether episode is over")
