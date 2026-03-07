"""
Pydantic models for OpenEnv Tetris environment.
Follows OpenEnv 0.2 spec: Action, Observation, StepResult.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ActionType(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    DROP = "drop"
    DOWN = "down"
    NOOP = "noop"


class TetrisAction(BaseModel):
    action: ActionType = Field(
        ...,
        description="Action to perform. One of: left, right, rotate_cw, rotate_ccw, drop, down, noop"
    )


class TetrisObservation(BaseModel):
    board: str = Field(..., description="Text representation of the board (10x20 grid)")
    current_piece: str = Field(..., description="Name of current piece (I, O, T, S, Z, L, J)")
    current_piece_shape: str = Field(..., description="Text shape of current piece")
    next_piece: str = Field(..., description="Name of next piece")
    next_piece_shape: str = Field(..., description="Text shape of next piece")
    piece_x: int = Field(..., description="Current piece X position")
    piece_y: int = Field(..., description="Current piece Y position")
    score: int = Field(..., description="Current score")
    total_lines: int = Field(..., description="Total lines cleared")
    steps: int = Field(..., description="Number of steps taken")
    max_height: int = Field(..., description="Height of tallest column")
    holes: int = Field(..., description="Number of holes in the board")


class StepResult(BaseModel):
    observation: TetrisObservation
    reward: float = Field(..., description="Reward for this step")
    done: bool = Field(..., description="Whether the episode is over")


class ResetResult(BaseModel):
    observation: TetrisObservation


class ResetRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Optional random seed for reproducibility")


class EnvInfo(BaseModel):
    name: str = "tetris-env"
    description: str = "Tetris environment for LLM agent training with combo scoring"
    version: str = "0.1.0"
    action_space: list[str] = ["left", "right", "rotate_cw", "rotate_ccw", "drop", "down", "noop"]
    observation_format: str = "text"
    board_size: str = "10x20"
    reward_structure: dict = {
        "1_line": 100,
        "2_lines": 300,
        "3_lines": 700,
        "4_lines_tetris": 1500,
        "step_penalty": -1,
        "height_penalty": "-2 * max_height",
        "hole_penalty": "-5 * holes",
        "game_over": -500,
    }
