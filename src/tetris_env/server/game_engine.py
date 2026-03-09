"""
Tetris Environment for OpenEnv.
Full game logic with combo scoring reward system.
"""
__version__ = "0.6.0"  # configurable height_breach_penalty per instance

import random
import copy
from typing import Optional

# Standard Tetris pieces (rotations handled by rotation logic)
PIECES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1],
          [1, 1]],
    "T": [[0, 1, 0],
          [1, 1, 1]],
    "S": [[0, 1, 1],
          [1, 1, 0]],
    "Z": [[1, 1, 0],
          [0, 1, 1]],
    "L": [[1, 0],
          [1, 0],
          [1, 1]],
    "J": [[0, 1],
          [0, 1],
          [1, 1]],
}

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Combo scoring: more lines cleared at once = disproportionately higher reward
LINE_REWARDS = {
    1: 100,
    2: 300,
    3: 700,
    4: 1500,  # "Tetris!" — the dream
}

STEP_PENALTY = -0.1
HOLE_PENALTY_MULT = -5
GAME_OVER_PENALTY = -50
HEIGHT_BREACH_THRESHOLD = 4
HEIGHT_BREACH_PENALTY = -50  # per level above threshold, decays with pieces_locked


def rotate_cw(piece: list[list[int]]) -> list[list[int]]:
    """Rotate piece 90 degrees clockwise."""
    rows = len(piece)
    cols = len(piece[0])
    rotated = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            rotated[c][rows - 1 - r] = piece[r][c]
    return rotated


def rotate_ccw(piece: list[list[int]]) -> list[list[int]]:
    """Rotate piece 90 degrees counter-clockwise."""
    rows = len(piece)
    cols = len(piece[0])
    rotated = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            rotated[cols - 1 - c][r] = piece[r][c]
    return rotated


class TetrisEnv:
    def __init__(self, seed: Optional[int] = None, height_breach_penalty: float = HEIGHT_BREACH_PENALTY):
        self.rng = random.Random(seed)
        self.height_breach_penalty = height_breach_penalty
        self.reset_state()

    def reset_state(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.board = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.total_lines = 0
        self.steps = 0
        self.done = False
        self.current_piece = None
        self.current_piece_name = ""
        self.current_x = 0
        self.current_y = 0
        self.next_piece_name = ""
        self.next_piece = None
        self.max_penalized_height = HEIGHT_BREACH_THRESHOLD
        self.pieces_locked = 0
        self._spawn_next()
        self._spawn_next()

    def _spawn_next(self):
        """Move next piece to current, generate new next piece."""
        self.current_piece = self.next_piece
        self.current_piece_name = self.next_piece_name
        self.next_piece_name = self.rng.choice(list(PIECES.keys()))
        self.next_piece = copy.deepcopy(PIECES[self.next_piece_name])

        if self.current_piece is not None:
            piece_width = len(self.current_piece[0])
            self.current_x = BOARD_WIDTH // 2 - piece_width // 2
            self.current_y = 0

            if not self._is_valid_position(self.current_piece, self.current_x, self.current_y):
                self.done = True

    def _is_valid_position(self, piece: list[list[int]], x: int, y: int) -> bool:
        """Check if piece at (x, y) doesn't collide with board or walls."""
        for row_idx, row in enumerate(piece):
            for col_idx, cell in enumerate(row):
                if cell == 0:
                    continue
                board_x = x + col_idx
                board_y = y + row_idx
                if board_x < 0 or board_x >= BOARD_WIDTH:
                    return False
                if board_y < 0 or board_y >= BOARD_HEIGHT:
                    return False
                if self.board[board_y][board_x] != 0:
                    return False
        return True

    def _lock_piece(self):
        """Lock current piece into the board."""
        for row_idx, row in enumerate(self.current_piece):
            for col_idx, cell in enumerate(row):
                if cell:
                    bx = self.current_x + col_idx
                    by = self.current_y + row_idx
                    if 0 <= by < BOARD_HEIGHT and 0 <= bx < BOARD_WIDTH:
                        self.board[by][bx] = 1

    def _clear_lines(self) -> int:
        """Clear completed lines. Returns number of lines cleared."""
        lines_cleared = 0
        new_board = []
        for row in self.board:
            if all(cell == 1 for cell in row):
                lines_cleared += 1
            else:
                new_board.append(row)

        # Add empty rows at the top
        while len(new_board) < BOARD_HEIGHT:
            new_board.insert(0, [0] * BOARD_WIDTH)

        self.board = new_board
        self.total_lines += lines_cleared
        return lines_cleared

    def _count_holes(self) -> int:
        """Count holes: empty cells with at least one filled cell above them."""
        holes = 0
        for col in range(BOARD_WIDTH):
            found_block = False
            for row in range(BOARD_HEIGHT):
                if self.board[row][col] == 1:
                    found_block = True
                elif found_block and self.board[row][col] == 0:
                    holes += 1
        return holes

    def _max_height(self) -> int:
        """Height of the tallest column."""
        for row in range(BOARD_HEIGHT):
            if any(cell == 1 for cell in self.board[row]):
                return BOARD_HEIGHT - row
        return 0

    def _drop_piece(self):
        """Hard drop: move piece down until it can't go further."""
        while self._is_valid_position(self.current_piece, self.current_x, self.current_y + 1):
            self.current_y += 1

    def get_board_with_piece(self) -> list[list[int]]:
        """Return board with current piece overlaid (for observation)."""
        display = copy.deepcopy(self.board)
        if self.current_piece and not self.done:
            for row_idx, row in enumerate(self.current_piece):
                for col_idx, cell in enumerate(row):
                    if cell:
                        bx = self.current_x + col_idx
                        by = self.current_y + row_idx
                        if 0 <= by < BOARD_HEIGHT and 0 <= bx < BOARD_WIDTH:
                            display[by][bx] = 2  # 2 = current piece
        return display

    def board_to_text(self) -> str:
        """Render board as text for LLM observation."""
        display = self.get_board_with_piece()
        symbols = {0: ".", 1: "#", 2: "@"}
        lines = []
        lines.append("+" + "-" * BOARD_WIDTH + "+")
        for row in display:
            line = "|" + "".join(symbols[c] for c in row) + "|"
            lines.append(line)
        lines.append("+" + "-" * BOARD_WIDTH + "+")
        return "\n".join(lines)

    def piece_to_text(self, piece: list[list[int]]) -> str:
        """Render a piece as text."""
        return "\n".join("".join("#" if c else "." for c in row) for row in piece)

    def step(self, action: str) -> dict:
        """
        Execute one action. Valid actions:
        - "left": move piece left
        - "right": move piece right
        - "rotate_cw": rotate clockwise
        - "rotate_ccw": rotate counter-clockwise
        - "drop": hard drop and lock
        - "down": soft drop one row
        - "noop": do nothing (piece falls one row)

        Returns dict with: observation, reward, done, info
        """
        if self.done:
            return self._make_result(0)

        self.steps += 1
        reward = STEP_PENALTY  # base penalty per step

        holes_before = self._count_holes()

        action = action.strip().lower()

        if action == "left":
            if self._is_valid_position(self.current_piece, self.current_x - 1, self.current_y):
                self.current_x -= 1
        elif action == "right":
            if self._is_valid_position(self.current_piece, self.current_x + 1, self.current_y):
                self.current_x += 1
        elif action == "rotate_cw":
            rotated = rotate_cw(self.current_piece)
            if self._is_valid_position(rotated, self.current_x, self.current_y):
                self.current_piece = rotated
        elif action == "rotate_ccw":
            rotated = rotate_ccw(self.current_piece)
            if self._is_valid_position(rotated, self.current_x, self.current_y):
                self.current_piece = rotated
        elif action == "drop":
            self._drop_piece()
        elif action == "down":
            if self._is_valid_position(self.current_piece, self.current_x, self.current_y + 1):
                self.current_y += 1
        elif action == "noop":
            pass

        # After action: try to move piece down (gravity)
        if action != "drop":
            if self._is_valid_position(self.current_piece, self.current_x, self.current_y + 1):
                self.current_y += 1
            else:
                # Can't move down — lock piece
                self._lock_piece()
                self.pieces_locked += 1
                lines = self._clear_lines()
                if lines > 0:
                    reward += LINE_REWARDS.get(lines, lines * 400)
                    self.score += LINE_REWARDS.get(lines, lines * 400)
                self._spawn_next()
        else:
            # Drop action: lock immediately
            self._lock_piece()
            self.pieces_locked += 1
            lines = self._clear_lines()
            if lines > 0:
                reward += LINE_REWARDS.get(lines, lines * 400)
                self.score += LINE_REWARDS.get(lines, lines * 400)
            self._spawn_next()

        # Penalty only for NEW holes created by this step
        new_holes = self._count_holes() - holes_before
        if new_holes > 0:
            reward += HOLE_PENALTY_MULT * new_holes

        # One-time penalty for each height level breached above threshold
        # Decays by 5 per piece locked: piece 0 → -50, piece 9 → -5, piece 10+ → 0
        current_height = self._max_height()
        if current_height > self.max_penalized_height:
            penalty_per_level = min(0, self.height_breach_penalty + 5 * self.pieces_locked)
            if penalty_per_level < 0:
                new_levels = current_height - self.max_penalized_height
                reward += penalty_per_level * new_levels
            self.max_penalized_height = current_height

        if self.done:
            reward += GAME_OVER_PENALTY

        return self._make_result(reward)

    def _make_result(self, reward: float) -> dict:
        """Build the observation/result dict."""
        return {
            "board": self.board_to_text(),
            "current_piece": self.current_piece_name,
            "current_piece_shape": self.piece_to_text(self.current_piece) if self.current_piece else "",
            "next_piece": self.next_piece_name,
            "next_piece_shape": self.piece_to_text(self.next_piece) if self.next_piece else "",
            "piece_x": self.current_x,
            "piece_y": self.current_y,
            "score": self.score,
            "total_lines": self.total_lines,
            "steps": self.steps,
            "max_height": self._max_height(),
            "holes": self._count_holes(),
            "reward": reward,
            "done": self.done,
        }

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset the environment. Returns initial observation."""
        self.reset_state(seed)
        return self._make_result(0)
