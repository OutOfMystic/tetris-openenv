"""
FastAPI server for Tetris OpenEnv environment.
Supports multiple concurrent sessions via session_id.
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from environment import TetrisEnv
from models import (
    TetrisAction,
    TetrisObservation,
    StepResult,
    ResetResult,
    ResetRequest,
    EnvInfo,
)

app = FastAPI(
    title="Tetris OpenEnv",
    description="Tetris RL environment with combo scoring for LLM agent training",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session store: session_id -> TetrisEnv
sessions: dict[str, TetrisEnv] = {}


def _dict_to_observation(d: dict) -> TetrisObservation:
    return TetrisObservation(
        board=d["board"],
        current_piece=d["current_piece"],
        current_piece_shape=d["current_piece_shape"],
        next_piece=d["next_piece"],
        next_piece_shape=d["next_piece_shape"],
        piece_x=d["piece_x"],
        piece_y=d["piece_y"],
        score=d["score"],
        total_lines=d["total_lines"],
        steps=d["steps"],
        max_height=d["max_height"],
        holes=d["holes"],
    )


@app.get("/")
async def root():
    return {"status": "ok", "env": "tetris-env", "version": "0.1.0"}


@app.get("/info")
async def info() -> EnvInfo:
    return EnvInfo()


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> dict:
    """Create a new session and reset the environment."""
    session_id = str(uuid.uuid4())
    env = TetrisEnv(seed=request.seed)
    sessions[session_id] = env
    result = env.reset(seed=request.seed)
    obs = _dict_to_observation(result)
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/step/{session_id}")
async def step(session_id: str, action: TetrisAction) -> StepResult:
    """Execute one step in the environment."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /reset first.")
    env = sessions[session_id]
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    result = env.step(action.action.value)
    obs = _dict_to_observation(result)

    return StepResult(
        observation=obs,
        reward=result["reward"],
        done=result["done"],
    )


@app.get("/state/{session_id}")
async def state(session_id: str) -> dict:
    """Get current state without taking an action."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    env = sessions[session_id]
    result = env._make_result(0)
    obs = _dict_to_observation(result)
    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
        "done": env.done,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session to free memory."""
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/sessions")
async def list_sessions():
    """List active sessions (for debugging)."""
    return {
        "count": len(sessions),
        "sessions": [
            {"id": sid, "done": env.done, "score": env.score, "steps": env.steps}
            for sid, env in sessions.items()
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
