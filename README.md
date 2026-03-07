---
title: Tetris OpenEnv
emoji: 🎮
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# Tetris OpenEnv

A Tetris RL environment for LLM agent training, built on the OpenEnv spec.

LLM agents receive a text-based board representation and must choose spatial actions (left, right, rotate, drop) to play Tetris. Features combo scoring where clearing multiple lines simultaneously gives disproportionately higher rewards.

## API

- `POST /reset` — Start new episode, returns session_id + initial observation
- `POST /step/{session_id}` — Take an action, returns observation + reward + done
- `GET /state/{session_id}` — Get current state without acting
- `GET /info` — Environment metadata

## Reward Structure

| Lines Cleared | Reward | Multiplier |
|---|---|---|
| 1 | +100 | x1 |
| 2 | +300 | x3 |
| 3 | +700 | x7 |
| 4 (Tetris!) | +1500 | x15 |

Penalties: -1/step, -2*height, -5*holes, -500 game over.
