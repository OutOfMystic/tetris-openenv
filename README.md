---
title: Tetris OpenEnv
emoji: 🎮
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# Tetris OpenEnv

A Tetris RL environment for LLM agent training, built on OpenEnv 0.2.1.

LLM agents receive a text-based board representation and must choose spatial actions (left, right, rotate, drop) to play Tetris. Features combo scoring where clearing multiple lines simultaneously gives disproportionately higher rewards.

## Problem Statement

**Wild Card (#5)** - Teaching LLMs spatial reasoning through Tetris. The agent must interpret a 2D text grid and plan piece placements, a fundamentally non-linguistic task solved through language.

## Quick Start

```python
from tetris_env import TetrisEnvClient, TetrisAction

with TetrisEnvClient(base_url="https://VortexedSquirrel-tetris-env.hf.space") as env:
    result = env.reset(seed=42)
    while not result.done:
        action = TetrisAction(action="drop")
        result = env.step(action)
        print(f"Reward: {result.reward}, Score: {result.observation.score}")
```

## Actions

| Action | Description |
|---|---|
| `left` | Move piece left |
| `right` | Move piece right |
| `rotate_cw` | Rotate clockwise |
| `rotate_ccw` | Rotate counter-clockwise |
| `drop` | Hard drop to bottom |
| `down` | Soft drop one row |
| `noop` | Do nothing |

## Reward Structure

| Lines Cleared | Reward | Multiplier |
|---|---|---|
| 1 | +100 | x1 |
| 2 | +300 | x3 |
| 3 | +700 | x7 |
| 4 (Tetris!) | +1500 | x15 |

Penalties: -1/step, -2*height, -5*holes, -500 game over.

## Built With

- [OpenEnv 0.2.1](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/VortexedSquirrel/tetris-env)
