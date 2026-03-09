# Per-Piece Tetris Training Design

## Overview

Redesign training so the model sees the board before EVERY piece placement,
instead of outputting 100 blind actions from a single prompt.

## Key Parameters

- **Model**: Qwen2.5-3B-Instruct + LoRA
- **GPU**: L4
- **Games per iteration**: 8 (same seed, GRPO-style comparison)
- **Max steps per game**: 200 (total actions across all pieces)
- **Max tokens per piece**: 20 (if piece not placed -> forced drop + penalty)
- **Training iterations**: 100 (each with a new seed)
- **No history**: each model call = fresh prompt with current board only

## One Model Call (one piece)

1. Build prompt: system message + current board + current piece + next piece
2. Model generates up to 20 tokens (L/R/C/W/D/S)
3. Play actions one by one on the engine
4. Stop when piece locks (new piece spawns) or 20 tokens exhausted
5. If 20 tokens used and piece NOT placed: forced drop + penalty (-10)

### How to detect piece lock

After each `env.step()`, check if `current_piece` changed from the one
shown in the prompt. If it changed -> piece was placed, stop processing tokens.

## One Game (one playthrough)

```
seed = iteration_seed
env.reset(seed)
total_reward = 0
all_log_probs = []
steps = 0

while not game_over and steps < 200:
    board_state = env.get_state()
    current_piece = board_state['current_piece']

    prompt = build_prompt(board_state)  # fresh each time, no history
    tokens, log_probs = model_generate(prompt, max_tokens=20)

    piece_placed = False
    for token in tokens:
        action = token_to_action(token)
        result = env.step(action)
        total_reward += result['reward']
        total_reward -= 0.1 if action in ('left', 'right') else 0
        all_log_probs.append(log_probs[token])
        steps += 1

        if result['current_piece'] != current_piece:
            # Piece was placed, new piece spawned
            piece_placed = True
            total_reward += 1.0  # bonus for placing piece
            break

        if result['done']:
            game_over = True
            break

    if not piece_placed and not game_over:
        # Piece not placed in 20 tokens -> force drop + penalty
        env.step('drop')
        total_reward -= 10.0
        steps += 1
```

## Reward Structure

Per-step rewards from game engine (summed across ALL steps):
- Step penalty: -1 per step
- Line clears: +100/+300/+700/+1500 (1/2/3/4 lines)
- Height penalty: -2 * max_height (per step)
- Hole penalty: -5 * holes (per step)
- Game over: -500

Additional rewards added by training loop:
- L or R move: -0.1 (discourages aimless shuffling)
- Piece placed: +1.0 (encourages completing placements)
- Piece NOT placed in 20 tokens: -10.0 (forces learning to drop)
- Line clear bonus: +100 per line (1 line=+100, 2=+200, 3=+300, 4=+400)

Total game reward = sum of all the above across entire game.

## One Training Iteration

```
seed = random_seed_for_this_iteration

# Play 8 games with same seed
rewards = []
all_game_log_probs = []

for game_idx in range(8):
    reward, log_probs_sum = play_one_game(model, seed)
    rewards.append(reward)
    all_game_log_probs.append(log_probs_sum)

# GRPO-style advantage
rewards = torch.tensor(rewards)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# Policy gradient loss
loss = 0
for i in range(8):
    loss -= all_game_log_probs[i] * advantages[i]
loss = loss / 8

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Full Training

```
for iteration in range(100):
    seed = iteration  # deterministic but different each time
    train_one_iteration(model, seed)

    if iteration % 10 == 0:
        print(f"Iter {iteration}: avg_reward={mean}, std={std}")
```

## Action Token Mapping

Pre-compute token IDs for action characters:
```python
ACTION_TOKENS = {
    tokenizer.encode('L', add_special_tokens=False)[0]: 'left',
    tokenizer.encode('R', add_special_tokens=False)[0]: 'right',
    tokenizer.encode('C', add_special_tokens=False)[0]: 'rotate_cw',
    tokenizer.encode('W', add_special_tokens=False)[0]: 'rotate_ccw',
    tokenizer.encode('D', add_special_tokens=False)[0]: 'drop',
    tokenizer.encode('S', add_special_tokens=False)[0]: 'down',
}
```

During generation: mask logits to only allow these 6 tokens.
Sample from softmax over 6 logits -> get action + log_prob.

## Notebook Cell Structure

1. Install deps (peft, trl, accelerate, etc.)
2. Load Qwen2.5-3B-Instruct + LoRA
3. Download game_engine.py, define prompt builder
4. **Demo: untrained model plays one game** (show board after each piece)
5. Define `play_one_game()` and `train_one_iteration()`
6. Training loop (100 iterations)
7. Plot reward curve
8. **Demo: trained model plays one game** (compare with untrained)
9. Push model to HF Hub

## Time Estimate

- 3B model forward pass: ~0.05s per token on T4
- Per piece: ~20 tokens * 0.05s = ~1s (forward passes)
- Per game: ~30-50 pieces * 1s = ~40s
- Per iteration: 8 games (sequential) = ~320s OR batched = ~40-80s
- 100 iterations: ~70-130 min

Fits in a Colab T4 session (4h limit).

Note: batching 8 games in parallel requires handling variable-length episodes.
Simpler approach: run 8 games sequentially (~5 min/iteration, ~8h total).
Better approach: batch the forward passes across 8 games at each "step",
masking out finished games.
