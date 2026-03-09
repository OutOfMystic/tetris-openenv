# Per-Piece Tetris GRPO Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Colab notebook so the model sees the board before every piece, plays actions until piece locks, and learns via GRPO-style policy gradient over full games.

**Architecture:** Custom REINFORCE/GRPO loop (no GRPOTrainer). Two-phase per iteration: (1) rollout 8 games without grad, storing (prompt, actions) pairs; (2) recompute log_probs with grad, multiply by advantage, backward. This keeps memory low — only one piece's activations in memory at a time.

**Tech Stack:** transformers, peft (LoRA), torch, game_engine.py (local Tetris)

---

### Task 1: Cell 1 — Install Dependencies

**Files:**
- Modify: `tetris_training.ipynb` Cell 1

**Step 1: Write cell content**

```python
# Cell 1: Install dependencies
!pip install peft accelerate -q
```

Note: we no longer need `trl` (no GRPOTrainer), `openenv-core`, or `datasets`.

**Step 2: Run cell, verify no errors**

**Step 3: Commit**

```bash
git add tetris_training.ipynb
git commit -m "cell 1: minimal deps for custom training loop"
```

---

### Task 2: Cell 2 — Load Model + LoRA

**Files:**
- Modify: `tetris_training.ipynb` Cell 2

**Step 1: Write cell content**

```python
# Cell 2: Load Qwen2.5-3B-Instruct + LoRA
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Note: 3B model, r=16 (smaller than 7B's r=32), bf16.

**Step 2: Run cell, verify output shows trainable params**

**Step 3: Commit**

---

### Task 3: Cell 3 — Game Engine + Constants + Prompt Builder

**Files:**
- Modify: `tetris_training.ipynb` Cell 3 (replaces old TetrisClient cell)

**Step 1: Write cell content**

```python
# Cell 3: Game engine + constants + prompt builder
import random
import torch.nn.functional as F

# Download game engine
!wget -q -O game_engine.py https://raw.githubusercontent.com/OutOfMystic/tetris-openenv/main/src/tetris_env/server/game_engine.py
from game_engine import TetrisEnv

# === Constants ===
MAX_ACTIONS_PER_PIECE = 20
MAX_STEPS_PER_GAME = 200
GAMES_PER_ITER = 8
NUM_ITERATIONS = 100
TEMPERATURE = 0.7

# Training reward modifiers (on top of engine rewards)
LR_PENALTY = -0.1          # per L/R move
PIECE_PLACED_BONUS = 1.0   # per piece successfully placed
NO_PLACE_PENALTY = -10.0   # if 20 tokens exhausted without placing
LINE_CLEAR_BONUS = 100.0   # per line cleared (1=+100, 2=+200, 3=+300, 4=+400)

# Action mapping
ACTION_CHARS = ['L', 'R', 'C', 'W', 'D', 'S']
ACTION_TO_ENGINE = {
    'L': 'left', 'R': 'right', 'C': 'rotate_cw',
    'W': 'rotate_ccw', 'D': 'drop', 'S': 'down'
}

# Pre-compute token IDs for action characters
ACTION_TOKEN_IDS = []
for ch in ACTION_CHARS:
    ids = tokenizer.encode(ch, add_special_tokens=False)
    ACTION_TOKEN_IDS.append(ids[0])
    print(f"  '{ch}' -> token_id {ids[0]}")
ACTION_TOKEN_IDS = torch.tensor(ACTION_TOKEN_IDS, device=model.device)

# Token ID -> action char lookup
TOKEN_TO_CHAR = {ACTION_TOKEN_IDS[i].item(): ACTION_CHARS[i] for i in range(6)}

SYSTEM_PROMPT = """You are a Tetris AI. You see the board and current piece.
Output actions as single letters: L=left R=right C=rotate_cw W=rotate_ccw D=drop S=down
Place the piece to fill complete rows. Drop when positioned."""

def build_prompt(result):
    return f"""Board:
{result['board']}

Piece: {result['current_piece']} Next: {result['next_piece']}
Score: {result['score']} Lines: {result['total_lines']} Height: {result['max_height']} Holes: {result['holes']}

Your actions:"""

print("Game engine loaded. Action tokens mapped.")
```

**Step 2: Run cell, verify 6 token mappings printed**

**Step 3: Commit**

---

### Task 4: Cell 4 — Core Functions (play_one_game + train_one_iteration)

**Files:**
- Modify: `tetris_training.ipynb` Cell 4

This is the most critical cell. Two phases per iteration:
- **Rollout** (no grad): play 8 games, store (prompt_ids, action_token_ids) per piece
- **Update** (with grad): recompute log_probs, multiply by advantage, backward per piece

**Step 1: Write cell content**

```python
# Cell 4: Core training functions

def play_one_game(model, tokenizer, seed, temperature=TEMPERATURE):
    """
    Play a full Tetris game. Model sees board before each piece,
    generates up to 20 action tokens per piece.
    Returns reward and piece data for gradient computation.
    """
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)

    # Initial random offset (same as old prompt generation)
    rng = random.Random(seed)
    moves = rng.randint(0, 4)
    direction = rng.choice(["left", "right"])
    for _ in range(moves):
        if env.done:
            break
        env.step(direction)

    total_reward = 0.0
    total_steps = 0
    pieces_data = []

    while not env.done and total_steps < MAX_STEPS_PER_GAME:
        current_piece_name = env.current_piece_name
        lines_before = env.total_lines

        # Build fresh prompt for this piece
        result = env._make_result(0)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(result)},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        # Autoregressive generation with KV cache (no grad)
        input_ids = prompt_ids
        action_ids_list = []
        piece_placed = False
        past_kv = None

        with torch.no_grad():
            for _ in range(MAX_ACTIONS_PER_PIECE):
                if past_kv is None:
                    out = model(input_ids, use_cache=True)
                    past_kv = out.past_key_values
                else:
                    out = model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values

                logits = out.logits[:, -1, :]  # [1, vocab]

                # Mask to only 6 action tokens
                masked = torch.full_like(logits, float('-inf'))
                masked[0, ACTION_TOKEN_IDS] = logits[0, ACTION_TOKEN_IDS]
                probs = F.softmax(masked / temperature, dim=-1)

                # Sample
                token_id = torch.multinomial(probs, 1).item()
                action_ids_list.append(token_id)

                # Execute in engine
                action_char = TOKEN_TO_CHAR[token_id]
                action_name = ACTION_TO_ENGINE[action_char]
                step_result = env.step(action_name)

                total_reward += step_result['reward']
                total_steps += 1

                if action_name in ('left', 'right'):
                    total_reward += LR_PENALTY

                # Check if piece placed (new piece spawned)
                if env.current_piece_name != current_piece_name:
                    piece_placed = True
                    total_reward += PIECE_PLACED_BONUS
                    lines_cleared = env.total_lines - lines_before
                    if lines_cleared > 0:
                        total_reward += lines_cleared * LINE_CLEAR_BONUS
                    break

                if env.done:
                    break

                # Append token for next autoregressive step
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token_id]], device=model.device)
                ], dim=-1)

        # Force drop if piece not placed
        if not piece_placed and not env.done:
            env.step('drop')
            total_reward += NO_PLACE_PENALTY
            total_steps += 1
            lines_cleared = env.total_lines - lines_before
            if lines_cleared > 0:
                total_reward += lines_cleared * LINE_CLEAR_BONUS

        # Store for gradient computation
        if action_ids_list:
            pieces_data.append({
                'prompt_ids': prompt_ids.cpu(),
                'action_ids': torch.tensor(action_ids_list, dtype=torch.long),
            })

    return {
        'reward': total_reward,
        'pieces': pieces_data,
        'total_steps': total_steps,
        'total_lines': env.total_lines,
        'pieces_placed': len(pieces_data),
    }


def train_one_iteration(model, optimizer, seed, temperature=TEMPERATURE):
    """
    One GRPO iteration:
    1. Play 8 games (same seed) without grad
    2. Compute advantages
    3. Recompute log_probs with grad, apply policy gradient
    """
    # Phase 1: Rollout
    games = []
    for _ in range(GAMES_PER_ITER):
        game = play_one_game(model, tokenizer, seed, temperature)
        games.append(game)

    rewards = torch.tensor([g['reward'] for g in games], dtype=torch.float32)
    mean_r = rewards.mean().item()
    std_r = rewards.std().item()

    # Phase 2: Advantages (GRPO-style)
    if std_r < 1e-8:
        # All games got same reward — no learning signal
        return {'mean_reward': mean_r, 'std_reward': 0.0, 'loss': 0.0,
                'avg_steps': sum(g['total_steps'] for g in games) / GAMES_PER_ITER,
                'avg_lines': sum(g['total_lines'] for g in games) / GAMES_PER_ITER}

    advantages = ((rewards - rewards.mean()) / (rewards.std() + 1e-8)).tolist()

    # Phase 3: Update — recompute log_probs with grad
    optimizer.zero_grad()
    total_pieces = sum(len(g['pieces']) for g in games)
    total_loss = 0.0

    for game_idx, game in enumerate(games):
        adv = advantages[game_idx]
        for piece in game['pieces']:
            prompt = piece['prompt_ids'].to(model.device)
            actions = piece['action_ids'].to(model.device)
            if len(actions) == 0:
                continue

            # Teacher-forced forward pass: prompt + actions
            full_input = torch.cat([prompt.squeeze(0), actions]).unsqueeze(0)
            logits = model(full_input).logits

            # Log_probs at positions where actions were generated
            P = prompt.shape[-1]
            action_logits = logits[0, P-1 : P-1+len(actions), :]

            # Mask to action tokens, apply temperature
            masked = torch.full_like(action_logits, float('-inf'))
            masked[:, ACTION_TOKEN_IDS] = action_logits[:, ACTION_TOKEN_IDS]
            log_probs = F.log_softmax(masked / temperature, dim=-1)

            selected = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            piece_loss = -(selected.sum() * adv) / total_pieces

            piece_loss.backward()
            total_loss += piece_loss.item()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'mean_reward': mean_r,
        'std_reward': std_r,
        'loss': total_loss,
        'avg_steps': sum(g['total_steps'] for g in games) / GAMES_PER_ITER,
        'avg_lines': sum(g['total_lines'] for g in games) / GAMES_PER_ITER,
        'avg_pieces': sum(g['pieces_placed'] for g in games) / GAMES_PER_ITER,
    }

print("Training functions defined.")
```

**Step 2: Run cell, verify "Training functions defined." printed**

**Step 3: Quick smoke test**

```python
# Smoke test: play 1 game
test_game = play_one_game(model, tokenizer, seed=0)
print(f"Reward: {test_game['reward']:.1f}, Steps: {test_game['total_steps']}, "
      f"Pieces: {test_game['pieces_placed']}, Lines: {test_game['total_lines']}")
```

**Step 4: Commit**

---

### Task 5: Cell 5 — Demo Untrained Model

**Files:**
- Modify: `tetris_training.ipynb` Cell 5

**Step 1: Write cell content**

```python
# Cell 5: Demo — UNTRAINED model plays one game
print("=== UNTRAINED MODEL ===\n")

game = play_one_game(model, tokenizer, seed=42)

print(f"Total steps: {game['total_steps']}")
print(f"Pieces placed: {game['pieces_placed']}")
print(f"Lines cleared: {game['total_lines']}")
print(f"Game reward: {game['reward']:+.1f}")

# Show final board
env = TetrisEnv(seed=42)
env.reset(seed=42)
rng = random.Random(42)
moves = rng.randint(0, 4)
direction = rng.choice(["left", "right"])
for _ in range(moves):
    env.step(direction)

# Replay the actions to get final board
for piece in game['pieces']:
    for token_id in piece['action_ids'].tolist():
        action_name = ACTION_TO_ENGINE[TOKEN_TO_CHAR[token_id]]
        if not env.done:
            env.step(action_name)
    if not env.done:
        # Check if piece was placed; if not, force drop
        pass  # engine handles lock internally

print(f"\nFinal board:")
print(env.board_to_text())

untrained_reward = game['reward']
```

**Step 2: Run cell, verify board prints**

**Step 3: Commit**

---

### Task 6: Cell 6 — Training Loop

**Files:**
- Modify: `tetris_training.ipynb` Cell 6

**Step 1: Write cell content**

```python
# Cell 6: Training loop — 100 iterations
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-6,
    weight_decay=0.01,
)

history = []

print("Starting per-piece GRPO training...")
print(f"Config: {GAMES_PER_ITER} games/iter, max {MAX_STEPS_PER_GAME} steps, "
      f"max {MAX_ACTIONS_PER_PIECE} tokens/piece, {NUM_ITERATIONS} iterations\n")

for iteration in range(NUM_ITERATIONS):
    stats = train_one_iteration(model, optimizer, seed=iteration)
    history.append(stats)

    if iteration % 5 == 0 or iteration == NUM_ITERATIONS - 1:
        print(f"[Iter {iteration:3d}] "
              f"reward={stats['mean_reward']:+8.1f} "
              f"std={stats['std_reward']:6.1f} "
              f"loss={stats['loss']:7.3f} "
              f"steps={stats['avg_steps']:5.1f} "
              f"lines={stats['avg_lines']:4.1f} "
              f"pieces={stats['avg_pieces']:4.1f}")

print("\nTraining complete!")
```

**Step 2: Run cell, verify reward logs appear**

**Step 3: Commit**

---

### Task 7: Cell 7 — Plot Reward Curve

**Files:**
- Modify: `tetris_training.ipynb` Cell 7

**Step 1: Write cell content**

```python
# Cell 7: Plot reward curve
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

iters = range(len(history))

axes[0].plot(iters, [h['mean_reward'] for h in history])
axes[0].set_title('Mean Reward per Iteration')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Reward')

axes[1].plot(iters, [h['avg_lines'] for h in history])
axes[1].set_title('Avg Lines Cleared')
axes[1].set_xlabel('Iteration')

axes[2].plot(iters, [h['loss'] for h in history])
axes[2].set_title('Policy Gradient Loss')
axes[2].set_xlabel('Iteration')

plt.tight_layout()
plt.savefig('reward_curve.png', dpi=150)
plt.show()
```

**Step 2: Run, verify plots**

**Step 3: Commit**

---

### Task 8: Cell 8 — Demo Trained Model

**Files:**
- Modify: `tetris_training.ipynb` Cell 8

**Step 1: Write cell content**

```python
# Cell 8: Demo — TRAINED model plays 3 games
print("=== TRAINED MODEL ===\n")

trained_rewards = []
for seed in [42, 123, 7]:
    game = play_one_game(model, tokenizer, seed=seed)
    print(f"Seed {seed}: reward={game['reward']:+.1f}, "
          f"steps={game['total_steps']}, lines={game['total_lines']}, "
          f"pieces={game['pieces_placed']}")
    trained_rewards.append(game['reward'])

avg_trained = sum(trained_rewards) / len(trained_rewards)
print(f"\n{'='*50}")
print(f"UNTRAINED reward (seed=42): {untrained_reward:+.1f}")
print(f"TRAINED avg reward (3 games): {avg_trained:+.1f}")
print(f"Improvement: {avg_trained - untrained_reward:+.1f}")
print('='*50)
```

**Step 2: Run, verify comparison**

**Step 3: Commit**

---

### Task 9: Cell 9 — Push to HF Hub

**Files:**
- Modify: `tetris_training.ipynb` Cell 9

**Step 1: Write cell content**

```python
# Cell 9: Push trained model to HF Hub
model.push_to_hub("VortexedSquirrel/tetris-agent-grpo")
tokenizer.push_to_hub("VortexedSquirrel/tetris-agent-grpo")
print("Model pushed to https://huggingface.co/VortexedSquirrel/tetris-agent-grpo")
```

**Step 2: Commit + push**

```bash
git add tetris_training.ipynb
git commit -m "rewrite notebook: per-piece GRPO training on 3B"
git push origin main
```

---

## Key Implementation Details

### Piece lock detection
After `env.step()`, check `env.current_piece_name != current_piece_name`.
The engine calls `_spawn_next()` when a piece locks, changing the current piece.

### Two-phase gradient computation
- **Rollout phase**: `torch.no_grad()`, KV cache for fast autoregressive generation
- **Update phase**: teacher-forced forward pass (prompt + actions in one call),
  `piece_loss.backward()` after each piece to keep memory low

### Memory-safe gradient accumulation
Each piece does its own `.backward()` adding to accumulated gradients.
Only one piece's computation graph in memory at a time.
`optimizer.step()` called once after all 8 games processed.

### Time estimate (L4, 3B model)
- Rollout: 8 games × ~200 steps × ~0.02s/token ≈ 32s
- Update: 8 games × ~30 pieces × ~0.1s/piece ≈ 24s
- Total per iteration: ~56s
- 100 iterations: ~93 min ≈ 1.5 hours
