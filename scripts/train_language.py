"""
LANGUAGE TRAINING - Teach Themis to write English using TinyStories.

Frames language modeling as next-token prediction:
  given the recent text context, predict the next character/token.
The recurrent world-model state carries longer-range context, while the
encoder sees a sliding window of recent characters.
"""

import sys
import os
import torch
import json
import random

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.trainer import ActiveInferenceTrainer
from themis.training.replay_buffer import ReplayBuffer


# Sliding context window (characters) the encoder sees each step
CONTEXT_WINDOW = 64


def build_trajectories(stories, tokenizer, replay_buffer):
    """Turn each story into (context, next_token) training pairs."""
    eos_id = tokenizer.eos_id
    for story in stories:
        tokens = tokenizer.encode(story, add_special_tokens=False) + [eos_id]

        obs_list, actions_list, dones_list = [], [], []
        text_so_far = ""
        n = len(tokens)
        for i, tok in enumerate(tokens):
            # Context = last CONTEXT_WINDOW chars seen so far (or a space if empty)
            context = text_so_far[-CONTEXT_WINDOW:] if text_so_far else " "
            obs_list.append(context)
            actions_list.append(tok)
            dones_list.append(i == n - 1)
            text_so_far += tokenizer.decode([tok])

        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)


def main():
    print("=" * 80)
    print(" " * 25 + "LANGUAGE TRAINING (TinyStories)")
    print("=" * 80)
    print()

    config = ThemisConfig()
    # Lighter inner loops for speed; we rely on supervised next-token signal
    config.perception.n_iterations = 4
    config.training.learning_rate = 1e-3
    config.training.batch_size = 16
    config.training.replay_buffer_size = 300_000  # hold all language transitions
    # Emphasize next-token (policy) learning; keep a little VFE for the world model.
    # Without this, the large reconstruction loss drowns the language signal.
    config.training.vfe_weight = 0.02
    config.training.policy_loss_weight = 1.0

    device = config.resolve_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load prepared stories
    print("[1] Loading TinyStories...")
    with open("tinystories_prepared.json", "r", encoding="utf-8") as f:
        stories = json.load(f)
    print(f"Loaded {len(stories)} stories")
    print()

    # Build model
    print("[2] Initializing scaled model...")
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    tokenizer = orchestrator.markov_blanket.tokenizer
    n_params = sum(p.numel() for p in orchestrator.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print()

    # Build training data
    print("[3] Building next-token training pairs...")
    build_trajectories(stories, tokenizer, replay_buffer)
    print(f"Replay buffer: {len(replay_buffer):,} transitions")
    print()

    # Train
    print("[4] Training...")
    epochs = 12000
    seq_len = 16
    print(f"Epochs: {epochs} | seq_len: {seq_len} | batch: {config.training.batch_size}")
    print()

    best_loss = float('inf')
    loss_history = []

    for epoch in range(1, epochs + 1):
        obs_b, action_b, done_b = replay_buffer.sample_batch(
            batch_size=config.training.batch_size,
            seq_len=seq_len,
            device=device
        )
        metrics = trainer.train_step(obs_b, action_b, done_b)
        current = metrics['loss']
        loss_history.append(metrics['policy_loss'])  # track LANGUAGE signal
        if metrics['policy_loss'] < best_loss:
            best_loss = metrics['policy_loss']

        if epoch % 1000 == 0:
            path = "checkpoint_language.pt"
            try:
                orchestrator.save_checkpoint(path)
                if os.path.exists(path):
                    print(f"  💾 saved {path}")
            except Exception as e:
                print(f"  save error: {e}")

        if epoch % 200 == 0 or epoch == 1:
            avg = sum(loss_history[-200:]) / len(loss_history[-200:])
            # policy_loss is the next-token cross-entropy = the real language metric
            print(f"Epoch {epoch:05d}/{epochs} | Policy(CE): {metrics['policy_loss']:6.3f} | Avg: {avg:6.3f} | Best: {best_loss:6.3f} | VFE: {metrics['vfe']:8.1f}")

    # Final save
    path = "checkpoint_language.pt"
    orchestrator.save_checkpoint(path)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"\n✓ Final model saved: {path} ({size:.1f} MB)")
    print(f"✓ Best loss: {best_loss:.3f}")
    print()
    print("Next: chat with it using scripts/chat_language.py")


if __name__ == "__main__":
    main()
