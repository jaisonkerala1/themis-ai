"""
COMBINED FOUNDATION TRAINING for the 200M Themis model.

Trains ONE model on math + QA + facts + English stories together, so it learns
everything in a single foundation (no separate models, no catastrophic forgetting).

AUTO-RESUME: if checkpoint_combined.pt already exists, it loads those weights and
continues training. This means:
  - Colab disconnects don't waste progress (rerun -> continues)
  - Future skill additions = put new data in combined_dataset.json + rerun (continues)
"""

import sys
import os
import torch
import json

sys.path.append(os.getcwd())

from themis.config import ThemisConfig
from themis.layers.orchestrator import Orchestrator
from themis.training.trainer import ActiveInferenceTrainer
from themis.training.replay_buffer import ReplayBuffer

CONTEXT_WINDOW = 64
CHECKPOINT = "checkpoint_combined.pt"


def build_trajectories(texts, tokenizer, replay_buffer):
    eos_id = tokenizer.eos_id
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_id]
        obs_list, actions_list, dones_list = [], [], []
        so_far = ""
        n = len(tokens)
        for i, tok in enumerate(tokens):
            context = so_far[-CONTEXT_WINDOW:] if so_far else " "
            obs_list.append(context)
            actions_list.append(tok)
            dones_list.append(i == n - 1)
            so_far += tokenizer.decode([tok])
        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)


def main():
    print("=" * 80)
    print(" " * 22 + "COMBINED FOUNDATION TRAINING (200M)")
    print("=" * 80)
    print()

    config = ThemisConfig()
    config.perception.n_iterations = 4
    config.training.learning_rate = 5e-4
    config.training.batch_size = 8           # 200M model -> small batch for T4 memory
    config.training.replay_buffer_size = 10_000_000
    config.training.vfe_weight = 0.02
    config.training.policy_loss_weight = 1.0
    config.training.freeze_encoder = False   # train the transformer (real 200M learns)

    device = config.resolve_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("[1] Loading combined dataset...")
    with open("combined_dataset.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"Loaded {len(texts):,} text lines")
    print()

    print("[2] Initializing 200M model...")
    orchestrator = Orchestrator(config)

    # AUTO-RESUME: continue from existing checkpoint if present
    start_note = "fresh (from scratch)"
    if os.path.exists(CHECKPOINT):
        try:
            ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
            orchestrator.load_state_dict(ckpt["model_state"])
            start_note = f"RESUMED from {CHECKPOINT}"
        except Exception as e:
            print(f"  (could not resume: {e}) - starting fresh")

    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    tokenizer = orchestrator.markov_blanket.tokenizer
    n_params = sum(p.numel() for p in orchestrator.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Start: {start_note}")
    print()

    print("[3] Building next-token training pairs...")
    build_trajectories(texts, tokenizer, replay_buffer)
    print(f"Replay buffer: {len(replay_buffer):,} transitions")
    print()

    print("[4] Training...")
    epochs = 40000
    seq_len = 8          # shorter BPTT window -> less activation memory for the 200M model
    print(f"Epochs: {epochs} | seq_len: {seq_len} | batch: {config.training.batch_size}")

    drive_dir = "/content/drive/MyDrive/themis"
    drive_available = os.path.isdir(drive_dir)
    if drive_available:
        print(f"Drive detected - checkpoints copied to {drive_dir}")
    print()

    best = float('inf')
    hist = []
    for epoch in range(1, epochs + 1):
        obs_b, act_b, done_b = replay_buffer.sample_batch(
            batch_size=config.training.batch_size, seq_len=seq_len, device=device)
        m = trainer.train_step(obs_b, act_b, done_b)
        hist.append(m['policy_loss'])
        if m['policy_loss'] < best:
            best = m['policy_loss']

        if epoch % 1000 == 0:
            try:
                orchestrator.save_checkpoint(CHECKPOINT)
                if drive_available and os.path.exists(CHECKPOINT):
                    import shutil
                    shutil.copy(CHECKPOINT, os.path.join(drive_dir, CHECKPOINT))
                print(f"  💾 saved {CHECKPOINT}")
            except Exception as e:
                print(f"  save error: {e}")

        if epoch % 200 == 0 or epoch == 1:
            avg = sum(hist[-200:]) / len(hist[-200:])
            print(f"Epoch {epoch:05d}/{epochs} | Policy(CE): {m['policy_loss']:6.3f} | Avg: {avg:6.3f} | Best: {best:6.3f}")

    orchestrator.save_checkpoint(CHECKPOINT)
    if drive_available:
        import shutil
        shutil.copy(CHECKPOINT, os.path.join(drive_dir, CHECKPOINT))
    print(f"\n✓ Done. Best CE: {best:.3f}. Saved {CHECKPOINT}")


if __name__ == "__main__":
    main()
