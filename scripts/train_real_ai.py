"""
REAL AI TRAINING - Proper Dataset, Proper Learning
This will create an AI that actually generalizes!
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


def load_dataset(filename):
    """Load the comprehensive dataset"""
    with open(filename, 'r') as f:
        return json.load(f)


def run_real_training():
    print("=" * 80)
    print(" " * 25 + "REAL AI TRAINING")
    print(" " * 20 + "Building Intelligence from Scratch")
    print("=" * 80)
    print()
    
    # Configuration
    config = ThemisConfig()
    config.perception.n_iterations = 4
    config.planning.n_candidate_policies = 4
    config.planning.n_policy_samples = 2
    config.planning.planning_horizon = 2
    config.training.learning_rate = 1e-3
    config.training.batch_size = 32  # Increased for better gradients
    
    device = config.resolve_device()
    print(f"✓ Device: {device}")
    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print()
    
    # Load dataset
    print("[PHASE 1] Loading Training Dataset...")
    dataset = load_dataset("training_dataset_large.json")
    print(f"✓ Loaded {len(dataset)} tasks")
    
    # Show statistics
    categories = {}
    for item in dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nDataset Coverage:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} tasks")
    print()
    
    # Initialize models
    print("[PHASE 2] Initializing AI Architecture...")
    orchestrator = Orchestrator(config)
    trainer = ActiveInferenceTrainer(config, orchestrator)
    replay_buffer = ReplayBuffer(config)
    tokenizer = orchestrator.markov_blanket.tokenizer
    print("✓ 7-layer Active Inference architecture ready")
    print("✓ ~3.5M parameters initialized")
    print()
    
    # Collect training episodes
    print("[PHASE 3] Collecting Expert Demonstrations...")
    num_episodes = 3000  # Each task seen ~20 times
    print(f"Collecting {num_episodes} episodes...")
    print("(This trains the AI to recognize patterns)")
    print()
    
    for episode in range(num_episodes):
        # Sample random task
        task = random.choice(dataset)
        question = task['question']
        answer = task['answer']

        # Tokenize answer and append EOS so the model learns WHEN TO STOP
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        answer_tokens = answer_tokens + [tokenizer.eos_id]

        # Build aligned (context, action) pairs with PROPER accumulation.
        # context_i is the full text the model sees BEFORE producing token_i:
        #   token 0  <- "question "
        #   token 1  <- "question <tok0>"
        #   ...
        #   EOS      <- "question <full answer>"
        obs_list = []
        actions_list = []
        dones_list = []

        generated = ""
        n = len(answer_tokens)
        for i, token_id in enumerate(answer_tokens):
            context = question + " " + generated
            obs_list.append(context)
            actions_list.append(token_id)
            dones_list.append(i == n - 1)
            # Accumulate generated text (EOS decodes to "" since it's special)
            generated += tokenizer.decode([token_id])

        replay_buffer.add_trajectory(obs_list, actions_list, dones_list)

        if (episode + 1) % 300 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes collected...")
    
    print(f"✓ Replay buffer: {len(replay_buffer)} transitions")
    print()
    
    # Training loop
    print("[PHASE 4] Deep Learning Optimization...")
    epochs = 5000  # More epochs for better learning
    print(f"Training for {epochs} epochs...")
    print("(This is where the AI learns to generalize)")
    print()
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        # Sample batch
        obs_b, action_b, done_b = replay_buffer.sample_batch(
            batch_size=config.training.batch_size,
            seq_len=8,
            device=device
        )
        
        # Training step
        metrics = trainer.train_step(obs_b, action_b, done_b)
        
        # Track progress
        current_loss = metrics['loss']
        loss_history.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
        
        # Save checkpoint every 1000 epochs (SAFETY!)
        if epoch % 1000 == 0:
            temp_checkpoint = f"checkpoint_real_ai_epoch{epoch}.pt"
            try:
                result = orchestrator.save_checkpoint(temp_checkpoint)
                if os.path.exists(temp_checkpoint):
                    file_size = os.path.getsize(temp_checkpoint)
                    print(f"  💾 Safety checkpoint saved: {temp_checkpoint} ({file_size / 1024 / 1024:.1f} MB)")
                else:
                    print(f"  ⚠️  WARNING: Checkpoint save claimed success but file doesn't exist!")
            except Exception as e:
                print(f"  ❌ ERROR saving checkpoint: {e}")
        
        # Print progress
        if epoch % 200 == 0 or epoch == 1:
            avg_recent = sum(loss_history[-100:]) / len(loss_history[-100:]) if len(loss_history) >= 100 else current_loss
            print(f"Epoch {epoch:04d}/{epochs} | Loss: {current_loss:7.4f} | Avg: {avg_recent:7.4f} | Best: {best_loss:7.4f}")
    
    print()
    print("=" * 80)
    print(" " * 30 + "TRAINING COMPLETE!")
    print("=" * 80)
    print()
    
    # Save checkpoint
    checkpoint_path = "checkpoint_real_ai.pt"
    try:
        result = orchestrator.save_checkpoint(checkpoint_path)
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path)
            print(f"✓ Model saved: {checkpoint_path} ({file_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"❌ ERROR: Checkpoint save claimed success but file doesn't exist!")
            print(f"   Trying alternative save method...")
            # Try direct torch.save as fallback
            torch.save({
                "model_state": orchestrator.state_dict(),
                "config": orchestrator.config,
                "step": orchestrator.step_counter
            }, checkpoint_path)
            if os.path.exists(checkpoint_path):
                print(f"✓ Alternative save method worked!")
            else:
                print(f"❌ Alternative save also failed!")
    except Exception as e:
        print(f"❌ ERROR saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
    print(f"✓ Final Loss: {current_loss:.4f}")
    print(f"✓ Best Loss: {best_loss:.4f}")
    print(f"✓ Total Training: {num_episodes} episodes, {epochs} epochs")
    print()
    
    # Training summary
    print("=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Dataset Size: {len(dataset)} unique tasks")
    print(f"Episodes Collected: {num_episodes} (each task seen ~{num_episodes/len(dataset):.0f}x)")
    print(f"Optimization Epochs: {epochs}")
    print(f"Total Gradient Updates: {epochs}")
    print(f"Final Loss: {current_loss:.4f}")
    print()
    print("Next Step: Test the AI with:")
    print("  .venv\\Scripts\\python.exe scripts\\test_real_ai.py")
    print("=" * 80)


if __name__ == "__main__":
    run_real_training()
